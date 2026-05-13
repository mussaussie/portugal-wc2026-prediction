"""Phase 6 runner — XGBoost / Logistic Regression model + Monte Carlo simulation."""
import sys, warnings, time
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from src.simulation import load_groups, GROUPS, _best_thirds, _build_bracket, _rank_group
from src.tournament_rules import QF_PAIRS, R16_PAIRS, SF_PAIRS, play_official_knockout_round
from src.team_names import norm as _norm

Path('models').mkdir(exist_ok=True)
Path('simulation').mkdir(exist_ok=True)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
matches = pd.read_csv('data/processed/matches_clean.csv', parse_dates=['date'])
team_tf = pd.read_csv('data/processed/team_features.csv')
groups  = pd.read_csv('data/raw/wc_2026_groups.csv')
groups['team'] = groups['team'].apply(_norm)
print(f"  {len(matches):,} matches | {len(team_tf)} teams | {len(groups)} WC teams")

# ── 2. Feature matrix ──────────────────────────────────────────────────────────
print("Building feature matrix...")
PRESTIGE = {
    'WC': 3, 'Euro': 2, 'CopaAmerica': 2, 'AFCON': 2,
    'AsianCup': 2, 'GoldCup': 2, 'WCQ': 1, 'NationsLeague': 1,
    'EuroQ': 1, 'CopaAmericaQ': 1, 'AFCONQ': 1, 'AsianCupQ': 1,
    'GoldCupQ': 1, 'NationsLeagueQ': 0, 'OtherQ': 0,
    'Friendly': 0, 'Other': 0,
}
df = matches.copy()
df['tournament_prestige'] = df['tournament_type'].map(PRESTIGE).fillna(0).astype(int)
df['avg_scored_diff']     = df['home_avg_scored']   - df['away_avg_scored']
df['avg_conceded_diff']   = df['home_avg_conceded'] - df['away_avg_conceded']
df['form_diff']           = df['home_form']         - df['away_form']

FEATURES = [
    'elo_diff', 'avg_scored_diff', 'avg_conceded_diff', 'form_diff',
    'h2h_home_win_rate', 'neutral_ground', 'tournament_prestige',
]

# Time-based split: train on ≤2017, backtest on WC 2018 and WC 2022.
# Imputer is fit on TRAINING data only to prevent leakage into holdout sets.
train_mask = df['date'].dt.year <= 2017
wc18_mask  = (df['tournament_type'] == 'WC') & (df['date'].dt.year == 2018)
wc22_mask  = (df['tournament_type'] == 'WC') & (df['date'].dt.year == 2022)

imputer_eval = SimpleImputer(strategy='median')
imputer_eval.fit(df.loc[train_mask, FEATURES])          # fit on train only

X_train = pd.DataFrame(imputer_eval.transform(df.loc[train_mask, FEATURES]), columns=FEATURES)
X_wc18  = pd.DataFrame(imputer_eval.transform(df.loc[wc18_mask,  FEATURES]), columns=FEATURES)
X_wc22  = pd.DataFrame(imputer_eval.transform(df.loc[wc22_mask,  FEATURES]), columns=FEATURES)

le = LabelEncoder()
y_all   = le.fit_transform(df['result'])
y_train = y_all[train_mask]
y_wc18  = y_all[wc18_mask]
y_wc22  = y_all[wc22_mask]
print(f"  Train: {len(X_train):,} | WC2018: {len(X_wc18)} | WC2022: {len(X_wc22)}")

# ── 3. Train evaluation models (train ≤ 2017 only) ────────────────────────────
def evaluate(name, model, Xt_18, yt_18, Xt_22, yt_22):
    rows = []
    for label, Xt, yt in [('WC 2018', Xt_18, yt_18), ('WC 2022', Xt_22, yt_22)]:
        if len(Xt) == 0:
            rows.append({'Model': name, 'Dataset': label,
                         'Accuracy': None, 'Log-loss': None, 'Brier': None})
            continue
        probs = model.predict_proba(Xt)
        preds = model.predict(Xt)
        brier = np.mean([brier_score_loss((yt == c).astype(int), probs[:, c]) for c in range(3)])
        rows.append({
            'Model': name, 'Dataset': label,
            'Accuracy': round(accuracy_score(yt, preds), 4),
            'Log-loss': round(log_loss(yt, probs), 4),
            'Brier':    round(brier, 4),
        })
    return pd.DataFrame(rows)

print("Training Logistic Regression (eval)...")
lr_eval = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
lr_eval.fit(X_train, y_train)
print(evaluate('Logistic Regression', lr_eval, X_wc18, y_wc18, X_wc22, y_wc22).to_string(index=False))

print("\nTraining XGBoost (eval)...")
t0 = time.time()
xgb_eval = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.2, reg_lambda=1.5,
    objective='multi:softprob', num_class=3, eval_metric='mlogloss',
    random_state=42, verbosity=0,
)
xgb_eval.fit(X_train, y_train, eval_set=[(X_wc22, y_wc22)], verbose=False)
print(f"  Done in {time.time()-t0:.1f}s")
print(evaluate('XGBoost', xgb_eval, X_wc18, y_wc18, X_wc22, y_wc22).to_string(index=False))

print("\nFeature importance:")
imp = pd.Series(xgb_eval.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(imp.round(4).to_string())

# ── 4. Probability calibration check ──────────────────────────────────────────
# Check if XGBoost probabilities are well-calibrated on WC 2018 holdout.
# We use isotonic calibration on the WC 2018 data and report whether it helps.
print("\nCalibration check (WC 2018)...")
if len(X_wc18) > 5:
    raw_ll_18  = log_loss(y_wc18, xgb_eval.predict_proba(X_wc18))
    cal_xgb = CalibratedClassifierCV(xgb_eval, method='isotonic', cv='prefit')
    cal_xgb.fit(X_wc18, y_wc18)
    cal_ll_18 = log_loss(y_wc18, cal_xgb.predict_proba(X_wc18))
    print(f"  XGBoost raw log-loss (WC18): {raw_ll_18:.4f}")
    print(f"  After isotonic cal (WC18):   {cal_ll_18:.4f} (in-sample — use with caution)")
    print("  Note: calibration fitted on WC18; use XGBoost raw probabilities for production.")

# ── 5. Production models (retrained on ALL available data) ────────────────────
# For the 2026 forecast, retrain on all data to maximise predictive power.
print("\nTraining production models (all data)...")
imputer_prod = SimpleImputer(strategy='median')
X_all = pd.DataFrame(
    imputer_prod.fit_transform(df[FEATURES]), columns=FEATURES
)

xgb_prod = XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.2, reg_lambda=1.5,
    objective='multi:softprob', num_class=3, eval_metric='mlogloss',
    random_state=42, verbosity=0,
)
xgb_prod.fit(X_all, y_all)

lr_prod = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
lr_prod.fit(X_all, y_all)

# Save models
joblib.dump(xgb_prod,     'models/xgboost_model.pkl')
joblib.dump(lr_prod,      'models/logistic_model.pkl')
joblib.dump(le,           'models/label_encoder.pkl')
joblib.dump(imputer_prod, 'models/xgb_imputer.pkl')
print("  Production models saved.")

# ── 6. Pre-compute 48×48 probability matrix (production XGBoost) ──────────────
print("\nPre-computing probability matrix (all WC team pairs)...")
global_medians = {f: imputer_prod.statistics_[i] for i, f in enumerate(FEATURES)}

# H2H rates: use only training-period matches (≤2017) to avoid test-data leakage
# in the production probability matrix used for simulation.
h2h_train = matches[matches['date'].dt.year <= 2017]
h2h_dict  = {}
for (ht, at), grp in h2h_train.groupby(['home_team', 'away_team'])['result']:
    h2h_dict[(ht, at)] = (grp == 'W').mean()

tf = team_tf.set_index('team')

def get_feats(t):
    t = _norm(t)
    if t in tf.index:
        r = tf.loc[t]
        return (
            r.get('elo_rating', np.nan),
            r.get('avg_goals_scored', np.nan),
            r.get('avg_goals_conceded', np.nan),
            r.get('form_index', np.nan),
        )
    return np.nan, np.nan, np.nan, np.nan

def build_row(ta, tb):
    ea, sa, ca, fa = get_feats(ta)
    eb, sb, cb, fb = get_feats(tb)
    ed = (ea - eb) if not (np.isnan(ea) or np.isnan(eb)) else global_medians['elo_diff']
    sd = (sa - sb) if not (np.isnan(sa) or np.isnan(sb)) else global_medians['avg_scored_diff']
    cd = (ca - cb) if not (np.isnan(ca) or np.isnan(cb)) else global_medians['avg_conceded_diff']
    fd = (fa - fb) if not (np.isnan(fa) or np.isnan(fb)) else global_medians['form_diff']
    h  = h2h_dict.get((ta, tb), 0.45)
    return [ed, sd, cd, fd, h, 1, 3]  # neutral_ground=1, prestige=WC(3)

wc_teams = groups['team'].tolist()
n        = len(wc_teams)
tidx     = {t: i for i, t in enumerate(wc_teams)}

idx_W = int(np.where(le.classes_ == 'W')[0])
idx_D = int(np.where(le.classes_ == 'D')[0])
idx_L = int(np.where(le.classes_ == 'L')[0])

t0    = time.time()
pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
X_p   = np.array([build_row(wc_teams[i], wc_teams[j]) for i, j in pairs])
probs = xgb_prod.predict_proba(X_p)

pm = np.zeros((n, n, 3))
for k, (i, j) in enumerate(pairs):
    pm[i, j, 0] = probs[k, idx_W]
    pm[i, j, 1] = probs[k, idx_D]
    pm[i, j, 2] = probs[k, idx_L]

print(f"  Done in {time.time()-t0:.2f}s")

pi = tidx.get('Portugal')
ci = tidx.get('Colombia')
si = tidx.get('Spain')
if pi is not None and ci is not None:
    pw, pd_, pl = pm[pi, ci]
    print(f"  Portugal vs Colombia: Win={pw:.1%}  Draw={pd_:.1%}  Loss={pl:.1%}")
if pi is not None and si is not None:
    pw, pd_, pl = pm[pi, si]
    print(f"  Portugal vs Spain:    Win={pw:.1%}  Draw={pd_:.1%}  Loss={pl:.1%}")

# ── 7. Monte Carlo simulation ──────────────────────────────────────────────────
#
# Group-stage goal simulation:
#   Outcome (W/D/L) is sampled from XGBoost class probabilities.
#   Goal totals (for GD/GF tiebreakers) use empirical per-outcome Poisson rates:
#     Win  → winner ~Poisson(1.7), loser ~Poisson(0.7)
#     Draw → both teams ~Poisson(1.0), then set equal (ga = gb)
#   Inconsistencies (e.g., Poisson gives wrong direction) are corrected by +1.
#   This approach is consistent with classifier output and avoids arbitrary formulas.
GOAL_RATE_WIN  = (1.7, 0.7)   # (winner, loser) mean goals when win outcome
GOAL_RATE_DRAW = (1.0, 1.0)   # both teams when draw outcome

def sim_group(indices, rng):
    """
    Simulate one 4-team group round-robin.
    Returns (sorted_indices, stats_dict) where stats_dict maps team_idx → {pts,gd,gf}.
    """
    stats = {ti: {'pts': 0, 'gd': 0, 'gf': 0} for ti in indices}
    matches = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            p = pm[i, j]  # [p_win_i, p_draw, p_loss_i]
            p = p / p.sum()   # renormalise for floating-point safety

            # Sample outcome from XGBoost probabilities
            outcome = rng.choice(3, p=p)   # 0 = i wins, 1 = draw, 2 = j wins

            if outcome == 0:       # i wins
                la, lb = GOAL_RATE_WIN
                stats[i]['pts'] += 3
            elif outcome == 1:     # draw
                la, lb = GOAL_RATE_DRAW
                stats[i]['pts'] += 1
                stats[j]['pts'] += 1
            else:                  # j wins
                la, lb = GOAL_RATE_WIN   # lb is j's rate (winner), la is i's (loser)
                la, lb = lb, la          # swap so la = i's (loser), lb = j's (winner)
                stats[j]['pts'] += 3

            ga = int(rng.poisson(la))
            gb = int(rng.poisson(lb))

            # Force consistency with sampled outcome
            if outcome == 1:
                gb = ga                  # draw: equalise goals
            elif outcome == 0 and ga <= gb:
                ga = gb + 1              # i must win
            elif outcome == 2 and gb <= ga:
                gb = ga + 1              # j must win

            stats[i]['gf'] += ga;  stats[j]['gf'] += gb
            stats[i]['gd'] += ga - gb;  stats[j]['gd'] += gb - ga
            matches.append((i, j, ga, gb))

    ranked = _rank_group(indices, stats, matches, rng)
    sorted_idx = [r['team'] for r in ranked]
    return sorted_idx, stats


def play(i, j, rng):
    """Knockout match: winner progresses.  Draw goes to penalties (50/50 + ELO edge)."""
    p = pm[i, j]
    p = p / p.sum()   # renormalise for floating-point safety
    # Winner probability = p_win + 0.5 * p_draw (symmetric penalty toss).
    return i if rng.random() < (p[0] + p[1] * 0.5) else j


groups_df = load_groups()
group_indices = {}
for grp in GROUPS:
    raw = groups_df[groups_df['group'] == grp]['team'].tolist()
    group_indices[grp] = [tidx[t] for t in raw if t in tidx]
    missing = [t for t in raw if t not in tidx]
    if missing:
        print(f"  WARNING: teams not in probability matrix for group {grp}: {missing}")

TARGET = 'Portugal'
TI     = tidx.get(TARGET)
N      = 100_000
rng    = np.random.default_rng(42)

win_counts   = np.zeros(n, dtype=int)
stage_counts = {
    'Group stage': 0, 'Round of 32': 0, 'Round of 16': 0,
    'Quarter-Final': 0, 'Semi-Final': 0, 'Runner-up': 0, 'Winner': 0,
}

print(f"\nRunning {N:,} simulations...")
t0 = time.time()

def play_idx(a_idx, b_idx):
    """Knockout match between two team indices. Handles sentinel -1 gracefully."""
    if a_idx < 0 or b_idx < 0:
        return a_idx if b_idx < 0 else b_idx
    return play(a_idx, b_idx, rng)

for sim_i in range(N):
    if sim_i > 0 and sim_i % 25_000 == 0:
        elapsed = time.time() - t0
        print(f"  {sim_i:,} done — {elapsed:.0f}s elapsed, "
              f"~{elapsed/sim_i*(N-sim_i):.0f}s remaining")

    # ── Group stage ────────────────────────────────────────────────────────────
    gr_results = {grp: sim_group(group_indices[grp], rng) for grp in GROUPS}

    # Build gr_named with ACTUAL simulated pts/gd/gf (fixes best-thirds selection bug)
    gr_named = {}
    for grp, (sorted_idx, stats) in gr_results.items():
        gr_named[grp] = [
            {'team': wc_teams[ti],
             'pts':  stats[ti]['pts'],
             'gd':   stats[ti]['gd'],
             'gf':   stats[ti]['gf']}
            for ti in sorted_idx
        ]

    thirds  = _best_thirds(gr_named)
    bracket = _build_bracket(gr_named, thirds)
    bidx    = [tidx.get(t, -1) for t in bracket]

    # Portugal qualification check (for path tracking)
    qual = {wc_teams[ti] for st, _ in gr_results.values() for ti in st[:2]}
    qual.update(thirds)
    port_stage = 'Group stage' if TARGET not in qual else None

    # ── Full knockout bracket (always run to determine the champion) ───────────
    r32 = [play_idx(bidx[2*k], bidx[2*k+1]) for k in range(16)]
    if port_stage is None and TI not in r32:
        port_stage = 'Round of 32'

    r16 = play_official_knockout_round(r32, R16_PAIRS, lambda a, b: play_idx(a, b))
    if port_stage is None and TI not in r16:
        port_stage = 'Round of 16'

    qf = play_official_knockout_round(r16, QF_PAIRS, lambda a, b: play_idx(a, b))
    if port_stage is None and TI not in qf:
        port_stage = 'Quarter-Final'

    sf = play_official_knockout_round(qf, SF_PAIRS, lambda a, b: play_idx(a, b))
    if port_stage is None and TI not in sf:
        port_stage = 'Semi-Final'

    champ = play(sf[0], sf[1], rng)
    win_counts[champ] += 1   # always track champion regardless of Portugal's path

    if port_stage is None:
        port_stage = 'Winner' if champ == TI else 'Runner-up'

    stage_counts[port_stage] += 1

elapsed = time.time() - t0
print(f"Simulations complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ── 8. Results ─────────────────────────────────────────────────────────────────
results = (
    pd.DataFrame({'winner': wc_teams, 'count': win_counts, 'probability': win_counts / N})
    .query('count > 0')
    .sort_values('probability', ascending=False)
    .reset_index(drop=True)
)
results.to_csv('simulation/xgboost_simulation_results.csv', index=False)

port_prob = results[results['winner'] == TARGET]['probability'].values
port_prob = port_prob[0] if len(port_prob) else 0.0
total_sim = sum(stage_counts.values())

print()
print("Top 15 champions (XGBoost model, production — all data):")
print(results.head(15).to_string(index=False))
print()
print(f"=== Portugal win probability (XGBoost): {port_prob:.2%} ===")
print()
print("Portugal stage-by-stage:")
for s, c in stage_counts.items():
    print(f"  {s:<15}: {c/N:.1%}")
print(f"  {'TOTAL':<15}: {total_sim/N:.1%}  (should be 100%)")
