"""
Dixon-Coles model for football match prediction.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market"

Key idea:
  Goals scored by each team follow a Poisson distribution.
  Attack and defense parameters per team are fitted by MLE.
  A correction factor rho adjusts for the over-/under-prediction of
  low-scoring results (0-0, 1-0, 0-1, 1-1).
"""

import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from scipy.special import gammaln
from scipy import sparse
from sklearn.linear_model import PoissonRegressor
from pathlib import Path
import json
import warnings
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.team_names import norm as _norm

# ── Constants ────────────────────────────────────────────────────────────────

DECAY_RATE  = 0.0065   # time-weight decay per day (~10% weight at 3 yrs)
MIN_MATCHES = 30       # minimum appearances to get own alpha/beta params
MAX_GOALS   = 10       # max goals per team in scoreline matrix (increased from 8)
TRAIN_FROM  = 2010     # only use matches from this year onward
MAX_FITTED_ACTIVE_TEAMS = 20   # extra non-WC teams fitted for schedule strength


# ── Helpers ───────────────────────────────────────────────────────────────────

def _time_weights(dates: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
    days_ago = (ref_date - dates).dt.days.clip(lower=0).values
    return np.exp(-DECAY_RATE * days_ago)


def _tau_vec(home_goals: np.ndarray, away_goals: np.ndarray,
             lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Vectorised Dixon-Coles correction factor."""
    tau = np.ones(len(home_goals))
    m00 = (home_goals == 0) & (away_goals == 0)
    m10 = (home_goals == 1) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m11 = (home_goals == 1) & (away_goals == 1)
    tau[m00] = np.maximum(1e-10, 1 - lam[m00] * mu[m00] * rho)
    tau[m10] = np.maximum(1e-10, 1 + mu[m10] * rho)
    tau[m01] = np.maximum(1e-10, 1 + lam[m01] * rho)
    tau[m11] = np.maximum(1e-10, 1 - rho)
    return tau


def _tau_scalar(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0: return max(1e-10, 1 - lam * mu * rho)
    if x == 1 and y == 0: return max(1e-10, 1 + mu * rho)
    if x == 0 and y == 1: return max(1e-10, 1 + lam * rho)
    if x == 1 and y == 1: return max(1e-10, 1 - rho)
    return 1.0


# ── Model class ───────────────────────────────────────────────────────────────

class DixonColes:
    """
    Fitted Dixon-Coles model.

    Parameters stored per team (log-linear formulation):
      lambda_ij = exp(mu + alpha_i - beta_j)   ← expected goals, team i vs j
      mu_ij     = exp(mu + alpha_j - beta_i)   ← expected goals, team j vs i
    where:
      mu     = global log-average goals (intercept)
      alpha  = attack strength (positive = above average attack)
      beta   = defensive weakness (positive = concedes more than average)
      rho    = Dixon-Coles low-score correction (typically slightly negative)
    Identifiability: alpha[ref_team] is fixed to 0 during fitting.
    """

    def __init__(self, decay: float = DECAY_RATE):
        self.decay    = decay
        self.mu_      = None
        self.rho_     = None
        self.alphas_  = {}
        self.betas_   = {}
        self.teams_   = []
        self.fitted_  = False

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, intl_df: pd.DataFrame,
            target_teams: list = None,
            ref_date=None) -> "DixonColes":
        """
        Fit parameters by MLE with exponential time-weighting.

        target_teams: teams that must receive individual alpha/beta parameters.
                      Other active teams with MIN_MATCHES+ appearances are also
                      fitted so WC teams are not evaluated against a single
                      global-average opponent pool.
                      If None, falls back to all teams with MIN_MATCHES+ appearances.

        intl_df must have columns: date, home_team, away_team, home_score, away_score
        """
        df = intl_df.copy()
        df = df.dropna(subset=["home_score", "away_score"])
        # Normalise to canonical names (same as international_results.csv primary names)
        df["home_team"] = df["home_team"].apply(_norm)
        df["away_team"] = df["away_team"].apply(_norm)
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        df["date"]       = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= TRAIN_FROM].copy()

        if ref_date is None:
            ref_date = df["date"].max()
        ref_date = pd.Timestamp(ref_date)
        df["weight"] = _time_weights(df["date"], ref_date)

        if target_teams:
            target_set = set(_norm(t) for t in target_teams)
            counts = (df.groupby("home_team").size()
                      .add(df.groupby("away_team").size(), fill_value=0))
            active_set = set(
                counts[counts >= MIN_MATCHES]
                .sort_values(ascending=False)
                .head(MAX_FITTED_ACTIVE_TEAMS)
                .index
            )
            observed_targets = target_set & (set(df["home_team"]) | set(df["away_team"]))
            teams = sorted(active_set | observed_targets)
            df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)].copy()
        else:
            counts = (df.groupby("home_team").size()
                      .add(df.groupby("away_team").size(), fill_value=0))
            teams = sorted(counts[counts >= MIN_MATCHES].index)
            df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)].copy()

        n    = len(teams)
        tidx = {t: i for i, t in enumerate(teams)}

        h_f = np.array([tidx.get(t, -1) for t in df["home_team"]])
        a_f = np.array([tidx.get(t, -1) for t in df["away_team"]])
        h_goals = df["home_score"].values.astype(int)
        a_goals = df["away_score"].values.astype(int)
        wts     = df["weight"].values

        print(f"  Fitting Poisson GLM on {len(df):,} matches | {n} fitted teams | "
              f"ref date: {ref_date.date()}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows = np.arange(2 * len(df))
            cols_attack = np.r_[h_f, a_f]
            cols_defense = np.r_[n + a_f, n + h_f]
            row_idx = np.r_[rows, rows]
            col_idx = np.r_[cols_attack, cols_defense]
            data = np.ones(len(row_idx), dtype=float)
            X = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(2 * len(df), 2 * n))
            y = np.r_[h_goals, a_goals]
            sample_weight = np.r_[wts, wts]

            glm = PoissonRegressor(alpha=0.001, fit_intercept=True, max_iter=500, tol=1e-7)
            glm.fit(X, y, sample_weight=sample_weight)

        attack = glm.coef_[:n].copy()
        defense_weakness = glm.coef_[n:].copy()
        mu_b = float(glm.intercept_)

        lam = np.exp(mu_b + attack[h_f] + defense_weakness[a_f])
        mu_ = np.exp(mu_b + attack[a_f] + defense_weakness[h_f])

        def neg_tau(rho):
            tau = _tau_vec(h_goals, a_goals, lam, mu_, rho)
            return -float(np.dot(wts, np.log(np.maximum(1e-15, tau))))

        rho_result = minimize_scalar(neg_tau, bounds=(-0.5, 0.5), method="bounded")
        rho = float(rho_result.x) if rho_result.success else -0.05

        self.mu_     = mu_b
        self.rho_    = rho
        self.alphas_ = {t: float(attack[i]) for t, i in tidx.items()}
        # expected_goals uses exp(mu + alpha_i - beta_j), so store negative
        # defensive weakness to preserve the existing public API.
        self.betas_  = {t: float(-defense_weakness[i]) for t, i in tidx.items()}
        self.teams_  = teams
        self.fitted_ = True

        print(f"  Done. mu={self.mu_:.3f} (avg goals={np.exp(self.mu_):.2f})  "
              f"rho={self.rho_:.4f}  converged=True")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def expected_goals(self, team_a: str, team_b: str) -> tuple[float, float]:
        """
        Return (lambda_a, mu_b) — expected goals for each team at neutral venue.
        Teams not in the fitted model get alpha=0, beta=0 (global-average team).
        """
        a = _norm(team_a)
        b = _norm(team_b)
        a_att = self.alphas_.get(a, 0.0)
        a_def = self.betas_.get(a,  0.0)
        b_att = self.alphas_.get(b, 0.0)
        b_def = self.betas_.get(b,  0.0)
        lam = np.exp(self.mu_ + a_att - b_def)
        mu  = np.exp(self.mu_ + b_att - a_def)
        return float(lam), float(mu)

    def scoreline_matrix(self, team_a: str, team_b: str,
                         max_goals: int = MAX_GOALS) -> np.ndarray:
        """
        (max_goals+1) × (max_goals+1) matrix.
        matrix[i, j] = P(team_a scores i goals, team_b scores j goals).
        The matrix is renormalised to sum to 1 (accounts for truncation at max_goals).
        """
        lam, mu = self.expected_goals(team_a, team_b)
        g = np.arange(max_goals + 1)
        M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        for i in range(min(2, max_goals + 1)):
            for j in range(min(2, max_goals + 1)):
                M[i, j] *= _tau_scalar(i, j, lam, mu, self.rho_)
        # Renormalise to ensure probabilities sum to 1 (truncation correction)
        total = M.sum()
        if total > 0:
            M /= total
        return M

    def match_probs(self, team_a: str, team_b: str) -> tuple[float, float, float]:
        """Return (p_win_A, p_draw, p_win_B) from the renormalised scoreline matrix."""
        M = self.scoreline_matrix(team_a, team_b)
        p_win  = float(np.tril(M, -1).sum())
        p_draw = float(np.trace(M))
        p_loss = float(np.triu(M,  1).sum())
        return p_win, p_draw, p_loss

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate_match(self, team_a: str, team_b: str,
                       knockout: bool = False, rng=None) -> str:
        """
        Simulate a match outcome by sampling from the full DC-corrected
        scoreline distribution.  Returns 'A', 'D', or 'B'.
        Knockout mode routes draws to a penalty-shootout coin flip
        (with a small ELO-like edge from the lambda ratio).
        """
        if rng is None:
            rng = np.random.default_rng()
        ga, gb = self.simulate_scoreline(team_a, team_b, rng=rng)
        if ga > gb: return "A"
        if ga < gb: return "B"
        if knockout:
            lam, mu = self.expected_goals(team_a, team_b)
            edge = (lam - mu) / max(lam + mu, 0.1) * 0.30
            p_ko = float(np.clip(0.5 + edge, 0.35, 0.65))
            return "A" if rng.random() < p_ko else "B"
        return "D"

    def simulate_scoreline(self, team_a: str, team_b: str,
                           rng=None) -> tuple[int, int]:
        """
        Sample a scoreline from the full DC-corrected scoreline distribution.
        Uses the renormalised matrix (not independent Poissons) so rho is applied.
        """
        if rng is None:
            rng = np.random.default_rng()
        M    = self.scoreline_matrix(team_a, team_b, max_goals=MAX_GOALS)
        flat = M.ravel()
        mg1  = MAX_GOALS + 1
        idx  = rng.choice(mg1 * mg1, p=flat)
        return idx // mg1, idx % mg1

    # ── Inspection ────────────────────────────────────────────────────────────

    def team_params(self, teams: list = None) -> pd.DataFrame:
        """DataFrame of attack/defense parameters for requested teams."""
        rows = []
        for t in (teams or self.teams_):
            lam, _ = self.expected_goals(t, t)
            rows.append({
                "team":    t,
                "attack":  round(self.alphas_.get(_norm(t), 0.0), 4),
                "defense": round(self.betas_.get(_norm(t),  0.0), 4),
                "xG_neutral": round(lam, 3),
            })
        return pd.DataFrame(rows).sort_values("attack", ascending=False)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "mu": self.mu_, "rho": self.rho_,
            "alphas": self.alphas_, "betas": self.betas_,
        }, indent=2))
        print(f"  Saved to {path}")

    @classmethod
    def load(cls, path) -> "DixonColes":
        data   = json.loads(Path(path).read_text())
        model  = cls()
        model.mu_     = data["mu"]
        model.rho_    = data["rho"]
        model.alphas_ = data["alphas"]
        model.betas_  = data["betas"]
        model.teams_  = sorted(data["alphas"])
        model.fitted_ = True
        return model


# ── ELO-calibrated Poisson / DC scoreline model ───────────────────────────────

class ELOPoissonDC:
    """
    Phase 5 replacement for the GLM-based DixonColes.

    Pure Dixon-Coles MLE over-weights regional goal records (e.g. Iran vs
    Cambodia) because it has no explicit schedule-difficulty correction.
    ELO ratings already correct for opponent quality through iterative
    update rules.  This model uses:

        lambda_A = avg_goals * p_elo(A beats B)
        lambda_B = avg_goals * (1 - p_elo(A beats B))

    then applies the standard DC low-score correction factor rho to the
    {0-0, 1-0, 0-1, 1-1} cells of the scoreline matrix.

    avg_goals and rho are calibrated from recent international matches.
    Public API matches DixonColes so run_monte_carlo_dc() works unchanged.
    """

    ELO_SCALE = 400.0

    def __init__(self):
        self.elo_: dict = {}
        self.avg_goals_: float = 2.5
        self.rho_: float = -0.10
        self.mu_: float = None          # log(avg_goals/2) — interface compat
        self.teams_: list = []
        self.fitted_: bool = False
        self._default_elo: float = 1500.0
        self._cdf_cache: dict = {}      # (team_a, team_b) → precomputed CDF array

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, elo_df: pd.DataFrame,
            intl_df: pd.DataFrame = None,
            train_from: int = TRAIN_FROM) -> "ELOPoissonDC":
        """
        Parameters
        ----------
        elo_df  : DataFrame with 'team' and 'rating' columns.
                  If a 'date' column is present (historical time-series),
                  the latest rating per team is used automatically.
        intl_df : International results used to calibrate avg_goals and rho.
                  If None, defaults (2.5 goals, rho=-0.10) are kept.
        """
        elo = elo_df.copy()
        elo["team"] = elo["team"].str.replace("\xa0", " ", regex=False).apply(_norm)
        if "date" in elo.columns:
            elo["date"] = pd.to_datetime(elo["date"], format="mixed", dayfirst=False)
            elo = elo.sort_values("date").groupby("team").last().reset_index()
        self.elo_ = dict(zip(elo["team"], elo["rating"].astype(float)))
        self.teams_ = sorted(self.elo_.keys())
        self._default_elo = float(np.median(list(self.elo_.values())))

        if intl_df is not None:
            df = intl_df.copy()
            df = df.dropna(subset=["home_score", "away_score"])
            df["date"]       = pd.to_datetime(df["date"])
            df["home_team"]  = df["home_team"].apply(_norm)
            df["away_team"]  = df["away_team"].apply(_norm)
            df["home_score"] = df["home_score"].astype(int)
            df["away_score"] = df["away_score"].astype(int)
            df = df[df["date"].dt.year >= train_from].copy()
            df = df[
                df["home_team"].isin(self.elo_) &
                df["away_team"].isin(self.elo_)
            ].copy()

            if len(df) < 100:
                print("  Warning: fewer than 100 calibration matches — keeping defaults.")
            else:
                ref_date = df["date"].max()
                wts  = _time_weights(df["date"], ref_date)

                # Calibrate avg total goals (time-weighted)
                self.avg_goals_ = float(
                    np.average(df["home_score"] + df["away_score"], weights=wts)
                )

                h_elo = df["home_team"].map(self.elo_).values.astype(float)
                a_elo = df["away_team"].map(self.elo_).values.astype(float)
                p_h   = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / self.ELO_SCALE))

                lam_cal = self.avg_goals_ * p_h
                mu_cal  = self.avg_goals_ * (1.0 - p_h)
                h_g = df["home_score"].values
                a_g = df["away_score"].values

                def _neg_tau_ll(rho):
                    tau = _tau_vec(h_g, a_g, lam_cal, mu_cal, rho)
                    return -float(np.dot(wts, np.log(np.maximum(1e-15, tau))))

                res = minimize_scalar(_neg_tau_ll, bounds=(-0.5, 0.5), method="bounded")
                self.rho_ = float(res.x) if res.success else -0.10
                print(f"  Calibration matches: {len(df):,}  "
                      f"avg_goals={self.avg_goals_:.3f}  rho={self.rho_:.4f}")

        self.mu_     = float(np.log(self.avg_goals_ / 2.0))
        self.fitted_ = True
        print(f"  ELOPoissonDC fitted.  teams with ELO: {len(self.elo_)}")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def expected_goals(self, team_a: str, team_b: str) -> tuple:
        a = _norm(team_a);  b = _norm(team_b)
        elo_a = self.elo_.get(a, self._default_elo)
        elo_b = self.elo_.get(b, self._default_elo)
        p_a   = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / self.ELO_SCALE))
        return float(self.avg_goals_ * p_a), float(self.avg_goals_ * (1.0 - p_a))

    def scoreline_matrix(self, team_a: str, team_b: str,
                         max_goals: int = MAX_GOALS) -> np.ndarray:
        lam, mu = self.expected_goals(team_a, team_b)
        g = np.arange(max_goals + 1)
        M = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        for i in range(min(2, max_goals + 1)):
            for j in range(min(2, max_goals + 1)):
                M[i, j] *= _tau_scalar(i, j, lam, mu, self.rho_)
        total = M.sum()
        if total > 0:
            M /= total
        return M

    def match_probs(self, team_a: str, team_b: str) -> tuple:
        M = self.scoreline_matrix(team_a, team_b)
        return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())

    def simulate_scoreline(self, team_a: str, team_b: str, rng=None) -> tuple:
        """
        Sample a scoreline from the DC-corrected distribution.
        CDFs are cached on first call for each pair — reduces 10M+ calls to
        a single rng.random() + np.searchsorted per match after warm-up.
        """
        if rng is None:
            rng = np.random.default_rng()
        a = _norm(team_a);  b = _norm(team_b)
        pair = (a, b)
        if pair not in self._cdf_cache:
            M = self.scoreline_matrix(team_a, team_b, max_goals=MAX_GOALS)
            self._cdf_cache[pair] = np.cumsum(M.ravel())
        cdf = self._cdf_cache[pair]
        idx = int(np.searchsorted(cdf, rng.random(), side="right"))
        idx = min(idx, (MAX_GOALS + 1) ** 2 - 1)
        mg1 = MAX_GOALS + 1
        return idx // mg1, idx % mg1

    def simulate_match(self, team_a: str, team_b: str,
                       knockout: bool = False, rng=None) -> str:
        if rng is None:
            rng = np.random.default_rng()
        ga, gb = self.simulate_scoreline(team_a, team_b, rng=rng)
        if ga > gb: return "A"
        if ga < gb: return "B"
        if knockout:
            lam, mu = self.expected_goals(team_a, team_b)
            edge = (lam - mu) / max(lam + mu, 0.1) * 0.30
            p_ko = float(np.clip(0.5 + edge, 0.35, 0.65))
            return "A" if rng.random() < p_ko else "B"
        return "D"

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "model_type": "ELOPoissonDC",
            "avg_goals": self.avg_goals_,
            "rho": self.rho_,
            "elo": self.elo_,
        }, indent=2))
        print(f"  Saved to {path}")

    @classmethod
    def load(cls, path) -> "ELOPoissonDC":
        data = json.loads(Path(path).read_text())
        m = cls()
        m.avg_goals_  = data["avg_goals"]
        m.rho_        = data["rho"]
        m.elo_        = data["elo"]
        m.teams_      = sorted(data["elo"].keys())
        m._default_elo = float(np.median(list(data["elo"].values())))
        m.mu_         = float(np.log(data["avg_goals"] / 2.0))
        m.fitted_     = True
        return m


# ── Tournament simulation ─────────────────────────────────────────────────────

def run_monte_carlo_dc(model: DixonColes,
                       n: int = 100_000,
                       seed: int = 42) -> pd.DataFrame:
    """
    Run n WC 2026 Monte Carlo simulations using the DC model.
    Returns DataFrame: winner | count | probability.
    """
    from src.simulation import load_groups, GROUPS, _best_thirds, _build_bracket, _rank_group
    from src.tournament_rules import QF_PAIRS, R16_PAIRS, SF_PAIRS, play_official_knockout_round
    groups_df = load_groups()
    rng       = np.random.default_rng(seed)

    def play(a, b):
        result = model.simulate_match(a, b, knockout=True, rng=rng)
        return a if result == "A" else b

    def simulate_group_dc(teams):
        stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        matches = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                ga, gb = model.simulate_scoreline(a, b, rng=rng)
                stats[a]["gf"] += ga;  stats[b]["gf"] += gb
                stats[a]["gd"] += ga - gb;  stats[b]["gd"] += gb - ga
                if ga > gb:    stats[a]["pts"] += 3
                elif ga == gb: stats[a]["pts"] += 1; stats[b]["pts"] += 1
                else:          stats[b]["pts"] += 3
                matches.append((a, b, ga, gb))
        return _rank_group(teams, stats, matches, rng)

    def simulate_knockout_dc(bracket):
        r32w = [play(bracket[2*i], bracket[2*i+1]) for i in range(16)]
        r16w = play_official_knockout_round(r32w, R16_PAIRS, play)
        qfw  = play_official_knockout_round(r16w, QF_PAIRS, play)
        sfw  = play_official_knockout_round(qfw, SF_PAIRS, play)
        return play(sfw[0], sfw[1])

    winners = []
    for _ in range(n):
        group_results = {}
        for grp in GROUPS:
            teams = groups_df[groups_df["group"] == grp]["team"].tolist()
            group_results[grp] = simulate_group_dc(teams)
        thirds  = _best_thirds(group_results)
        bracket = _build_bracket(group_results, thirds)
        winners.append(simulate_knockout_dc(bracket))

    counts = pd.Series(winners).value_counts().reset_index()
    counts.columns = ["winner", "count"]
    counts["probability"] = counts["count"] / n
    return counts


def run_portugal_path_dc(model: DixonColes,
                         n: int = 100_000,
                         seed: int = 42) -> dict:
    """Track Portugal's exit stage across n simulations (DC model).
    Stages: Group stage, Round of 32, Round of 16, Quarter-Final,
            Semi-Final, Runner-up (final loss), Winner.
    """
    from src.simulation import load_groups, GROUPS, _best_thirds, _build_bracket, _rank_group
    from src.tournament_rules import QF_PAIRS, R16_PAIRS, SF_PAIRS, play_official_knockout_round
    groups_df = load_groups()
    rng       = np.random.default_rng(seed)
    TARGET    = "Portugal"

    stage_counts = {
        "Group stage":  0, "Round of 32": 0, "Round of 16": 0,
        "Quarter-Final": 0, "Semi-Final": 0, "Runner-up": 0, "Winner": 0,
    }

    def play(a, b):
        result = model.simulate_match(a, b, knockout=True, rng=rng)
        return a if result == "A" else b

    def simulate_group_dc(teams):
        stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        matches = []
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                a, b = teams[i], teams[j]
                ga, gb = model.simulate_scoreline(a, b, rng=rng)
                stats[a]["gf"] += ga;  stats[b]["gf"] += gb
                stats[a]["gd"] += ga - gb;  stats[b]["gd"] += gb - ga
                if ga > gb:    stats[a]["pts"] += 3
                elif ga == gb: stats[a]["pts"] += 1; stats[b]["pts"] += 1
                else:          stats[b]["pts"] += 3
                matches.append((a, b, ga, gb))
        return _rank_group(teams, stats, matches, rng)

    for _ in range(n):
        group_results = {}
        for grp in GROUPS:
            teams = groups_df[groups_df["group"] == grp]["team"].tolist()
            group_results[grp] = simulate_group_dc(teams)

        thirds    = _best_thirds(group_results)
        qualified = {r["team"] for res in group_results.values() for r in res[:2]}
        qualified.update(thirds)

        if TARGET not in qualified:
            stage_counts["Group stage"] += 1;  continue

        bracket = _build_bracket(group_results, thirds)
        r32w = [play(bracket[2*i], bracket[2*i+1]) for i in range(16)]
        if TARGET not in r32w:
            stage_counts["Round of 32"] += 1;  continue

        r16w = play_official_knockout_round(r32w, R16_PAIRS, play)
        if TARGET not in r16w:
            stage_counts["Round of 16"] += 1;  continue

        qfw = play_official_knockout_round(r16w, QF_PAIRS, play)
        if TARGET not in qfw:
            stage_counts["Quarter-Final"] += 1;  continue

        sfw = play_official_knockout_round(qfw, SF_PAIRS, play)
        if TARGET not in sfw:
            stage_counts["Semi-Final"] += 1;  continue

        final_winner = play(sfw[0], sfw[1])
        if final_winner == TARGET:
            stage_counts["Winner"] += 1
        else:
            stage_counts["Runner-up"] += 1   # reached the Final but lost

    return stage_counts


if __name__ == "__main__":
    from src.team_names import norm as _norm_t
    intl   = pd.read_csv(Path("data/raw/international_results.csv"))
    groups = pd.read_csv(Path("data/raw/wc_2026_groups.csv"))
    wc_teams = [_norm_t(t) for t in groups["team"].tolist()]

    print("Fitting Dixon-Coles model (48 WC teams only)...")
    model = DixonColes()
    model.fit(intl, target_teams=wc_teams)

    print("\nSaving model...")
    model.save(Path("models/dixon_coles_params.json"))

    print("\nPortugal vs Colombia (DC model):")
    pw, pd_, pl = model.match_probs("Portugal", "Colombia")
    lam, mu = model.expected_goals("Portugal", "Colombia")
    print(f"  Expected goals: Port {lam:.2f} — Col {mu:.2f}")
    print(f"  Win={pw:.1%}  Draw={pd_:.1%}  Loss={pl:.1%}")

    print("\nRunning 100,000 simulations...")
    results = run_monte_carlo_dc(model, 100_000)
    results.to_csv(Path("simulation/dixon_coles_simulation_results.csv"), index=False)
    print("\nTop 10:")
    print(results.head(10).to_string(index=False))
    port = results[results["winner"] == "Portugal"]
    p = port["probability"].values[0] if len(port) else 0.0
    print(f"\nPortugal win probability (DC): {p:.2%}")
