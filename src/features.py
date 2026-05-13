"""
Phase 3 — Feature Engineering
Produces:
  data/processed/team_features.csv    current team-level stats (used in simulation)
  data/processed/matches_clean.csv    historical matches with ML-ready features
  data/processed/portugal_path.csv    Portugal's 2026 bracket path

Run: python src/features.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
# Ensure project root is on sys.path when running as a standalone script
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.team_names import norm
from src.tournament_classification import classify_tournament

RAW  = Path("data/raw")
PROC = Path("data/processed")
PROC.mkdir(parents=True, exist_ok=True)

# ── SECTION 1: LOAD RAW DATA ─────────────────────────────────────────────────
print("Loading raw data...")
intl = pd.read_csv(RAW / "international_results.csv")
intl["date"] = pd.to_datetime(intl["date"], format="mixed")

elo_raw = pd.read_csv(RAW / "elo_ratings.csv")
elo_raw["date"] = pd.to_datetime(elo_raw["date"], format="mixed")
# Normalize ELO team names to canonical (results-CSV) names
elo_raw["team"] = elo_raw["team"].apply(norm)

groups = pd.read_csv(RAW / "wc_2026_groups.csv")
groups["team"] = groups["team"].apply(norm)

# Use 1990+ only — modern football is structurally different from pre-WW2 era
intl = intl[intl["date"].dt.year >= 1990].copy()
intl = intl.dropna(subset=["home_score", "away_score"]).reset_index(drop=True)
print(f"  Matches (1990+, clean): {len(intl):,}")


# ── SECTION 2: BUILD TEAM-CENTRIC HISTORY ────────────────────────────────────
print("Building team history...")

home = intl[["date", "home_team", "away_team", "home_score", "away_score", "neutral", "tournament"]].copy()
home.columns = ["date", "team", "opponent", "goals_scored", "goals_conceded", "neutral", "tournament"]
home["is_home"] = True

away = intl[["date", "away_team", "home_team", "away_score", "home_score", "neutral", "tournament"]].copy()
away.columns = ["date", "team", "opponent", "goals_scored", "goals_conceded", "neutral", "tournament"]
away["is_home"] = False

th = pd.concat([home, away]).sort_values(["team", "date"]).reset_index(drop=True)
th["win"]  = (th["goals_scored"] > th["goals_conceded"]).astype(float)
th["draw"] = (th["goals_scored"] == th["goals_conceded"]).astype(float)
print(f"  Team-history rows: {len(th):,}")


# ── SECTION 3: ROLLING FEATURES ──────────────────────────────────────────────
print("Calculating rolling features (this takes ~30 seconds)...")

def weighted_form(arr):
    """Win rate weighted so that the most recent match matters most."""
    n = len(arr)
    weights = np.arange(1, n + 1, dtype=float)
    return float(np.dot(arr, weights) / weights.sum())

# th is sorted by (team, date) — shift(1) excludes the current match (no leakage)
grp = th.groupby("team", sort=False)
th["avg_scored_20"]   = grp["goals_scored"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
th["avg_conceded_20"] = grp["goals_conceded"].transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
th["form_10"]         = grp["win"].transform(
    lambda x: x.shift(1).rolling(10, min_periods=5).apply(weighted_form, raw=True)
)
print("  Rolling features done.")


# ── SECTION 4: ELO AT MATCH TIME (merge_asof) ────────────────────────────────
print("Merging ELO ratings at match time...")

elo_sorted = elo_raw.sort_values("date")[["date", "team", "rating"]].copy()
th_sorted  = th.sort_values("date").copy()

# For each team-match row, find the most recent ELO record before that date
th_with_elo = pd.merge_asof(
    th_sorted,
    elo_sorted.rename(columns={"rating": "elo_at_match"}),
    on="date",
    by="team",
    direction="backward"
)
th_with_elo = th_with_elo.sort_values(["team", "date"]).reset_index(drop=True)
print(f"  ELO coverage: {th_with_elo['elo_at_match'].notna().mean()*100:.1f}% of rows have ELO")


# ── SECTION 5: H2H WIN RATES (TIME-SAFE) ─────────────────────────────────────
print("Calculating time-safe H2H win rates...")

# Sort a working copy by (team, opponent, date) for expanding-window H2H.
# shift(1) before expanding ensures only PRIOR matches count — no leakage.
th_h2h = th[["date", "team", "opponent", "win"]].sort_values(
    ["team", "opponent", "date"]
).reset_index(drop=True)

grp_h2h = th_h2h.groupby(["team", "opponent"], sort=False)
th_h2h["h2h_cumwins"]  = grp_h2h["win"].transform(lambda x: x.shift(1).expanding().sum())
th_h2h["h2h_cumgames"] = grp_h2h["win"].transform(lambda x: x.shift(1).expanding().count())
th_h2h["h2h_win_rate"] = th_h2h["h2h_cumwins"] / th_h2h["h2h_cumgames"].clip(lower=1)
# Fill NaN (no prior H2H data) with the global empirical prior
th_h2h["h2h_win_rate"]  = th_h2h["h2h_win_rate"].fillna(0.45)
th_h2h["h2h_matches"]   = th_h2h["h2h_cumgames"].fillna(0).astype(int)

# Merge time-safe H2H back into the main table
th_with_elo = th_with_elo.merge(
    th_h2h[["date", "team", "opponent", "h2h_win_rate", "h2h_matches"]],
    on=["date", "team", "opponent"],
    how="left"
)
print(f"  H2H NaN rate after merge: {th_with_elo['h2h_win_rate'].isna().mean()*100:.1f}%")


# ── SECTION 6: TOURNAMENT TYPE CLASSIFICATION ─────────────────────────────────
print("Classifying tournament types...")

th_with_elo["tournament_type"] = th_with_elo["tournament"].apply(classify_tournament)


# ── SECTION 7: ASSEMBLE MATCHES_CLEAN.CSV ────────────────────────────────────
print("Assembling matches_clean.csv...")

home_feats = th_with_elo[th_with_elo["is_home"]].copy()
away_feats  = th_with_elo[~th_with_elo["is_home"]].copy()

matches = home_feats[[
    "date", "team", "opponent", "goals_scored", "goals_conceded",
    "neutral", "tournament", "tournament_type",
    "elo_at_match", "avg_scored_20", "avg_conceded_20", "form_10",
    "h2h_win_rate", "h2h_matches"
]].rename(columns={
    "team": "home_team", "opponent": "away_team",
    "goals_scored": "home_score", "goals_conceded": "away_score",
    "elo_at_match": "home_elo",
    "avg_scored_20": "home_avg_scored", "avg_conceded_20": "home_avg_conceded",
    "form_10": "home_form",
    "h2h_win_rate": "h2h_home_win_rate", "h2h_matches": "h2h_total_matches"
})

away_side = away_feats[[
    "date", "team", "opponent",
    "elo_at_match", "avg_scored_20", "avg_conceded_20", "form_10"
]].rename(columns={
    "team": "away_team", "opponent": "home_team",
    "elo_at_match": "away_elo",
    "avg_scored_20": "away_avg_scored", "avg_conceded_20": "away_avg_conceded",
    "form_10": "away_form"
})

matches_clean = matches.merge(away_side, on=["date", "home_team", "away_team"], how="inner")

# Derived features
matches_clean["elo_diff"]       = matches_clean["home_elo"] - matches_clean["away_elo"]
matches_clean["neutral_ground"] = matches_clean["neutral"].astype(int)

# Target variable (home team perspective)
matches_clean["result"] = np.where(
    matches_clean["home_score"] > matches_clean["away_score"], "W",
    np.where(matches_clean["home_score"] == matches_clean["away_score"], "D", "L")
)

col_order = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "result", "neutral_ground", "tournament", "tournament_type",
    "home_elo", "away_elo", "elo_diff",
    "home_avg_scored", "home_avg_conceded", "home_form",
    "away_avg_scored", "away_avg_conceded", "away_form",
    "h2h_home_win_rate", "h2h_total_matches"
]
matches_clean = matches_clean[col_order].sort_values("date").reset_index(drop=True)

# Drop rows where ALL core features are missing (very early matches with no lookback)
core_feats = ["home_elo", "away_elo", "home_avg_scored", "away_avg_scored"]
matches_clean = matches_clean.dropna(subset=core_feats, how="all")

print(f"  matches_clean rows : {len(matches_clean):,}")
print(f"  Feature NaN rates  :")
for c in ["home_elo", "elo_diff", "home_avg_scored", "home_form", "h2h_home_win_rate"]:
    pct = matches_clean[c].isna().mean() * 100
    print(f"    {c:25s}: {pct:.1f}% NaN")


# ── SECTION 8: TEAM_FEATURES.CSV (CURRENT SNAPSHOT) ──────────────────────────
print("\nBuilding team_features.csv (current snapshot)...")

active_cutoff = pd.Timestamp("2010-01-01")
active_teams  = elo_raw[elo_raw["date"] >= active_cutoff]["team"].unique()
current_elo   = (elo_raw[elo_raw["team"].isin(active_teams)]
                 .sort_values("date")
                 .groupby("team")["rating"]
                 .last()
                 .reset_index()
                 .rename(columns={"rating": "elo_rating"}))

latest_stats = (th_with_elo.sort_values("date")
                .groupby("team")
                .last()
                .reset_index()
                [["team", "avg_scored_20", "avg_conceded_20", "form_10"]]
                .rename(columns={
                    "avg_scored_20":   "avg_goals_scored",
                    "avg_conceded_20": "avg_goals_conceded",
                    "form_10":         "form_index"
                }))

match_counts = (th.groupby("team").size().reset_index().rename(columns={0: "matches_played"}))

# -- Transfermarkt squad values --
nt = pd.read_csv(RAW / "national_teams.csv")
# Normalize Transfermarkt names to canonical
nt["name"] = nt["name"].apply(norm)

nt_slim = (nt[["name", "total_market_value", "squad_size", "average_age", "fifa_ranking"]]
           .rename(columns={
               "name":               "tm_name",
               "total_market_value": "squad_market_value_eur",
               "squad_size":         "squad_size",
               "average_age":        "avg_squad_age",
               "fifa_ranking":       "fifa_ranking",
           }))

# England, Spain, France missing in source — estimated from Transfermarkt.co.uk (2025).
# ACTION: verify at transfermarkt.co.uk before publishing.
manual_tm = pd.DataFrame([
    {"tm_name": "England", "squad_market_value_eur": 1_250_000_000, "avg_squad_age": 27.3, "fifa_ranking": 4,  "squad_size": 26, "value_imputed": True},
    {"tm_name": "Spain",   "squad_market_value_eur":   950_000_000, "avg_squad_age": 26.7, "fifa_ranking": 2,  "squad_size": 26, "value_imputed": True},
    {"tm_name": "France",  "squad_market_value_eur": 1_080_000_000, "avg_squad_age": 26.9, "fifa_ranking": 1,  "squad_size": 26, "value_imputed": True},
])
nt_slim = nt_slim.dropna(subset=["squad_market_value_eur"])
nt_slim["value_imputed"] = False
nt_slim = pd.concat([nt_slim, manual_tm], ignore_index=True)

team_features = (current_elo
                 .merge(latest_stats, on="team", how="left")
                 .merge(match_counts, on="team", how="left"))

# tm_name = canonical team name (same as team column after normalization above)
team_features["tm_name"] = team_features["team"]
team_features = team_features.merge(nt_slim, on="tm_name", how="left").drop(columns="tm_name")
team_features["squad_market_value_m"] = (team_features["squad_market_value_eur"] / 1e6).round(1)

col_tf = ["team", "elo_rating", "squad_market_value_m", "squad_market_value_eur",
          "avg_goals_scored", "avg_goals_conceded", "form_index",
          "squad_size", "avg_squad_age", "fifa_ranking", "value_imputed", "matches_played"]
team_features = (team_features[col_tf]
                 .sort_values("elo_rating", ascending=False)
                 .reset_index(drop=True))

print(f"  Teams in team_features: {len(team_features)}")
print("\n  Top 10 by ELO:")
print(team_features.head(10)[["team","elo_rating","squad_market_value_m","avg_squad_age","fifa_ranking"]].to_string(index=False))
print("\n  Portugal:")
port_row = team_features[team_features["team"] == "Portugal"]
print(port_row[["team","elo_rating","squad_market_value_m","avg_squad_age","fifa_ranking","form_index","matches_played"]].to_string(index=False))

# Coverage check for WC 2026 teams
wc_teams_in_tf = set(groups["team"]) & set(team_features["team"])
wc_missing     = set(groups["team"]) - set(team_features["team"])
print(f"\n  WC 2026 teams with ELO data : {len(wc_teams_in_tf)}/48")
if wc_missing:
    print(f"  WC 2026 teams MISSING ELO  : {sorted(wc_missing)}")


# ── SECTION 9: PORTUGAL_PATH.CSV ─────────────────────────────────────────────
print("\nBuilding portugal_path.csv...")

portugal_path = pd.DataFrame([
    {"stage": "Group Stage",    "fixture": "Portugal vs DR Congo",   "type": "group",    "notes": "Group K, Match 1 — Jun 17"},
    {"stage": "Group Stage",    "fixture": "Portugal vs Uzbekistan", "type": "group",    "notes": "Group K, Match 2 — Jun 23"},
    {"stage": "Group Stage",    "fixture": "Colombia vs Portugal",   "type": "group",    "notes": "Group K, Match 3 — Jun 27"},
    {"stage": "Round of 32",    "fixture": "Portugal vs TBD",        "type": "knockout", "notes": "If 1K: faces Annex-C assigned 3rd-place team; if 2K: faces 2L"},
    {"stage": "Round of 16",    "fixture": "Portugal vs TBD",        "type": "knockout", "notes": "Winner of adjacent R32 match"},
    {"stage": "Quarter-Final",  "fixture": "Portugal vs TBD",        "type": "knockout", "notes": "TBD"},
    {"stage": "Semi-Final",     "fixture": "Portugal vs TBD",        "type": "knockout", "notes": "TBD"},
    {"stage": "Final",          "fixture": "Portugal vs TBD",        "type": "knockout", "notes": "TBD"},
])
print(portugal_path.to_string(index=False))


# ── SECTION 10: SAVE ALL OUTPUTS ─────────────────────────────────────────────
print("\nSaving processed files...")
matches_clean.to_csv(PROC / "matches_clean.csv",   index=False)
team_features.to_csv(PROC / "team_features.csv",   index=False)
portugal_path.to_csv(PROC / "portugal_path.csv",   index=False)

print(f"  Saved: data/processed/matches_clean.csv   ({len(matches_clean):,} rows)")
print(f"  Saved: data/processed/team_features.csv   ({len(team_features):,} rows)")
print(f"  Saved: data/processed/portugal_path.csv   ({len(portugal_path)} rows)")
print("\nPhase 3 complete.")
