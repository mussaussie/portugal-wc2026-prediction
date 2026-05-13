"""
2026 FIFA World Cup Monte Carlo simulation engine (ELO model).

Bracket structure (approximation of official 2026 FIFA bracket):
  3 macro-sections of 4 groups each:
    MS1 — Groups A/B/C/D
    MS2 — Groups E/F/G/H
    MS3 — Groups I/J/K/L  ← Portugal (Group K)

  Within each macro-section, groups are cross-paired (X,Y):
    1X vs 3rd_from_other_section   (1st place plays a 3rd-place qualifier)
    1Y vs 2X                       (1st vs 2nd, same section, different groups)
    1Z vs 3rd_from_other_section
    1W vs 2Z

  The 2 displaced runners-up per macro-section go to a 4th pool of 8 teams
  (the remaining 6 displaced runners-up + 2 remaining 3rd-place teams).
  This guarantees no 3rd-place team ever faces another 3rd-place team in R32.

  Assumption: 3rd-place teams are distributed cross-section:
    MS1 receives its 3rd-place slots filled by the 2 best 3rd from MS3.
    MS2 receives the 2 best 3rd from MS1 (displaced from MS1 slot assignments).
    MS3 receives the 2 best 3rd from MS2.
    Section 4 pool receives the remaining 2 3rd-place teams.

Path: R32 → R16 (cross-pair within section) → QF (cross-pair section pairs) → SF → Final
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.elo import simulate_match, simulate_scoreline, elo_win_prob
from src.team_names import norm
from src.tournament_rules import (
    QF_PAIRS,
    R16_PAIRS,
    SF_PAIRS,
    build_round_of_32_bracket,
    play_official_knockout_round,
)

# ── Constants ────────────────────────────────────────────────────────────────

GROUPS = list("ABCDEFGHIJKL")
DEFAULT_ELO    = 1500
DEFAULT_LAMBDA = 1.3
GLOBAL_AVG_GOALS = 1.35

# Macro-section group-pair definitions.
# Each pair (X, Y) produces 4 R32 slots:
#   1X vs 3rd_slot,  1Y vs 2X,  1Z vs 3rd_slot,  1W vs 2Z
# (where Z and W are the second pair in the macro-section)
MACRO_SECTIONS = [
    [("A", "B"), ("C", "D")],   # MS1
    [("E", "F"), ("G", "H")],   # MS2
    [("I", "J"), ("K", "L")],   # MS3 — Portugal is in Group K
]

# 3rd-place team cross-section routing:
#   MS1 slot receives 3rd-place from MS3 (indices 0,1 in best_thirds)
#   MS2 slot receives 3rd-place from MS1 (indices 2,3)
#   MS3 slot receives 3rd-place from MS2 (indices 4,5)
#   Section-4 pool receives remaining (indices 6,7)
# The 8 best 3rd-place teams are sorted best-first; this routing mixes sections.
THIRDS_ROUTING = [0, 1, 2, 3, 4, 5, 6, 7]   # just an index reference (see _build_bracket)


def _norm(name: str) -> str:
    """Apply canonical name normalisation (delegates to team_names.norm)."""
    return norm(name)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_team_data() -> tuple[dict, dict]:
    """Return (elo_dict, lambda_dict) keyed by normalised team name."""
    df = pd.read_csv(Path("data/processed/team_features.csv"))
    elo, lam = {}, {}
    for _, row in df.iterrows():
        team = norm(str(row["team"]))
        elo[team] = float(row["elo_rating"]) if pd.notna(row.get("elo_rating")) else DEFAULT_ELO
        scored   = row.get("avg_goals_scored")
        conceded = row.get("avg_goals_conceded")
        lam[team] = {
            "attack":  float(scored)   if pd.notna(scored)   else DEFAULT_LAMBDA,
            "defense": float(conceded) if pd.notna(conceded) else DEFAULT_LAMBDA,
        }
    return elo, lam


def load_groups() -> pd.DataFrame:
    """Return wc_2026_groups.csv with normalised team names."""
    df = pd.read_csv(Path("data/raw/wc_2026_groups.csv"))
    df["team"] = df["team"].apply(norm)
    return df


# ── Group stage ──────────────────────────────────────────────────────────────

def _goal_lambdas(a: str, b: str, lam: dict) -> tuple[float, float]:
    la_att = lam.get(a, {}).get("attack",  DEFAULT_LAMBDA)
    lb_def = lam.get(b, {}).get("defense", DEFAULT_LAMBDA)
    lb_att = lam.get(b, {}).get("attack",  DEFAULT_LAMBDA)
    la_def = lam.get(a, {}).get("defense", DEFAULT_LAMBDA)
    return (
        max(0.1, la_att * lb_def / GLOBAL_AVG_GOALS),
        max(0.1, lb_att * la_def / GLOBAL_AVG_GOALS),
    )


def simulate_group(teams: list, elo: dict, lam: dict, rng) -> list:
    """
    4-team round-robin using pure ELO win probabilities.
    Scorelines are sampled from Poisson lambdas derived from team attack/defense
    parameters — used solely for goal-difference / goals-for tiebreakers.
    Returns standings sorted by (pts, GD, GF) descending.
    Each entry: {'team': str, 'pts': int, 'gd': int, 'gf': int}
    """
    stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
    matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            # Outcome from pure ELO probabilities
            result = simulate_match(elo.get(a, DEFAULT_ELO), elo.get(b, DEFAULT_ELO),
                                    knockout=False, rng=rng)
            # Scoreline from Poisson lambdas (tiebreaker only)
            la, lb = _goal_lambdas(a, b, lam)
            ga, gb = simulate_scoreline(la, lb, rng)
            # Force scoreline direction to be consistent with ELO outcome
            if result == "A" and ga <= gb:
                ga = gb + 1
            elif result == "B" and gb <= ga:
                gb = ga + 1
            elif result == "D":
                gb = ga   # draw: same score

            stats[a]["gf"] += ga
            stats[b]["gf"] += gb
            stats[a]["gd"] += ga - gb
            stats[b]["gd"] += gb - ga
            if result == "A":
                stats[a]["pts"] += 3
            elif result == "D":
                stats[a]["pts"] += 1
                stats[b]["pts"] += 1
            else:
                stats[b]["pts"] += 3
            matches.append((a, b, ga, gb))

    return _rank_group(teams, stats, matches, rng)


def _rank_group(teams: list, stats: dict, matches: list, rng) -> list:
    """Rank group using FIFA-style goals rules, H2H for exact ties, then random draw."""
    rows = [{"team": t, **stats[t]} for t in teams]
    rows = sorted(rows, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    out = []
    i = 0
    while i < len(rows):
        key = (rows[i]["pts"], rows[i]["gd"], rows[i]["gf"])
        tied = [rows[i]]
        j = i + 1
        while j < len(rows) and (rows[j]["pts"], rows[j]["gd"], rows[j]["gf"]) == key:
            tied.append(rows[j])
            j += 1
        if len(tied) == 1:
            out.extend(tied)
        else:
            tied_teams = {r["team"] for r in tied}
            mini = {t: {"pts": 0, "gd": 0, "gf": 0, "draw": rng.random()} for t in tied_teams}
            for a, b, ga, gb in matches:
                if a not in tied_teams or b not in tied_teams:
                    continue
                mini[a]["gf"] += ga; mini[b]["gf"] += gb
                mini[a]["gd"] += ga - gb; mini[b]["gd"] += gb - ga
                if ga > gb:
                    mini[a]["pts"] += 3
                elif ga < gb:
                    mini[b]["pts"] += 3
                else:
                    mini[a]["pts"] += 1; mini[b]["pts"] += 1
            out.extend(sorted(
                tied,
                key=lambda x: (
                    mini[x["team"]]["pts"],
                    mini[x["team"]]["gd"],
                    mini[x["team"]]["gf"],
                    mini[x["team"]]["draw"],
                ),
                reverse=True,
            ))
        i = j
    return out


def _best_thirds(group_results: dict) -> list:
    """Select the 8 best 3rd-place finishers across all 12 groups."""
    thirds = [results[2] for results in group_results.values()]
    return [
        t["team"]
        for t in sorted(thirds, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)[:8]
    ]


# ── Knockout bracket ──────────────────────────────────────────────────────────

def _build_bracket(group_results: dict, best_thirds: list) -> list:
    """Return official 32-slot R32 bracket using FIFA 2026 Annex-C routing."""
    bracket = build_round_of_32_bracket(group_results, best_thirds)
    assert len(bracket) == 32, f"Expected 32-slot bracket, got {len(bracket)}"
    return bracket


def _play(a: str, b: str, elo: dict, rng) -> str:
    result = simulate_match(elo.get(a, DEFAULT_ELO), elo.get(b, DEFAULT_ELO),
                            knockout=True, rng=rng)
    return a if result == "A" else b


def _simulate_knockout(bracket: list, elo: dict, rng) -> str:
    """
    Run R32 → R16 → QF → SF → Final.

    R16 cross-pairing: within each section of 4 R32 winners, pair index 0v2, 1v3.
    QF cross-pairing: pair sections (1,2) and (3,4) the same way.
    """
    # R32 — 16 matches
    r32w = [_play(bracket[2*i], bracket[2*i+1], elo, rng) for i in range(16)]

    r16w = play_official_knockout_round(r32w, R16_PAIRS, lambda a, b: _play(a, b, elo, rng))
    qfw = play_official_knockout_round(r16w, QF_PAIRS, lambda a, b: _play(a, b, elo, rng))

    # SF
    sfw = play_official_knockout_round(qfw, SF_PAIRS, lambda a, b: _play(a, b, elo, rng))

    return _play(sfw[0], sfw[1], elo, rng)


# ── Full simulation ───────────────────────────────────────────────────────────

def simulate_tournament(groups_df: pd.DataFrame, elo: dict, lam: dict, rng) -> str:
    """One complete 2026 WC simulation. Returns tournament winner name."""
    group_results = {}
    for grp in GROUPS:
        teams = groups_df[groups_df["group"] == grp]["team"].tolist()
        group_results[grp] = simulate_group(teams, elo, lam, rng)

    thirds  = _best_thirds(group_results)
    bracket = _build_bracket(group_results, thirds)
    return _simulate_knockout(bracket, elo, rng)


def run_monte_carlo(n: int = 100_000, seed: int = 42) -> pd.DataFrame:
    """
    Run n tournament simulations. Returns a DataFrame with columns:
      winner, count, probability
    sorted by probability descending.
    """
    elo, lam = load_team_data()
    groups_df = load_groups()
    rng = np.random.default_rng(seed)

    winners = []
    for _ in range(n):
        winners.append(simulate_tournament(groups_df, elo, lam, rng))

    counts = pd.Series(winners).value_counts().reset_index()
    counts.columns = ["winner", "count"]
    counts["probability"] = counts["count"] / n
    return counts


def run_portugal_path_analysis(n: int = 100_000, seed: int = 42) -> dict:
    """
    Track Portugal's exit stage across n simulations.
    Returns dict: stage → count.
    Stages: Group stage, Round of 32, Round of 16, Quarter-Final,
            Semi-Final, Runner-up (final loss), Winner.
    """
    elo, lam = load_team_data()
    groups_df = load_groups()
    rng = np.random.default_rng(seed)
    TARGET = "Portugal"

    stage_counts = {
        "Group stage":  0,
        "Round of 32":  0,
        "Round of 16":  0,
        "Quarter-Final": 0,
        "Semi-Final":   0,
        "Runner-up":    0,
        "Winner":       0,
    }

    for _ in range(n):
        group_results = {}
        for grp in GROUPS:
            teams = groups_df[groups_df["group"] == grp]["team"].tolist()
            group_results[grp] = simulate_group(teams, elo, lam, rng)

        thirds    = _best_thirds(group_results)
        qualified = {r["team"] for res in group_results.values() for r in res[:2]}
        qualified.update(thirds)

        if TARGET not in qualified:
            stage_counts["Group stage"] += 1
            continue

        bracket = _build_bracket(group_results, thirds)
        r32w = [_play(bracket[2*i], bracket[2*i+1], elo, rng) for i in range(16)]
        if TARGET not in r32w:
            stage_counts["Round of 32"] += 1
            continue

        r16w = play_official_knockout_round(r32w, R16_PAIRS, lambda a, b: _play(a, b, elo, rng))
        if TARGET not in r16w:
            stage_counts["Round of 16"] += 1
            continue

        qfw = play_official_knockout_round(r16w, QF_PAIRS, lambda a, b: _play(a, b, elo, rng))
        if TARGET not in qfw:
            stage_counts["Quarter-Final"] += 1
            continue

        sfw = play_official_knockout_round(qfw, SF_PAIRS, lambda a, b: _play(a, b, elo, rng))
        if TARGET not in sfw:
            stage_counts["Semi-Final"] += 1
            continue

        final_winner = _play(sfw[0], sfw[1], elo, rng)
        if final_winner == TARGET:
            stage_counts["Winner"] += 1
        else:
            stage_counts["Runner-up"] += 1   # reached the Final but lost

    return stage_counts


if __name__ == "__main__":
    print("Running 100,000 simulations...")
    results = run_monte_carlo(100_000)
    print("\nTop 10 most likely champions:")
    print(results.head(10).to_string(index=False))

    port_row = results[results["winner"] == "Portugal"]
    if not port_row.empty:
        p = port_row.iloc[0]["probability"]
        print(f"\nPortugal win probability: {p:.1%}")
    else:
        print("\nPortugal: 0.0% (never won in 100k simulations)")
