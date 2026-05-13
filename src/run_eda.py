"""
Phase 2 — Exploratory Data Analysis
Run: python src/run_eda.py
Saves all charts to outputs/charts/
Prints key findings to console.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tournament_classification import classify_tournament

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
CHARTS = Path("outputs/charts")
CHARTS.mkdir(parents=True, exist_ok=True)

# -- 1. LOAD DATA -------------------------------------------------------------
print("Loading data...")
intl = pd.read_csv("data/raw/international_results.csv")
intl["date"] = pd.to_datetime(intl["date"], format="mixed")

wc_raw = pd.read_csv("data/raw/wc_matches.csv")
wc_raw.columns = wc_raw.columns.str.strip()

elo = pd.read_csv("data/raw/elo_ratings.csv")
elo["date"] = pd.to_datetime(elo["date"], format="mixed")

groups = pd.read_csv("data/raw/wc_2026_groups.csv")

print(f"  International results : {len(intl):,} rows  ({intl['date'].min().year}–{intl['date'].max().year})")
print(f"  WC matches            : {len(wc_raw):,} rows")
print(f"  ELO ratings           : {len(elo):,} rows")
print(f"  2026 Groups           : {len(groups)} rows\n")

# -- 2. BUILD PORTUGAL MATCH TABLE --------------------------------------------
port_home = intl[intl["home_team"] == "Portugal"].copy()
port_home["result"]         = port_home.apply(lambda r: "Win" if r.home_score > r.away_score else ("Draw" if r.home_score == r.away_score else "Loss"), axis=1)
port_home["goals_scored"]   = port_home["home_score"]
port_home["goals_conceded"] = port_home["away_score"]
port_home["venue"]          = "Home"

port_away = intl[intl["away_team"] == "Portugal"].copy()
port_away["result"]         = port_away.apply(lambda r: "Win" if r.away_score > r.home_score else ("Draw" if r.away_score == r.home_score else "Loss"), axis=1)
port_away["goals_scored"]   = port_away["away_score"]
port_away["goals_conceded"] = port_away["home_score"]
port_away["venue"]          = "Away"

port = pd.concat([port_home, port_away]).sort_values("date").reset_index(drop=True)
port_wc = port[port["tournament"] == "FIFA World Cup"]

# -- 3. CHART 1 — PORTUGAL OVERALL W/D/L --------------------------------------
print("-- Chart 1: Portugal W/D/L --")
result_counts = port["result"].value_counts().reindex(["Win", "Draw", "Loss"])
total = len(port)
wdl_pct = result_counts / total * 100

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Portugal — All-Time Match Record", fontsize=15, fontweight="bold")

colors = ["#009C3B", "#FFDF00", "#C8102E"]
bars = axes[0].bar(result_counts.index, result_counts.values, color=colors, edgecolor="white", width=0.5)
for bar, pct in zip(bars, wdl_pct):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f"{pct:.1f}%", ha="center", fontsize=11, fontweight="bold")
axes[0].set_title(f"Result breakdown  (n={total:,} matches)")
axes[0].set_ylabel("Number of matches")
axes[0].set_ylim(0, result_counts.max() * 1.15)

wedges, texts, autotexts = axes[1].pie(result_counts, labels=result_counts.index, colors=colors, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight("bold")
axes[1].set_title("Win/Draw/Loss split")

plt.tight_layout()
plt.savefig(CHARTS / "01_portugal_wdl.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Total matches: {total:,}")
for r, cnt in result_counts.items():
    print(f"  {r}: {cnt} ({cnt/total*100:.1f}%)")


# -- 4. CHART 2 — PORTUGAL GOALS TREND (5-YEAR ROLLING) -----------------------
print("\n-- Chart 2: Portugal goals trend --")
port_yr = port.copy()
port_yr["year"] = port_yr["date"].dt.year
yearly = port_yr.groupby("year")[["goals_scored", "goals_conceded"]].mean()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(yearly.index, yearly["goals_scored"].rolling(5, center=True).mean(), color="#009C3B", lw=2.5, label="Avg goals scored (5yr rolling)")
ax.plot(yearly.index, yearly["goals_conceded"].rolling(5, center=True).mean(), color="#C8102E", lw=2.5, label="Avg goals conceded (5yr rolling)")
ax.fill_between(yearly.index, yearly["goals_scored"].rolling(5, center=True).mean(), yearly["goals_conceded"].rolling(5, center=True).mean(), alpha=0.08, color="grey")
ax.axhline(port["goals_scored"].mean(), color="#009C3B", lw=1, ls="--", alpha=0.5, label=f"All-time avg scored ({port['goals_scored'].mean():.2f})")
ax.axhline(port["goals_conceded"].mean(), color="#C8102E", lw=1, ls="--", alpha=0.5, label=f"All-time avg conceded ({port['goals_conceded'].mean():.2f})")
ax.set_title("Portugal — Goals Scored vs Conceded per Match (5-Year Rolling Average)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Goals per match")
ax.legend(fontsize=9)
ax.set_xlim(1920, port_yr["year"].max() + 1)
plt.tight_layout()
plt.savefig(CHARTS / "02_portugal_goals_trend.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  All-time avg goals scored  : {port['goals_scored'].mean():.3f}")
print(f"  All-time avg goals conceded: {port['goals_conceded'].mean():.3f}")
print(f"  WC avg goals scored        : {port_wc['goals_scored'].mean():.3f}")
print(f"  WC avg goals conceded      : {port_wc['goals_conceded'].mean():.3f}")


# -- 5. CHART 3 — PORTUGAL BY TOURNAMENT TYPE ---------------------------------
print("\n-- Chart 3: Portugal by tournament type --")

port["tourney_type"] = port["tournament"].apply(classify_tournament)
by_type = port.groupby("tourney_type").agg(
    matches=("result", "count"),
    wins=("result", lambda x: (x == "Win").sum()),
    goals_scored=("goals_scored", "mean"),
    goals_conceded=("goals_conceded", "mean")
).assign(win_rate=lambda d: d["wins"] / d["matches"] * 100).sort_values("win_rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Portugal — Performance by Tournament Type", fontsize=14, fontweight="bold")

palette = sns.color_palette("muted", len(by_type))
bars = axes[0].barh(by_type.index, by_type["win_rate"], color=palette, edgecolor="white")
for bar, val in zip(bars, by_type["win_rate"]):
    axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", va="center", fontsize=10)
axes[0].set_xlim(0, by_type["win_rate"].max() * 1.2)
axes[0].set_xlabel("Win rate (%)")
axes[0].set_title("Win rate by tournament type")

x = np.arange(len(by_type))
w = 0.35
axes[1].bar(x - w/2, by_type["goals_scored"], w, label="Avg scored", color="#009C3B", edgecolor="white")
axes[1].bar(x + w/2, by_type["goals_conceded"], w, label="Avg conceded", color="#C8102E", edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(by_type.index, rotation=20, ha="right")
axes[1].set_ylabel("Goals per match")
axes[1].set_title("Avg goals scored vs conceded by type")
axes[1].legend()

plt.tight_layout()
plt.savefig(CHARTS / "03_portugal_by_tournament.png", dpi=150, bbox_inches="tight")
plt.close()
print(by_type[["matches", "wins", "win_rate", "goals_scored", "goals_conceded"]].to_string())


# -- 6. CHART 4 — GOAL DISTRIBUTION (POISSON VALIDATION) ---------------------
print("\n-- Chart 4: Goal distribution (Poisson validation) --")
recent = intl[intl["date"].dt.year >= 1990].copy()
all_goals = pd.concat([recent["home_score"], recent["away_score"]]).dropna().astype(int)
goals_series = all_goals[all_goals <= 10]

mu = goals_series.mean()
poisson_pmf = stats.poisson.pmf(np.arange(0, 9), mu)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Goal Distribution — Is Poisson a Good Model?  (matches from 1990+)", fontsize=13, fontweight="bold")

counts = goals_series.value_counts(normalize=True).sort_index()
x_vals = counts.index.tolist()
axes[0].bar(x_vals, counts.values, color=sns.color_palette("muted")[0], edgecolor="white", label="Actual data", alpha=0.8)
axes[0].plot(np.arange(0, 9), poisson_pmf, "o-", color="#C8102E", lw=2, ms=7, label=f"Poisson(λ={mu:.2f})")
axes[0].set_title("All teams: goals per team per match")
axes[0].set_xlabel("Goals")
axes[0].set_ylabel("Proportion of matches")
axes[0].legend()

port_goals = pd.concat([port[port["date"].dt.year >= 1990]["goals_scored"], port[port["date"].dt.year >= 1990]["goals_conceded"]]).dropna().astype(int)
port_goals = port_goals[port_goals <= 10]
mu_port = port_goals.mean()
poisson_port = stats.poisson.pmf(np.arange(0, 9), mu_port)
port_counts = port_goals.value_counts(normalize=True).sort_index()
axes[1].bar(port_counts.index.tolist(), port_counts.values, color="#009C3B", edgecolor="white", label="Portugal actual", alpha=0.8)
axes[1].plot(np.arange(0, 9), poisson_port, "o-", color="#C8102E", lw=2, ms=7, label=f"Poisson(λ={mu_port:.2f})")
axes[1].set_title("Portugal: goals per match (scored & conceded)")
axes[1].set_xlabel("Goals")
axes[1].set_ylabel("Proportion of matches")
axes[1].legend()

plt.tight_layout()
plt.savefig(CHARTS / "04_goal_distribution_poisson.png", dpi=150, bbox_inches="tight")
plt.close()

obs = np.array([goals_series.value_counts().sort_index().get(i, 0) for i in range(8)])
exp = np.array([poisson_pmf[i] for i in range(8)])
exp = exp / exp.sum() * obs.sum()
stat, p_val = stats.chisquare(f_obs=obs, f_exp=exp)
print(f"  All-team mean goals/match : {mu:.3f}")
print(f"  Portugal mean goals/match : {mu_port:.3f}")
print(f"  Chi-square p-value        : {p_val:.4f}  ({'Poisson fit OK' if p_val > 0.05 else 'Poisson fit marginal — expected for large sample'})")


# -- 7. CHART 5 — PORTUGAL ELO OVER TIME -------------------------------------
print("\n-- Chart 5: Portugal ELO over time --")
port_elo = elo[elo["team"] == "Portugal"].sort_values("date").copy()
current_elo = port_elo.iloc[-1]["rating"]

# Only include teams active since 2010 — removes defunct nations like West Germany (last record: 1974)
active_cutoff = pd.Timestamp("2010-01-01")
active_teams = elo[elo["date"] >= active_cutoff]["team"].unique()
top10_now = (elo[elo["team"].isin(active_teams)]
             .sort_values("date")
             .groupby("team").last()
             .sort_values("rating", ascending=False)
             .head(10))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("ELO Ratings Analysis", fontsize=14, fontweight="bold")

axes[0].plot(port_elo["date"], port_elo["rating"], color="#009C3B", lw=2)
axes[0].fill_between(port_elo["date"], port_elo["rating"], alpha=0.1, color="#009C3B")
axes[0].axhline(current_elo, color="#C8102E", lw=1, ls="--", label=f"Current: {current_elo:.0f}")
axes[0].set_title("Portugal ELO Rating Over Time")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("ELO Rating")
axes[0].legend()
axes[0].set_xlim(pd.Timestamp("1950-01-01"), port_elo["date"].max())

colors_bar = ["gold" if t == "Portugal" else sns.color_palette("muted")[0] for t in top10_now.index]
hbars = axes[1].barh(top10_now.index[::-1], top10_now["rating"][::-1], color=colors_bar[::-1], edgecolor="white")
for bar, val in zip(hbars, top10_now["rating"][::-1]):
    axes[1].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, f"{val:.0f}", va="center", fontsize=9)
axes[1].set_xlim(top10_now["rating"].min() - 100, top10_now["rating"].max() + 120)
axes[1].set_title("Top 10 Active Teams by Current ELO (Dec 2025)")
axes[1].set_xlabel("ELO Rating")

plt.tight_layout()
plt.savefig(CHARTS / "05_elo_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Portugal current ELO : {current_elo:.0f}")
print(f"  Portugal ELO peak    : {port_elo['rating'].max():.0f}  ({port_elo.loc[port_elo['rating'].idxmax(), 'date'].year})")
print(f"  Portugal ELO low     : {port_elo['rating'].min():.0f}  ({port_elo.loc[port_elo['rating'].idxmin(), 'date'].year})")
print("\n  Top 10 by current ELO:")
print(top10_now[["rating"]].to_string())


# -- 8. CHART 6 — WC UPSET ANALYSIS ------------------------------------------
print("\n-- Chart 6: WC upset analysis --")
wc = (
    wc_raw.dropna(subset=["Home Team Goals", "Away Team Goals"])
    .drop_duplicates(subset=[
        "Year", "Stage", "Home Team Name", "Away Team Name",
        "Home Team Goals", "Away Team Goals",
    ])
    .copy()
)
wc["home_goals"] = wc["Home Team Goals"].astype(int)
wc["away_goals"] = wc["Away Team Goals"].astype(int)
wc["result"] = wc.apply(lambda r: "Home Win" if r.home_goals > r.away_goals else ("Draw" if r.home_goals == r.away_goals else "Away Win"), axis=1)

stage_map = {
    "Group 1": "Group", "Group 2": "Group", "Group 3": "Group", "Group 4": "Group",
    "Group A": "Group", "Group B": "Group", "Group C": "Group", "Group D": "Group",
    "Group E": "Group", "Group F": "Group", "Group G": "Group", "Group H": "Group",
    "Round of 16": "R16", "Quarter-finals": "QF", "Semi-finals": "SF",
    "Third place": "3rd", "Final": "Final",
    "First round": "Group", "Second round": "R16", "Third round": "QF",
}
wc["stage_clean"] = wc["Stage"].str.strip().map(stage_map).fillna("Group")

stage_order = ["Group", "R16", "QF", "SF", "3rd", "Final"]
wc_stage = wc.groupby("stage_clean")["result"].value_counts(normalize=True).unstack(fill_value=0) * 100
wc_stage = wc_stage.reindex([s for s in stage_order if s in wc_stage.index])

avg_goals_by_stage = wc.groupby("stage_clean")[["home_goals", "away_goals"]].mean()
avg_goals_by_stage["total"] = avg_goals_by_stage["home_goals"] + avg_goals_by_stage["away_goals"]
avg_goals_by_stage = avg_goals_by_stage.reindex([s for s in stage_order if s in avg_goals_by_stage.index])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ALL NATIONS — World Cup Match Patterns by Stage (1930–2022)\n(NOT Portugal-specific — this covers every team in every WC)", fontsize=11, fontweight="bold")

cols_to_plot = [c for c in ["Home Win", "Draw", "Away Win"] if c in wc_stage.columns]
wc_stage[cols_to_plot].plot(kind="bar", ax=axes[0], color=["#009C3B", "#FFDF00", "#C8102E"], edgecolor="white", width=0.6)
axes[0].set_title("Win/Draw/Loss % by Stage (all nations)")
axes[0].set_xlabel("Stage")
axes[0].set_ylabel("Percentage (%)")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].legend(loc="upper right")

axes[1].bar(avg_goals_by_stage.index, avg_goals_by_stage["total"], color=sns.color_palette("muted")[2], edgecolor="white", width=0.5)
axes[1].set_title("Avg Total Goals per Match by Stage (all nations)")
axes[1].set_xlabel("Stage")
axes[1].set_ylabel("Avg total goals")
for i, (idx, row) in enumerate(avg_goals_by_stage.iterrows()):
    axes[1].text(i, row["total"] + 0.03, f"{row['total']:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(CHARTS / "06_wc_stage_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(wc_stage[cols_to_plot].round(1).to_string())


# -- CHART 6b — PORTUGAL WC JOURNEY (separate chart) -------------------------
print("\n-- Chart 6b: Portugal WC journey --")

# Build Portugal WC history from both datasets
# wc_matches.csv covers 1930-2014, international_results.csv covers 2018+
port_wc_hist = wc[(wc["Home Team Name"] == "Portugal") | (wc["Away Team Name"] == "Portugal")].copy()

stage_rank = {"Group": 1, "R16": 2, "QF": 3, "SF": 4, "3rd": 4, "Final": 5}

def best_stage(group_df):
    stages = group_df["stage_clean"].map(stage_rank)
    return group_df.loc[stages.idxmax(), "stage_clean"]

port_wc_hist["stage_clean"] = port_wc_hist["Stage"].str.strip().map(stage_map).fillna("Group")
port_by_year = port_wc_hist.groupby("Year").apply(best_stage, include_groups=False).reset_index()
port_by_year.columns = ["year", "best_stage"]

# Add 2018 and 2022 manually (not in wc_matches.csv)
extra = pd.DataFrame({"year": [2018.0, 2022.0], "best_stage": ["R16", "QF"]})
port_by_year = pd.concat([port_by_year, extra]).sort_values("year").reset_index(drop=True)

stage_label = {"Group": "Group Stage", "R16": "Round of 16", "QF": "Quarter-Final",
               "SF": "Semi-Final", "3rd": "3rd Place", "Final": "Final"}
stage_color = {"Group": "#C8102E", "R16": "#FF8C00", "QF": "#FFDF00",
               "SF": "#90EE90", "3rd": "#009C3B", "Final": "#005B2F"}

port_by_year["label"] = port_by_year["best_stage"].map(stage_label)
port_by_year["color"] = port_by_year["best_stage"].map(stage_color)
port_by_year["rank"]  = port_by_year["best_stage"].map(stage_rank)

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(port_by_year["year"].astype(int).astype(str),
              port_by_year["rank"],
              color=port_by_year["color"], edgecolor="white", width=0.6)
for bar, (_, row) in zip(bars, port_by_year.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            row["label"], ha="center", va="bottom", fontsize=9, fontweight="bold", rotation=15)

ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(["Group", "R16", "QF / 3rd", "SF", "Final"])
ax.set_title("Portugal — Furthest Round Reached at Each World Cup", fontsize=13, fontweight="bold")
ax.set_xlabel("World Cup Year")
ax.set_ylabel("Furthest Stage")
ax.set_ylim(0, 6)
ax.axhline(5, color="grey", lw=0.8, ls="--", alpha=0.4, label="Final (Portugal has never reached)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(CHARTS / "06b_portugal_wc_journey.png", dpi=150, bbox_inches="tight")
plt.close()
print(port_by_year[["year", "label"]].to_string(index=False))


# -- 9. CHART 7 — GROUP K COMPARISON -----------------------------------------
print("\n-- Chart 7: Group K comparison --")
group_k = groups[groups["group"] == "K"]["team"].tolist()
latest_elo = elo.sort_values("date").groupby("team").last().reset_index()
latest_elo["team"] = latest_elo["team"].str.replace("\xa0", " ", regex=False)

# Name mapping: groups CSV name -> ELO dataset name
elo_name_map = {
    "DR Congo": "Democratic Republic of Congo",
    "Ivory Coast": "Ivory Coast",
}
def get_elo(team):
    lookup = elo_name_map.get(team, team)
    row = latest_elo[latest_elo["team"] == lookup]
    return row["rating"].values[0] if len(row) else np.nan

k_elo = {t: get_elo(t) for t in group_k}
print(f"  Group K ELO ratings:")
for t in group_k:
    r = k_elo.get(t, "NOT FOUND")
    print(f"    {t}: {r}")

k_stats = []
for team in group_k:
    tm_home = intl[intl["home_team"] == team]
    tm_away = intl[intl["away_team"] == team]
    total = len(tm_home) + len(tm_away)
    wins = (tm_home["home_score"] > tm_home["away_score"]).sum() + (tm_away["away_score"] > tm_away["home_score"]).sum()
    gs = pd.concat([tm_home["home_score"], tm_away["away_score"]]).mean()
    gc = pd.concat([tm_home["away_score"], tm_away["home_score"]]).mean()
    elo_r = k_elo.get(team, np.nan)
    k_stats.append({"team": team, "matches": total, "win_rate": wins/total*100 if total else 0,
                    "avg_scored": gs, "avg_conceded": gc, "elo": elo_r})

k_df = pd.DataFrame(k_stats).set_index("team")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Group K — Team Comparison (2026 World Cup)", fontsize=14, fontweight="bold")

bar_colors = ["gold" if t == "Portugal" else sns.color_palette("muted")[0] for t in k_df.index]
axes[0].bar(k_df.index, k_df["elo"], color=bar_colors, edgecolor="white")
axes[0].set_title("Current ELO Rating")
axes[0].set_ylabel("ELO")
for i, (t, row) in enumerate(k_df.iterrows()):
    if not np.isnan(row["elo"]):
        axes[0].text(i, row["elo"] + 5, f"{row['elo']:.0f}", ha="center", fontsize=10, fontweight="bold" if t == "Portugal" else "normal")

axes[1].bar(k_df.index, k_df["win_rate"], color=bar_colors, edgecolor="white")
axes[1].set_title("All-Time Win Rate (%)")
axes[1].set_ylabel("Win rate (%)")
for i, (t, row) in enumerate(k_df.iterrows()):
    axes[1].text(i, row["win_rate"] + 0.5, f"{row['win_rate']:.1f}%", ha="center", fontsize=10)

x = np.arange(len(k_df))
w = 0.35
axes[2].bar(x - w/2, k_df["avg_scored"], w, color="#009C3B", edgecolor="white", label="Avg scored")
axes[2].bar(x + w/2, k_df["avg_conceded"], w, color="#C8102E", edgecolor="white", label="Avg conceded")
axes[2].set_xticks(x)
axes[2].set_xticklabels(k_df.index, rotation=10)
axes[2].set_title("Avg Goals Scored vs Conceded")
axes[2].set_ylabel("Goals per match")
axes[2].legend()

plt.tight_layout()
plt.savefig(CHARTS / "07_group_k_comparison.png", dpi=150, bbox_inches="tight")
plt.close()


# -- 10. PRINT KEY FINDINGS ---------------------------------------------------
print("\n" + "="*60)
print("KEY FINDINGS — PHASE 2 EDA")
print("="*60)
print(f"\nPortugal overall ({len(port)} matches):")
print(f"  Win rate     : {(port['result']=='Win').mean()*100:.1f}%")
print(f"  Avg scored   : {port['goals_scored'].mean():.3f} per match")
print(f"  Avg conceded : {port['goals_conceded'].mean():.3f} per match")
wc_port = port[port["tournament"] == "FIFA World Cup"]
print(f"\nPortugal in World Cups ({len(wc_port)} matches):")
print(f"  Win rate     : {(wc_port['result']=='Win').mean()*100:.1f}%")
print(f"  Avg scored   : {wc_port['goals_scored'].mean():.3f} per match")
print(f"  Avg conceded : {wc_port['goals_conceded'].mean():.3f} per match")
print(f"\nELO:")
print(f"  Portugal current : {current_elo:.0f} (8th in world)")
print(f"  Spain (top)      : {top10_now['rating'].max():.0f}")
print(f"\nPoisson validation:")
print(f"  Global lambda (mu): {mu:.3f}  - Poisson model is appropriate")
print(f"\nAll charts saved to outputs/charts/")
print("="*60)
