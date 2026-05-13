"""
Phase 7 - Model comparison charts and final report.

Combines results from all three models:
  Model A: ELO simulation        (simulation/elo_simulation_results.csv)
  Model B: ELOPoissonDC          (simulation/dixon_coles_simulation_results.csv)
  Model C: XGBoost               (simulation/xgboost_simulation_results.csv)

Outputs:
  outputs/charts/win_probability_comparison.png
  outputs/charts/portugal_path_comparison.png
  outputs/charts/model_comparison_scatter.png
  outputs/reports/final_report.md

Usage:
    python src/run_phase7.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
CHARTS = ROOT / "outputs" / "charts"
REPORTS = ROOT / "outputs" / "reports"
SIM = ROOT / "simulation"
CHARTS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────

C_ELO   = "#2196F3"   # blue
C_DC    = "#FF9800"   # orange
C_XGB   = "#4CAF50"   # green
C_PORT  = "#C8102E"   # portugal red

MODEL_LABELS = {
    "elo":  "Model A: ELO",
    "dc":   "Model B: ELO-Poisson/DC",
    "xgb":  "Model C: XGBoost",
}

# ── Load simulation results ───────────────────────────────────────────────────

def load_results():
    elo = pd.read_csv(SIM / "elo_simulation_results.csv")
    dc  = pd.read_csv(SIM / "dixon_coles_simulation_results.csv")
    xgb = pd.read_csv(SIM / "xgboost_simulation_results.csv")
    return elo, dc, xgb


def get_prob(df, team):
    row = df[df["winner"] == team]
    return float(row["probability"].iloc[0]) if len(row) else 0.0


# ── Portugal path data ────────────────────────────────────────────────────────
# ELO and XGBoost path data from verified 100k simulation runs (see handoff notes).
# DC path loaded from CSV.

STAGES = ["Group stage", "Round of 32", "Round of 16",
          "Quarter-Final", "Semi-Final", "Runner-up", "Winner"]

PATH_ELO = {
    "Group stage":  0.07255,
    "Round of 32":  0.35219,
    "Round of 16":  0.25343,
    "Quarter-Final":0.15101,
    "Semi-Final":   0.08151,
    "Runner-up":    0.05078,
    "Winner":       0.03853,
}

PATH_XGB = {
    "Group stage":  0.14700,
    "Round of 32":  0.34200,
    "Round of 16":  0.22500,
    "Quarter-Final":0.13500,
    "Semi-Final":   0.07700,
    "Runner-up":    0.04100,
    "Winner":       0.03450,
}


def load_dc_path():
    df = pd.read_csv(SIM / "dixon_coles_portugal_path.csv")
    return dict(zip(df["stage"], df["probability"]))


# ── Chart 1: Win probability comparison (top teams, all 3 models) ─────────────

def chart_win_probability(elo, dc, xgb):
    # Union of top teams across all models
    top_elo = set(elo.head(10)["winner"])
    top_dc  = set(dc.head(10)["winner"])
    top_xgb = set(xgb.head(10)["winner"])
    teams = sorted(top_elo | top_dc | top_xgb,
                   key=lambda t: -(get_prob(elo, t) + get_prob(dc, t) + get_prob(xgb, t)))
    teams = teams[:12]

    x = np.arange(len(teams))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 7))
    bars_elo = ax.bar(x - w,   [get_prob(elo, t) * 100 for t in teams], w, label=MODEL_LABELS["elo"],  color=C_ELO,  alpha=0.88)
    bars_dc  = ax.bar(x,       [get_prob(dc,  t) * 100 for t in teams], w, label=MODEL_LABELS["dc"],   color=C_DC,   alpha=0.88)
    bars_xgb = ax.bar(x + w,   [get_prob(xgb, t) * 100 for t in teams], w, label=MODEL_LABELS["xgb"], color=C_XGB,  alpha=0.88)

    # Highlight Portugal bars
    for bars in (bars_elo, bars_dc, bars_xgb):
        for bar, team in zip(bars, teams):
            if team == "Portugal":
                bar.set_edgecolor(C_PORT)
                bar.set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(teams, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Win Probability (%)", fontsize=12)
    ax.set_title("FIFA World Cup 2026 — Champion Probability by Model\n(100,000 simulations each)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    # Ensemble average line for Portugal only
    port_avg = np.mean([get_prob(elo, "Portugal"), get_prob(dc, "Portugal"), get_prob(xgb, "Portugal")]) * 100
    port_idx = teams.index("Portugal")
    ax.annotate(f"Ensemble: {port_avg:.1f}%",
                xy=(port_idx, port_avg / 100 * 100),
                xytext=(port_idx + 1.5, port_avg + 1.5),
                fontsize=10, color=C_PORT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_PORT, lw=1.5))

    fig.tight_layout()
    out = CHARTS / "win_probability_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Chart 2: Portugal path comparison (all 3 models) ─────────────────────────

def chart_portugal_path(path_dc):
    x = np.arange(len(STAGES))
    w = 0.25

    elo_vals = [PATH_ELO[s] * 100 for s in STAGES]
    dc_vals  = [path_dc.get(s, 0) * 100 for s in STAGES]
    xgb_vals = [PATH_XGB[s] * 100 for s in STAGES]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, elo_vals, w, label=MODEL_LABELS["elo"],  color=C_ELO,  alpha=0.88)
    ax.bar(x,     dc_vals,  w, label=MODEL_LABELS["dc"],   color=C_DC,   alpha=0.88)
    ax.bar(x + w, xgb_vals, w, label=MODEL_LABELS["xgb"],  color=C_XGB,  alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.set_title("Portugal 2026 — Exit Stage Probability by Model\n(100,000 simulations each)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = CHARTS / "portugal_path_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Chart 3: Model agreement scatter (top 15 teams) ──────────────────────────

def chart_model_scatter(elo, dc, xgb):
    # Show ELO (x) vs XGBoost (y), bubble size = DC probability
    all_teams = set(elo.head(15)["winner"]) | set(xgb.head(15)["winner"])
    teams = [t for t in all_teams]

    elo_probs = np.array([get_prob(elo, t) * 100 for t in teams])
    xgb_probs = np.array([get_prob(xgb, t) * 100 for t in teams])
    dc_probs  = np.array([get_prob(dc,  t) * 100 for t in teams])

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(elo_probs, xgb_probs,
                         s=dc_probs * 60 + 40,
                         c=dc_probs, cmap="YlOrRd",
                         alpha=0.85, edgecolors="grey", linewidths=0.5)

    for i, team in enumerate(teams):
        offset = (0.3, 0.3)
        if team == "Portugal":
            offset = (0.4, -1.0)
            ax.scatter(elo_probs[i], xgb_probs[i], s=200,
                       facecolors="none", edgecolors=C_PORT, linewidths=2.5)
        ax.annotate(team, (elo_probs[i] + offset[0], xgb_probs[i] + offset[1]),
                    fontsize=8.5, ha="left")

    # 45-degree agreement line
    lim = max(elo_probs.max(), xgb_probs.max()) + 3
    ax.plot([0, lim], [0, lim], "--", color="grey", alpha=0.5, lw=1, label="ELO = XGBoost")

    plt.colorbar(scatter, ax=ax, label="Model B (ELO-Poisson/DC) probability %")
    ax.set_xlabel("Model A: ELO probability (%)", fontsize=12)
    ax.set_ylabel("Model C: XGBoost probability (%)", fontsize=12)
    ax.set_title("Model Agreement — ELO vs XGBoost\n(bubble size = ELO-Poisson/DC probability)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = CHARTS / "model_comparison_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Chart 4: Portugal summary card ────────────────────────────────────────────

def chart_portugal_summary(elo, dc, xgb, path_dc):
    p_elo = get_prob(elo, "Portugal") * 100
    p_dc  = get_prob(dc,  "Portugal") * 100
    p_xgb = get_prob(xgb, "Portugal") * 100
    p_avg = np.mean([p_elo, p_dc, p_xgb])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: model probabilities bar
    ax = axes[0]
    models  = [MODEL_LABELS["elo"], MODEL_LABELS["dc"], MODEL_LABELS["xgb"], "Ensemble avg"]
    probs   = [p_elo, p_dc, p_xgb, p_avg]
    colours = [C_ELO, C_DC, C_XGB, C_PORT]
    bars = ax.barh(models, probs, color=colours, alpha=0.88, height=0.5)
    for bar, prob in zip(bars, probs):
        ax.text(prob + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{prob:.2f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Tournament Win Probability (%)", fontsize=11)
    ax.set_title("Portugal Win Probability\nAll Three Models", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(probs) * 1.35)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Right: survival curve (proportion still in at each stage)
    ax2 = axes[1]
    stage_labels = ["Start", "R32", "R16", "QF", "SF", "Final", "Win"]

    def survival(path):
        s = [1.0]
        for stage in ["Round of 32", "Round of 16", "Quarter-Final", "Semi-Final", "Runner-up", "Winner"]:
            eliminated = path.get(stage, 0) if stage != "Winner" else 0
            s.append(s[-1] - (path.get(
                {"Round of 32": "Round of 32",
                 "Round of 16": "Round of 16",
                 "Quarter-Final": "Quarter-Final",
                 "Semi-Final": "Semi-Final",
                 "Runner-up": "Runner-up"}. get(stage, stage), 0) if stage != "Winner" else 0))
        return s

    # Build survival curves from path data
    def path_to_survival(path_dict):
        still_in = 1.0
        curve = [still_in]
        for stage in ["Group stage", "Round of 32", "Round of 16",
                      "Quarter-Final", "Semi-Final", "Runner-up"]:
            still_in -= path_dict.get(stage, 0)
            curve.append(max(still_in, 0))
        return curve

    surv_elo = path_to_survival(PATH_ELO)
    surv_xgb = path_to_survival(PATH_XGB)
    surv_dc  = path_to_survival(path_dc)

    x = range(len(stage_labels))
    ax2.plot(x, [v * 100 for v in surv_elo], "o-", color=C_ELO,  lw=2.2, label=MODEL_LABELS["elo"],  ms=6)
    ax2.plot(x, [v * 100 for v in surv_dc],  "s-", color=C_DC,   lw=2.2, label=MODEL_LABELS["dc"],   ms=6)
    ax2.plot(x, [v * 100 for v in surv_xgb], "^-", color=C_XGB,  lw=2.2, label=MODEL_LABELS["xgb"], ms=6)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(stage_labels, fontsize=10)
    ax2.set_ylabel("Portugal still in tournament (%)", fontsize=11)
    ax2.set_title("Portugal Survival Curve\nAll Three Models", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle("Portugal — FIFA World Cup 2026 Prediction Summary", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = CHARTS / "portugal_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Final report ──────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """# Portugal — FIFA World Cup 2026 Win Probability
## Final Model Comparison Report

**Analyst:** Abdul Mussavir
**Date:** 13 May 2026
**Project:** football_pred_analysis

---

## Research Question

> What is the probability that Portugal wins the FIFA World Cup 2026,
> and how does that estimate vary across different modelling methodologies?

---

## Methodology Summary

Three independent models were built, each using a different approach to estimate match outcomes.
Each model ran 100,000 Monte Carlo tournament simulations using the official FIFA 2026 bracket
structure (48 teams, 12 groups of 4, Round of 32 bracket, official third-place routing).

| Model | Approach | Key Input | Complexity |
|---|---|---|---|
| A — ELO Simulation | Win probability from ELO rating difference | ELO ratings (current) | Low |
| B — ELO-Poisson/DC | Scoreline simulation via ELO-calibrated Poisson + DC low-score correction | ELO ratings + calibrated avg goals | Medium |
| C — XGBoost | Match outcome classifier trained on historical features | ELO diff, form, H2H, goals, stage | High |

### Model B Note

A pure Dixon-Coles MLE model was tested but failed validation: it over-weighted regional goal
records (e.g. Iran consistently beating weak AFC opponents) and produced unrealistic champion
probabilities (Iran 31%, Spain 2.2%). The final Model B therefore uses ELO-calibrated Poisson
expected goals with a Dixon-Coles low-score correction factor (rho = {rho:.4f}). ELO ratings
already encode schedule difficulty through their iterative opponent-weighted update formula.

---

## Key Results

### Portugal Win Probability

| Model | Portugal | Spain | Argentina | France | England |
|---|---|---|---|---|---|
| A — ELO Simulation | **4.03%** | 27.6% | 17.0% | 11.3% | 9.0% |
| B — ELO-Poisson/DC | **2.53%** | 39.4% | 22.0% | 12.2% | 8.9% |
| C — XGBoost | **3.45%** | 18.3% | 16.0% | 9.7% | 10.5% |
| **Ensemble Average** | **~3.3%** | ~28.4% | ~18.3% | ~11.1% | ~9.5% |

**Conclusion:** Across all three independently-built models, Portugal's tournament win probability
sits in the **2.5–4.0% range**, with an ensemble estimate of approximately **3.3%**. This places
Portugal as a credible dark horse — roughly 8th–10th most likely champion — but not a favourite.

---

## Portugal Stage-by-Stage Analysis

| Stage | ELO | ELO-Poisson/DC | XGBoost |
|---|---|---|---|
| Group stage exit | 7.3% | 2.2% | 14.7% |
| Round of 32 exit | 35.2% | 32.8% | 34.2% |
| Round of 16 exit | 25.3% | 29.7% | 22.5% |
| Quarter-Final exit | 15.1% | 19.0% | 13.5% |
| Semi-Final exit | 8.2% | 8.3% | 7.7% |
| Runner-up | 5.1% | 5.3% | 4.1% |
| **Winner** | **3.9%** | **2.6%** | **3.5%** |

**Key observation:** All three models agree Portugal is most likely to exit at the Round of 32
(~33–35%), reflecting that Portugal qualifies from Group K comfortably (strong against
Uzbekistan and DR Congo) but then faces a significantly tougher opponent in the R32.

The XGBoost model gives Portugal a higher group stage exit risk (14.7%) compared to ELO (7.3%)
and ELO-Poisson/DC (2.2%). This likely reflects Portugal's mixed recent form index captured
by XGBoost features.

---

## Group K Analysis

Portugal's group draw is favourable by ELO:

| Team | ELO | Expected finish |
|---|---|---|
| Colombia | 1998 | 1st–2nd |
| **Portugal** | **1976** | **1st–2nd** |
| Uzbekistan | 1735 | 3rd |
| DR Congo | 1616 | 4th |

Colombia (ELO 1998) is technically stronger than Portugal by current ELO and is a genuine
challenge for the group win. Portugal's probability of advancing from the group is:
- ELO model: ~92.7%
- ELOPoissonDC: ~97.8%
- XGBoost: ~85.3%

---

## Model Comparison Notes

### Where models agree
- Portugal win probability: 2.5–4.0% across all three (tight band)
- Spain is the strong favourite in ELO and ELO-Poisson/DC
- Argentina and France are consistently 2nd–4th tier
- Round of 32 is Portugal's most likely exit point in all three models

### Where models disagree
- Spain's probability ranges from 18.3% (XGBoost) to 39.4% (ELO-Poisson/DC)
  XGBoost captures tactical upsets and recent form more heavily, flattening the ELO advantage
- Group stage exit varies: XGBoost sees more group-stage variance for Portugal (14.7% vs 2.2%)
  This reflects XGBoost picking up on Portugal's form_index and H2H features

### Known limitations
1. **Bracket approximation:** The official FIFA 2026 R32 slot pairings give Portugal a likely
   R32 opponent from Group J (Argentina/Algeria/Austria/Jordan) — not Group L (England/Croatia).
   Our implementation uses a section-based approximation that may make Portugal's R32 harder
   than reality. This likely **understates** Portugal's win probability slightly.
2. **Squad value not used in simulation:** Portugal's squad (€864.5M, 4th globally) is not
   directly reflected in the ELO or ELO-Poisson/DC models. XGBoost partially captures this
   via market value features.
3. **No injury model:** Ronaldo injury or suspension is not modelled. A sensitivity run would
   estimate the impact on Portugal's probability.
4. **ELO snapshot:** ELO ratings are as of late 2025. Any matches between then and the 2026
   tournament start are not captured.

---

## Final Estimate

**Portugal win probability: ~3.3% (ensemble)**

This is the average of three independently-built models using different data sources,
mathematical approaches, and complexity levels. The convergence to a tight 2.5–4.0% range
across all three models gives us reasonable confidence in this estimate.

For context, at 3.3%:
- Portugal is roughly a **30-to-1 dark horse**
- They have approximately a **1-in-30 chance** of winning the tournament
- The bookmakers' implied probability for Portugal is typically 5–8%, suggesting our
  models may slightly underestimate Portugal due to bracket approximation limitations

---

## Charts

- `outputs/charts/win_probability_comparison.png` — Champion probability by model (top 12 teams)
- `outputs/charts/portugal_path_comparison.png` — Portugal stage exit probability by model
- `outputs/charts/model_comparison_scatter.png` — ELO vs XGBoost model agreement scatter
- `outputs/charts/portugal_summary.png` — Portugal summary card with survival curves

---

*Generated by `src/run_phase7.py` | football_pred_analysis | Abdul Mussavir*
"""


def write_report(dc_rho=-0.1073):
    report = REPORT_TEMPLATE.format(rho=dc_rho)
    out = REPORTS / "final_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 7 - Model Comparison + Final Report")
    print("=" * 60)

    elo, dc, xgb = load_results()
    path_dc = load_dc_path()

    print(f"\nLoaded results: ELO {len(elo)} teams | DC {len(dc)} teams | XGB {len(xgb)} teams")

    p_elo = get_prob(elo, "Portugal")
    p_dc  = get_prob(dc,  "Portugal")
    p_xgb = get_prob(xgb, "Portugal")
    print(f"\nPortugal: ELO {p_elo:.2%} | DC {p_dc:.2%} | XGB {p_xgb:.2%} | Ensemble {np.mean([p_elo,p_dc,p_xgb]):.2%}")

    print("\nGenerating charts...")
    chart_win_probability(elo, dc, xgb)
    chart_portugal_path(path_dc)
    chart_model_scatter(elo, dc, xgb)
    chart_portugal_summary(elo, dc, xgb, path_dc)

    print("\nWriting final report...")
    write_report()

    print("\nPhase 7 complete.")
    print(f"  Charts:  {CHARTS}")
    print(f"  Report:  {REPORTS / 'final_report.md'}")


if __name__ == "__main__":
    main()
