"""
Phase 5 - ELO-calibrated Poisson / Dixon-Coles scoreline model.

Usage:
    python src/run_phase5.py
"""

import time
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.dixon_coles import ELOPoissonDC, run_monte_carlo_dc, run_portugal_path_dc

N_SIMS = 100_000


def main():
    print("=" * 60)
    print("Phase 5 - ELO-calibrated Poisson / DC scoreline model")
    print("=" * 60)

    intl   = pd.read_csv("data/raw/international_results.csv")
    elo_df = pd.read_csv("data/raw/elo_ratings.csv")

    print(f"\nLoaded: {len(intl):,} international results | {len(elo_df):,} ELO rows")

    print("\nFitting model...")
    t0 = time.time()
    model = ELOPoissonDC()
    model.fit(elo_df, intl_df=intl)
    print(f"  Fit elapsed: {time.time() - t0:.1f} s")

    Path("models").mkdir(exist_ok=True)
    model.save("models/dixon_coles_params.json")

    print("\nSanity checks (neutral ground):")
    checks = [
        ("Japan",    "Spain"),
        ("Japan",    "Portugal"),
        ("Portugal", "Colombia"),
        ("Spain",    "Argentina"),
        ("Portugal", "Spain"),
        ("Iran",     "Portugal"),
        ("Iran",     "France"),
    ]
    for a, b in checks:
        lam, mu = model.expected_goals(a, b)
        pw, pd_, pl = model.match_probs(a, b)
        print(f"  {a:10s} vs {b:10s} | xG {lam:.2f}-{mu:.2f} | "
              f"W/D/L {pw:.1%}/{pd_:.1%}/{pl:.1%}")

    print(f"\nRunning {N_SIMS:,} tournament simulations...")
    Path("simulation").mkdir(exist_ok=True)
    t0 = time.time()
    results = run_monte_carlo_dc(model, N_SIMS, seed=42)
    elapsed = time.time() - t0
    results.to_csv("simulation/dixon_coles_simulation_results.csv", index=False)
    print(f"  Elapsed: {elapsed/60:.1f} min")
    print("\nTop 15:")
    print(results.head(15).to_string(index=False))

    port = results[results["winner"] == "Portugal"]
    p = float(port["probability"].iloc[0]) if len(port) else 0.0
    print(f"\nPortugal win probability (ELOPoissonDC): {p:.2%}")

    print(f"\nRunning Portugal path analysis ({N_SIMS:,} sims)...")
    t0 = time.time()
    path_counts = run_portugal_path_dc(model, N_SIMS, seed=42)
    path_df = pd.DataFrame([
        {"stage": k, "count": v, "probability": v / N_SIMS}
        for k, v in path_counts.items()
    ])
    path_df.to_csv("simulation/dixon_coles_portugal_path.csv", index=False)
    print(f"  Elapsed: {(time.time() - t0)/60:.1f} min")
    print(path_df.to_string(index=False))


if __name__ == "__main__":
    main()
