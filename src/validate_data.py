"""
Validates all raw datasets are present and correctly shaped.
Run from project root: python src/validate_data.py
"""

import pandas as pd
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"

CHECKS = {
    "international_results.csv": {
        "min_rows": 40_000,
        "required_cols": ["date", "home_team", "away_team", "home_score", "away_score", "tournament"],
    },
    "wc_matches.csv": {
        "min_rows": 800,
        "required_cols": ["Year", "Stage", "Home Team Name", "Away Team Name", "Home Team Goals", "Away Team Goals"],
    },
    "wc_summary.csv": {
        "min_rows": 20,
        "required_cols": [],
    },
    "elo_ratings.csv": {
        "min_rows": 5000,
        "required_cols": ["date", "team", "rating", "change"],
    },
    "wc_2026_groups.csv": {
        "min_rows": 48,
        "required_cols": ["group", "team", "confederation"],
    },
}

OPTIONAL = ["wc_odds_2018.csv", "wc_odds_2022.csv"]


def check_dataset(filename, rules):
    path = RAW / filename
    if not path.exists():
        print(f"  MISSING   {filename}")
        return False

    df = pd.read_csv(path)
    issues = []

    if len(df) < rules["min_rows"]:
        issues.append(f"only {len(df)} rows (expected >={rules['min_rows']})")

    missing_cols = [c for c in rules["required_cols"] if c not in df.columns]
    if missing_cols:
        issues.append(f"missing columns: {missing_cols}")

    if issues:
        print(f"  WARNING   {filename}: {'; '.join(issues)}")
        return False

    if filename == "wc_matches.csv":
        score_cols = ["Home Team Goals", "Away Team Goals"]
        valid = df.dropna(subset=score_cols)
        duplicate_valid = valid.duplicated(subset=[
            "Year", "Stage", "Home Team Name", "Away Team Name",
            "Home Team Goals", "Away Team Goals"
        ]).sum()
        if len(valid) < 800:
            print(
                f"  WARNING   {filename}: {len(valid):,} valid scored rows; "
                f"{duplicate_valid} duplicate valid matches"
            )
            return False
        if duplicate_valid:
            print(
                f"  NOTE      {filename}: {duplicate_valid} duplicate valid matches; "
                "EDA scripts drop these before analysis"
            )

    print(f"  OK        {filename} — {len(df):,} rows, {len(df.columns)} cols")
    return True


def main():
    print("\n=== Phase 1 Data Validation ===\n")
    results = []
    for fname, rules in CHECKS.items():
        results.append(check_dataset(fname, rules))

    print("\n--- Optional datasets ---")
    for fname in OPTIONAL:
        path = RAW / fname
        status = "present" if path.exists() else "not downloaded"
        print(f"  {fname}: {status}")

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} core datasets ready.")
    if passed == total:
        print("Phase 1 complete — proceed to 02_eda.ipynb")
    else:
        print("Download missing datasets before proceeding.")


if __name__ == "__main__":
    main()
