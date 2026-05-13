"""
Scrapes current ELO ratings from eloratings.net and saves to data/raw/elo_ratings.csv.
Run from project root: python src/scrape_elo.py

eloratings.net is JS-rendered, so we try their known JSON endpoints first.
If those fail, instructions for manual download are printed.
"""

import requests
import pandas as pd
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "data" / "raw" / "elo_ratings.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# eloratings.net exposes ranking data via this endpoint
JSON_URL = "https://www.eloratings.net/api/top"


def try_json_api():
    try:
        resp = requests.get(JSON_URL, headers=HEADERS, timeout=15)
        if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/json"):
            data = resp.json()
            return data
    except Exception:
        pass
    return None


def try_world_page():
    """Try to parse the preloaded JSON blob embedded in the page source."""
    import re
    try:
        resp = requests.get("https://www.eloratings.net/World", headers=HEADERS, timeout=15)
        # Look for JSON array embedded in script tags
        match = re.search(r'rankings\s*=\s*(\[.*?\]);', resp.text, re.DOTALL)
        if match:
            import json
            return json.loads(match.group(1))
    except Exception:
        pass
    return None


def scrape_elo():
    print("Attempting JSON API...")
    data = try_json_api()

    if data is None:
        print("JSON API failed. Trying page source embed...")
        data = try_world_page()

    if data and isinstance(data, list):
        rows = []
        for i, entry in enumerate(data):
            if isinstance(entry, dict):
                rows.append({
                    "rank": entry.get("rank", i + 1),
                    "country": entry.get("name", entry.get("country", "")),
                    "rating": entry.get("rating", entry.get("elo", "")),
                    "confederation": entry.get("confederation", entry.get("conf", "")),
                })
        if rows:
            df = pd.DataFrame(rows)
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df.dropna(subset=["rating"], inplace=True)
            df.to_csv(OUTPUT, index=False)
            print(f"Saved {len(df)} teams to {OUTPUT}")
            return df

    # Manual fallback
    print("\nAutomatic scrape failed (site requires JS rendering).")
    print("Manual steps:")
    print("  1. Open https://www.eloratings.net/World in your browser")
    print("  2. Press Ctrl+A to select all, then Ctrl+C to copy")
    print("  3. Paste into Excel/Google Sheets and clean to: rank, country, rating, confederation")
    print("  4. Save as: data/raw/elo_ratings.csv")
    print("\nAlternative: The kaggle dataset 'eloratings' by tadhgfitzgerald has historical ELO data.")
    print("  URL: https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now")
    print("  Save as: data/raw/elo_ratings.csv")


if __name__ == "__main__":
    scrape_elo()
