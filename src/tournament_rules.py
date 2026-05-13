"""FIFA 2026 tournament bracket and group-ranking helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


THIRD_PLACE_SLOT_COLUMNS = {
    "1A": "slot_1A",
    "1B": "slot_1B",
    "1D": "slot_1D",
    "1E": "slot_1E",
    "1G": "slot_1G",
    "1I": "slot_1I",
    "1K": "slot_1K",
    "1L": "slot_1L",
}

# Match order follows FIFA's published R32 match numbers 73-88.
ROUND_OF_32_TEMPLATE = [
    ("2A", "2B"),
    ("1E", "3"),
    ("1F", "2C"),
    ("1C", "2F"),
    ("1I", "3"),
    ("2E", "2I"),
    ("1A", "3"),
    ("1L", "3"),
    ("1D", "3"),
    ("1G", "3"),
    ("2K", "2L"),
    ("1H", "2J"),
    ("1B", "3"),
    ("1J", "2H"),
    ("1K", "3"),
    ("2D", "2G"),
]

R16_PAIRS = [(0, 2), (1, 4), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
QF_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
SF_PAIRS = [(0, 1), (2, 3)]


@lru_cache(maxsize=1)
def third_place_table(path: str = "data/raw/wc_2026_third_place_table.csv") -> dict[str, dict[str, str]]:
    """Load Annex-C third-place assignments keyed by the eight qualified groups."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"{table_path} is required for official 2026 third-place routing"
        )

    df = pd.read_csv(table_path, dtype=str)
    out: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        key = "".join(sorted(str(row["third_groups"])))
        out[key] = {
            slot: str(row[col])
            for slot, col in THIRD_PLACE_SLOT_COLUMNS.items()
        }
    return out


def resolve_slot(slot: str, group_results: dict[str, list], third_assignments: dict[str, str] | None = None) -> str:
    """Resolve a bracket slot such as 1A, 2K, or a third-place slot opponent."""
    if slot == "3":
        raise ValueError("Generic third-place slot must be resolved by caller")

    if slot.startswith("3"):
        group = slot[1]
        return group_results[group][2]["team"]

    position = int(slot[0]) - 1
    group = slot[1]
    return group_results[group][position]["team"]


def build_round_of_32_bracket(group_results: dict[str, list], best_thirds: list[str]) -> list[str]:
    """Build the official 32-slot R32 bracket using Annex-C third-place routing."""
    third_group_by_team = {
        results[2]["team"]: group
        for group, results in group_results.items()
    }
    third_groups = "".join(sorted(third_group_by_team[team] for team in best_thirds))
    assignments = third_place_table()[third_groups]

    bracket: list[str] = []
    for left_slot, right_slot in ROUND_OF_32_TEMPLATE:
        left = resolve_slot(left_slot, group_results)
        if right_slot == "3":
            right_assignment = assignments[left_slot]
            right = resolve_slot(right_assignment, group_results)
        else:
            right = resolve_slot(right_slot, group_results)
        bracket.extend([left, right])
    return bracket


def play_official_knockout_round(winners: list, pairs: list[tuple[int, int]], play_func) -> list:
    """Play a knockout round using FIFA's published match-number pairings."""
    return [play_func(winners[a], winners[b]) for a, b in pairs]
