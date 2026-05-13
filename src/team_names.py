"""
Canonical team name normalization.

Canonical = names used in international_results.csv (primary dataset).
All other files normalize to these names before any join or lookup.
"""

# Maps variant names → canonical name used throughout this project.
CANONICAL_MAP: dict[str, str] = {
    # ELO file uses "Czechia"; results CSV (and WC groups) use "Czech Republic"
    "Czechia":                        "Czech Republic",
    # ELO file uses long name; results CSV and WC groups use "DR Congo"
    "Democratic Republic of Congo":   "DR Congo",
    # WC groups CSV lacks the cedilla accent; results CSV has it
    "Curacao":                        "Curaçao",
    # national_teams.csv (Transfermarkt) uses "Turkiye"; all football datasets use "Turkey"
    "Turkiye":                        "Turkey",
    # national_teams.csv uses hyphen; results CSV uses " and "
    "Bosnia-Herzegovina":             "Bosnia and Herzegovina",
    # Minor whitespace variant
    "Bosnia & Herzegovina":           "Bosnia and Herzegovina",
    # Encoding variant that may appear in some CSV reads
    "Curaçao":                   "Curaçao",
}


def norm(name: str) -> str:
    """Return the canonical team name.  Strips non-breaking spaces first."""
    if not isinstance(name, str):
        return name
    name = name.replace("\xa0", " ").strip()
    return CANONICAL_MAP.get(name, name)
