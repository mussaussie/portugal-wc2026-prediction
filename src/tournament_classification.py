"""Tournament type classification shared by feature engineering and EDA."""


def classify_tournament(tournament: str) -> str:
    """Return a stable tournament bucket for modelling features."""
    t = str(tournament)
    tl = t.lower()

    if "fifa world cup" in tl and "qualification" not in tl and "qualif" not in tl:
        return "WC"
    if "fifa world cup qualification" in tl:
        return "WCQ"
    if "uefa euro qualification" in tl or "european championship qualification" in tl:
        return "EuroQ"
    if "uefa euro" in tl or "european championship" in tl:
        return "Euro"
    if "copa america qualification" in tl or "copa américa qualification" in tl:
        return "CopaAmericaQ"
    if "copa america" in tl or "copa américa" in tl:
        return "CopaAmerica"
    if "african cup of nations qualification" in tl or "africa cup qualification" in tl:
        return "AFCONQ"
    if "african cup of nations" in tl or "africa cup" in tl or "afcon" in tl:
        return "AFCON"
    if "afc asian cup qualification" in tl or "asian cup qualification" in tl:
        return "AsianCupQ"
    if "afc asian cup" in tl or "asian cup" in tl:
        return "AsianCup"
    if "gold cup qualification" in tl:
        return "GoldCupQ"
    if "gold cup" in tl:
        return "GoldCup"
    if "nations league qualification" in tl:
        return "NationsLeagueQ"
    if "nations league" in tl:
        return "NationsLeague"
    if "friendly" in t:
        return "Friendly"
    if "qualification" in tl or "qualif" in tl:
        return "OtherQ"
    return "Other"
