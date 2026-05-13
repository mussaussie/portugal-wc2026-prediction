import pandas as pd


def _fake_group_results():
    results = {}
    for group in "ABCDEFGHIJKL":
        results[group] = [
            {"team": f"{group}1", "pts": 9, "gd": 6, "gf": 8},
            {"team": f"{group}2", "pts": 6, "gd": 2, "gf": 5},
            {"team": f"{group}3", "pts": 3, "gd": 0, "gf": 4},
            {"team": f"{group}4", "pts": 0, "gd": -8, "gf": 1},
        ]
    return results


def test_official_bracket_has_no_round_of_32_same_group_rematches():
    from src.simulation import _build_bracket

    group_results = _fake_group_results()
    best_thirds = [f"{g}3" for g in "ABCDEFGH"]
    bracket = _build_bracket(group_results, best_thirds)
    team_group = {f"{g}{pos}": g for g in "ABCDEFGHIJKL" for pos in range(1, 5)}

    for i in range(0, len(bracket), 2):
        left, right = bracket[i], bracket[i + 1]
        assert team_group[left] != team_group[right], (left, right)


def test_tournament_classification_handles_real_competition_names():
    from src.tournament_classification import classify_tournament

    assert classify_tournament("FIFA World Cup qualification") == "WCQ"
    assert classify_tournament("UEFA Euro qualification") == "EuroQ"
    assert classify_tournament("Copa América") == "CopaAmerica"
    assert classify_tournament("African Cup of Nations") == "AFCON"
    assert classify_tournament("AFC Asian Cup qualification") == "AsianCupQ"


def test_saved_simulation_outputs_use_canonical_team_names():
    for path in [
        "simulation/elo_simulation_results.csv",
        "simulation/dixon_coles_simulation_results.csv",
        "simulation/xgboost_simulation_results.csv",
    ]:
        df = pd.read_csv(path)
        assert "Democratic Republic of Congo" not in set(df["winner"])
        assert abs(df["probability"].sum() - 1.0) < 1e-9
