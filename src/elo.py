"""
ELO-based win probability and match simulation.
Formula: E = 1 / (1 + 10^((R_B - R_A) / 400))
"""
import numpy as np

MAX_DRAW_PROB = 0.26   # draw probability when teams are exactly equal


def elo_win_prob(rating_a: float, rating_b: float) -> float:
    """P(A wins) from pure ELO ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def match_probs(rating_a: float, rating_b: float) -> tuple:
    """Return (p_win_A, p_draw, p_win_B). Draw peaks at equal ratings."""
    p = elo_win_prob(rating_a, rating_b)
    d = MAX_DRAW_PROB * (1.0 - abs(2.0 * p - 1.0))
    return p * (1 - d), d, (1 - p) * (1 - d)


def simulate_match(rating_a: float, rating_b: float,
                   knockout: bool = False, rng=None) -> str:
    """
    Simulate one match. Returns 'A', 'D', or 'B'.
    In knockout mode draws go to extra time/penalties (ELO edge dampened to 30%).
    """
    if rng is None:
        rng = np.random.default_rng()
    p_a, p_d, _ = match_probs(rating_a, rating_b)
    r = rng.random()
    if r < p_a:
        return 'A'
    if r < p_a + p_d:
        if knockout:
            p_ko = 0.5 + (elo_win_prob(rating_a, rating_b) - 0.5) * 0.30
            return 'A' if rng.random() < p_ko else 'B'
        return 'D'
    return 'B'


def simulate_scoreline(lambda_a: float, lambda_b: float, rng=None) -> tuple:
    """Poisson-draw scoreline for group-stage goal-difference tiebreaker."""
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.poisson(lambda_a)), int(rng.poisson(lambda_b))
