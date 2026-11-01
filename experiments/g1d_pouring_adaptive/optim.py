"""Cross-entropy method for bounded action-sequence optimization."""
import numpy as np


def cem(objective, horizon, action_dim, lower, upper, *, population=256, elites=32, iterations=6, seed=0, smoothing=0.2):
    """Minimize objective(sequence); return best sequence and its cost.

    Objective receives [horizon, action_dim]. Bounds broadcast over the action
    dimension. A seeded generator makes the optimizer reproducible.
    """
    lo = np.broadcast_to(np.asarray(lower, float), (action_dim,))
    hi = np.broadcast_to(np.asarray(upper, float), (action_dim,))
    if horizon < 1 or action_dim < 1 or population < 2 or not 1 <= elites <= population or iterations < 1 or not 0 <= smoothing < 1 or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo >= hi):
        raise ValueError("invalid optimizer settings")
    rng = np.random.default_rng(seed)
    mean = np.broadcast_to((lo + hi) / 2, (horizon, action_dim)).copy()
    std = np.broadcast_to((hi - lo) / 2, mean.shape).copy()
    best, best_cost = None, float("inf")
    for _ in range(iterations):
        candidates = np.clip(rng.normal(mean, std, (population, horizon, action_dim)), lo, hi)
        costs = np.array([objective(x) for x in candidates], dtype=float)
        if costs.shape != (population,) or not np.isfinite(costs).all():
            raise ValueError("objective must return finite scalar costs")
        idx = np.argsort(costs)[:elites]
        if costs[idx[0]] < best_cost:
            best, best_cost = candidates[idx[0]].copy(), float(costs[idx[0]])
        selected = candidates[idx]
        mean = smoothing * mean + (1 - smoothing) * selected.mean(axis=0)
        std = smoothing * std + (1 - smoothing) * selected.std(axis=0)
    return best, best_cost
