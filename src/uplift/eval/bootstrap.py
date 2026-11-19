"""Seeded bootstrap helpers shared by every interval in the project."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from uplift.config import ALPHA, BOOTSTRAP_RESAMPLES, SEED


def bootstrap_indices(n: int, resamples: int, seed: int) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        yield rng.integers(0, n, size=n)


def percentile_interval(
    samples: list[float] | np.ndarray, alpha: float = ALPHA
) -> tuple[float, float]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"), float("nan"))
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def bootstrap_statistic(
    statistic: Callable[[np.ndarray], float],
    n: int,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> tuple[float, float]:
    samples = [statistic(idx) for idx in bootstrap_indices(n, resamples, seed)]
    low, high = percentile_interval(samples, alpha)
    return low, high
