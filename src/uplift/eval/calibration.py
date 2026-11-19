"""Calibration of uplift scores: predicted vs observed, by score decile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uplift.config import ALPHA, SEED
from uplift.eval.bootstrap import percentile_interval


@dataclass(frozen=True)
class CalibrationBin:
    decile: int
    n: int
    predicted_mean: float
    observed_uplift: float
    ci_low: float
    ci_high: float


def _rank_bins(score: np.ndarray, bins: int) -> np.ndarray:
    if bins < 1:
        raise ValueError("bins must be positive")
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    assignment = np.empty(len(score), dtype=int)
    for decile, indices in enumerate(np.array_split(order, bins)):
        assignment[indices] = decile
    return assignment


def calibration_table(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    *,
    bins: int = 10,
    resamples: int = 200,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> list[CalibrationBin]:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    score = np.asarray(score, dtype=float)
    assignment = _rank_bins(score, bins)
    rng = np.random.default_rng(seed)

    table: list[CalibrationBin] = []
    for decile in range(bins):
        mask = assignment == decile
        y_bin, t_bin = y[mask], t[mask]
        n_treated = int(t_bin.sum())
        n_control = int(len(t_bin) - n_treated)
        if n_treated == 0 or n_control == 0:
            raise ValueError(f"decile {decile} contains a single arm")
        observed = float(y_bin[t_bin == 1].mean() - y_bin[t_bin == 0].mean())

        samples: list[float] = []
        for _ in range(resamples):
            idx = rng.integers(0, len(y_bin), size=len(y_bin))
            y_b, t_b = y_bin[idx], t_bin[idx]
            n_t = t_b.sum()
            if n_t == 0 or n_t == len(t_b):
                continue
            samples.append(float(y_b[t_b == 1].mean() - y_b[t_b == 0].mean()))
        low, high = percentile_interval(samples, alpha)

        table.append(
            CalibrationBin(
                decile=decile,
                n=int(mask.sum()),
                predicted_mean=float(score[mask].mean()),
                observed_uplift=observed,
                ci_low=low,
                ci_high=high,
            )
        )
    return table


def calibration_error(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, *, bins: int = 10
) -> float:
    """Sample-weighted mean absolute gap between predicted and observed uplift."""
    table = calibration_table(y, t, score, bins=bins, resamples=2)
    total = sum(row.n for row in table)
    weighted = sum(
        row.n * abs(row.predicted_mean - row.observed_uplift) for row in table
    )
    return float(weighted / total)


def is_overconfident(y: np.ndarray, t: np.ndarray, score: np.ndarray, *, bins: int = 10) -> bool:
    """A score is overconfident when it claims more uplift than it delivers."""
    table = calibration_table(y, t, score, bins=bins, resamples=2)
    predicted = sum(row.n * row.predicted_mean for row in table) / sum(row.n for row in table)
    observed = sum(row.n * row.observed_uplift for row in table) / sum(row.n for row in table)
    return bool(predicted > observed * 1.25 and predicted - observed > 0.01)
