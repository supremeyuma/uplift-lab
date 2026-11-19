"""Qini curve and coefficient.

The Qini curve is cumulative incremental conversions when customers are
targeted in score order:

    Q(k) = (conversions among treated in the top k) * N / N_treated
         - (conversions among control in the top k) * N / N_control

The coefficient is the area between the curve and the random-targeting line,
divided by the area of an oracle ranking. 1.0 means the ranking puts every
treated converter first; 0.0 means no better than random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uplift.config import ALPHA, BOOTSTRAP_RESAMPLES, SEED
from uplift.eval.bootstrap import bootstrap_indices, percentile_interval


@dataclass(frozen=True)
class QiniResult:
    value: float
    ci_low: float
    ci_high: float
    resamples: int


@dataclass(frozen=True)
class QiniCurve:
    fractions: np.ndarray
    qini: np.ndarray

    def random_line(self, rate_difference: float) -> np.ndarray:
        return self.fractions * len(self.fractions) * rate_difference


def _prefix_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> QiniCurve:
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    y_sorted = np.asarray(y, dtype=float)[order]
    t_sorted = np.asarray(t, dtype=float)[order]
    n = len(y_sorted)
    n_treated = t_sorted.sum()
    n_control = n - n_treated
    if n_treated == 0 or n_control == 0:
        raise ValueError("Qini needs both treatment arms")

    cumulative_treated = np.cumsum(y_sorted * t_sorted)
    cumulative_control = np.cumsum(y_sorted * (1.0 - t_sorted))
    qini = cumulative_treated * (n / n_treated) - cumulative_control * (n / n_control)
    return QiniCurve(fractions=np.arange(1, n + 1) / n, qini=qini)


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> QiniCurve:
    return _prefix_curve(y, t, score)


def _area(curve: QiniCurve) -> float:
    fractions = np.concatenate([[0.0], curve.fractions])
    values = np.concatenate([[0.0], curve.qini])
    return float(np.trapezoid(values, fractions))


def qini_area(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    return _area(_prefix_curve(y, t, score))


def analytic_max_area(y: np.ndarray, t: np.ndarray) -> float:
    """Area of the best possible Qini curve, in closed form.

    The oracle orders every treated converter first, then rows that move the
    curve by zero, then every control converter (the only rows that move it
    down). Areas are in fraction-of-population units.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    n_treated = t.sum()
    n_control = n - n_treated
    if n_treated == 0 or n_control == 0:
        raise ValueError("Qini needs both treatment arms")
    treated_conversions = float(np.sum(y * t))
    control_conversions = float(np.sum(y * (1.0 - t)))
    c_t = n / n_treated
    c_c = n / n_control

    area = c_t * treated_conversions**2 / (2.0 * n)
    width = n - treated_conversions - control_conversions
    area += c_t * treated_conversions * width / n
    area += (
        control_conversions * treated_conversions * c_t
        - c_c * control_conversions**2 / 2.0
    ) / n
    return float(area)


def random_area(y: np.ndarray, t: np.ndarray) -> float:
    """Expected area under random targeting: ATE * N * (N+1) / (2N)."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(y)
    rate_difference = y[t == 1].mean() - y[t == 0].mean()
    return float(rate_difference * n / 2.0)


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    denominator = analytic_max_area(y, t) - random_area(y, t)
    if denominator <= 0.0:
        return 0.0
    return float((qini_area(y, t, score) - random_area(y, t)) / denominator)


def qini_with_ci(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> QiniResult:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    score = np.asarray(score, dtype=float)
    value = qini_coefficient(y, t, score)

    samples: list[float] = []
    for idx in bootstrap_indices(len(y), resamples, seed):
        y_b, t_b = y[idx], t[idx]
        if y_b[t_b == 1].size == 0 or y_b[t_b == 0].size == 0:
            continue
        samples.append(qini_coefficient(y_b, t_b, score[idx]))
    low, high = percentile_interval(samples, alpha)
    return QiniResult(value=value, ci_low=low, ci_high=high, resamples=resamples)


def qini_of_random_is_zero(
    y: np.ndarray,
    t: np.ndarray,
    seed: int = SEED,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> bool:
    """True when the random-ranking coefficient interval contains zero."""
    rng = np.random.default_rng(seed)
    result = qini_with_ci(
        y, t, rng.normal(size=len(y)), resamples=resamples, seed=seed + 1
    )
    return bool(result.ci_low <= 0.0 <= result.ci_high)
