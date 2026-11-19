"""Uplift at the top k% of a score, with a seeded bootstrap interval."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from uplift.config import ALPHA, BOOTSTRAP_RESAMPLES, SEED
from uplift.eval.bootstrap import bootstrap_indices, percentile_interval


@dataclass(frozen=True)
class UpliftAtK:
    k: float
    n_targeted: int
    uplift: float
    ci_low: float
    ci_high: float
    treated_share: float
    resamples: int


def top_k_mask(score: np.ndarray, k: float) -> np.ndarray:
    if not 0.0 < k <= 1.0:
        raise ValueError("k must be in (0, 1]")
    score = np.asarray(score, dtype=float)
    n_top = max(1, math.ceil(k * len(score)))
    order = np.argsort(-score, kind="stable")
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:n_top]] = True
    return mask


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, k: float = 0.1) -> float:
    mask = top_k_mask(score, k)
    y_top = np.asarray(y, dtype=float)[mask]
    t_top = np.asarray(t, dtype=float)[mask]
    n_treated = t_top.sum()
    n_control = len(t_top) - n_treated
    if n_treated == 0 or n_control == 0:
        raise ValueError("top-k selection contains a single arm")
    return float(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())


def uplift_at_k_with_ci(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    k: float = 0.1,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> UpliftAtK:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    score = np.asarray(score, dtype=float)
    mask = top_k_mask(score, k)
    value = uplift_at_k(y, t, score, k)

    samples: list[float] = []
    for idx in bootstrap_indices(len(y), resamples, seed):
        y_b, t_b, mask_b = y[idx], t[idx], mask[idx]
        n_treated = t_b[mask_b].sum()
        n_control = mask_b.sum() - n_treated
        if n_treated == 0 or n_control == 0:
            continue
        samples.append(float(y_b[mask_b & (t_b == 1)].mean() - y_b[mask_b & (t_b == 0)].mean()))
    low, high = percentile_interval(samples, alpha)
    return UpliftAtK(
        k=k,
        n_targeted=int(mask.sum()),
        uplift=value,
        ci_low=low,
        ci_high=high,
        treated_share=float(t[mask].mean()),
        resamples=resamples,
    )


def mean_predicted_uplift(score: np.ndarray, k: float = 0.1) -> float:
    """Average model score inside the targeted group (for predicted-vs-observed)."""
    return float(np.asarray(score, dtype=float)[top_k_mask(score, k)].mean())
