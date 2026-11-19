"""Synthetic experiments with a known treatment effect.

Tests and the offline CI pipeline use these generators instead of the real
dataset: the true ATE and the true subgroup heterogeneity are known exactly,
so recovery can be asserted rather than hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from uplift.config import SEED


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _history_segment(history: np.ndarray) -> np.ndarray:
    edges = np.array([0, 100, 200, 350, 500, 750, 1000], dtype=float)
    labels = np.array(
        [
            "1) $0 - $100",
            "2) $100 - $200",
            "3) $200 - $350",
            "4) $350 - $500",
            "5) $500 - $750",
            "6) $750 - $1,000",
            "7) $1,000 +",
        ]
    )
    idx = np.clip(np.searchsorted(edges, history, side="right") - 1, 0, len(labels) - 1)
    return labels[idx]


@dataclass(frozen=True)
class SyntheticExperiment:
    frame: pd.DataFrame
    true_uplift: np.ndarray
    seed: int

    @property
    def ate(self) -> float:
        return float(np.mean(self.true_uplift))


def make_experiment(
    n: int = 4000,
    seed: int = SEED,
    *,
    tau: float = 0.05,
    heterogeneity: float = 0.0,
    treatment_share: float = 0.5,
    outcome: str = "conversion",
) -> SyntheticExperiment:
    """Generate an experiment whose per-row true uplift is known.

    ``tau`` is the average absolute uplift in conversion probability. When
    ``heterogeneity`` is non-zero, the effect is larger for customers with an
    above-median spend history and correspondingly smaller below it (mean
    effect stays ``tau``).
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    rng = np.random.default_rng(seed)

    history = rng.gamma(shape=2.0, scale=60.0, size=n)
    recency = rng.integers(1, 13, size=n)
    mens = rng.binomial(1, 0.5, size=n)
    womens = rng.binomial(1, 0.5, size=n)
    newbie = rng.binomial(1, 0.4, size=n)
    zip_code = rng.choice(["Rural", "Surburban", "Urban"], size=n, p=[0.2, 0.6, 0.2])
    channel = rng.choice(["Multichannel", "Phone", "Web"], size=n, p=[0.2, 0.3, 0.5])

    logit = (
        -3.0
        + 0.9 * np.log1p(history) / 3.0
        + 0.45 * mens
        + 0.20 * womens
        - 0.30 * newbie
        + 0.25 * (channel == "Multichannel")
        - 0.15 * recency / 12.0
    )
    baseline = np.clip(_sigmoid(logit), 0.01, 0.60)

    above = (history > np.median(history)).astype(float)
    uplift = tau + heterogeneity * (above - 0.5)

    treated = rng.binomial(1, treatment_share, size=n).astype(bool)
    prob = np.clip(baseline + treated * uplift, 0.0, 0.95)
    conversion = rng.binomial(1, prob).astype(int)
    spend = np.where(conversion == 1, rng.gamma(shape=2.0, scale=40.0, size=n), 0.0)
    visit = rng.binomial(1, np.clip(prob * 3.0, 0.0, 1.0)).astype(int)

    frame = pd.DataFrame(
        {
            "recency": recency.astype(int),
            "history_segment": _history_segment(history),
            "history": history,
            "mens": mens.astype(int),
            "womens": womens.astype(int),
            "zip_code": zip_code,
            "newbie": newbie.astype(int),
            "channel": channel,
            "segment": np.where(treated, "Mens E-Mail", "No E-Mail"),
            "visit": visit,
            "conversion": conversion,
            "spend": spend,
        }
    )
    if outcome == "spend":
        # same assignment, spend is the outcome of interest
        pass
    elif outcome != "conversion":
        raise ValueError(f"unknown outcome {outcome!r}")

    # Shuffle so that neither arm nor effect is ordered by row position.
    order = rng.permutation(n)
    frame = frame.iloc[order].reset_index(drop=True)
    return SyntheticExperiment(
        frame=frame, true_uplift=uplift[order], seed=seed
    )
