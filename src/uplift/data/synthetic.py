"""Synthetic experiments with a known treatment effect.

Tests and the offline CI pipeline use these generators instead of the real
dataset: the counterfactual conversion probabilities are known exactly, so
recovery can be asserted rather than hoped for.

The treatment acts on the log-odds scale (``logit_effect``), which makes the
per-arm logistic specification correct by construction. The reported truth is
always the probability-scale ATE, averaged over the actual rows.
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
    logit_effect: float = 0.35,
    heterogeneity: float = 0.0,
    treatment_share: float = 0.5,
    outcome: str = "conversion",
) -> SyntheticExperiment:
    """Generate an experiment whose per-row true uplift is known exactly.

    ``logit_effect`` is the shift in log-odds applied to treated rows. When
    ``heterogeneity`` is non-zero the shift is larger for customers with an
    above-median spend history and smaller below it, centred so the average
    shift stays ``logit_effect``.
    """
    if logit_effect < 0.0:
        raise ValueError("logit_effect must be non-negative")
    rng = np.random.default_rng(seed)

    history = rng.gamma(shape=2.0, scale=60.0, size=n)
    recency = rng.integers(1, 13, size=n)
    mens = rng.binomial(1, 0.5, size=n)
    womens = rng.binomial(1, 0.5, size=n)
    newbie = rng.binomial(1, 0.4, size=n)
    zip_code = rng.choice(["Rural", "Surburban", "Urban"], size=n, p=[0.2, 0.6, 0.2])
    channel = rng.choice(["Multichannel", "Phone", "Web"], size=n, p=[0.2, 0.3, 0.5])

    # Linear in the covariates the learner actually sees (log1p(history) is
    # reconstructed from history by the tree learners; logistic uses raw history,
    # so keep the history term linear).
    logit = (
        -2.6
        + 0.0020 * history
        + 0.45 * mens
        + 0.20 * womens
        - 0.30 * newbie
        + 0.25 * (channel == "Multichannel")
        - 0.02 * recency
        + 0.10 * (zip_code == "Surburban")
    )
    baseline = _sigmoid(logit)

    above = (history > np.median(history)).astype(float)
    shift = logit_effect + heterogeneity * (above - 0.5)
    treated_prob = _sigmoid(logit + shift)

    treated = rng.binomial(1, treatment_share, size=n).astype(bool)
    prob = np.where(treated, treated_prob, baseline)
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
        pass  # same assignment; spend is analysed as the outcome
    elif outcome != "conversion":
        raise ValueError(f"unknown outcome {outcome!r}")

    # Shuffle so that neither arm nor effect is ordered by row position.
    order = rng.permutation(n)
    frame = frame.iloc[order].reset_index(drop=True)
    return SyntheticExperiment(
        frame=frame, true_uplift=(treated_prob - baseline)[order], seed=seed
    )
