from __future__ import annotations

import numpy as np
import pytest

from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment
from uplift.eval.uplift_at_k import (
    mean_predicted_uplift,
    top_k_mask,
    uplift_at_k,
    uplift_at_k_with_ci,
)


def test_uplift_at_k_matches_hand_computed_example() -> None:
    # 100 rows, top 10 = first 10 (score descending); 6 treated with 3 conversions,
    # 4 control with 1 conversion -> 0.5 - 0.25 = 0.25.
    t = np.zeros(100)
    t[:6] = 1
    t[10:60] = 1
    y = np.zeros(100)
    y[:3] = 1
    y[6] = 1
    score = np.arange(100, 0, -1, dtype=float)

    assert top_k_mask(score, 0.1).sum() == 10
    assert uplift_at_k(y, t, score, 0.1) == pytest.approx(0.25)


def test_top_k_mask_uses_ceiling_and_rejects_bad_k() -> None:
    mask = top_k_mask(np.arange(11, dtype=float), 0.1)
    assert mask.sum() == 2
    with pytest.raises(ValueError, match="k must be"):
        top_k_mask(np.arange(5.0), 0.0)
    with pytest.raises(ValueError, match="k must be"):
        top_k_mask(np.arange(5.0), 1.5)


def test_uplift_at_k_interval_is_deterministic_and_brackets_estimate() -> None:
    experiment = make_experiment(n=3000, seed=31)
    frame = experiment.frame
    t = HILLSTROM.is_treated(frame).to_numpy(dtype=float)
    y = frame["conversion"].to_numpy(dtype=float)
    score = experiment.true_uplift

    first = uplift_at_k_with_ci(y, t, score, 0.1, resamples=200, seed=4)
    second = uplift_at_k_with_ci(y, t, score, 0.1, resamples=200, seed=4)
    assert first == second
    assert first.ci_low <= first.uplift <= first.ci_high
    assert first.n_targeted == 300
    assert 0.0 < first.treated_share < 1.0


def test_mean_predicted_uplift_reads_the_targeted_group() -> None:
    score = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    assert mean_predicted_uplift(score, 0.4) == pytest.approx(0.45)


def test_uplift_at_k_rejects_single_arm_in_the_top_group() -> None:
    t = np.zeros(50)
    t[40:] = 1
    y = np.zeros(50)
    score = np.arange(50, dtype=float)  # highest scores are all control
    with pytest.raises(ValueError, match="single arm"):
        uplift_at_k(y, t, score, 0.1)
