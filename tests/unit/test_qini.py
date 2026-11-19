from __future__ import annotations

import numpy as np
import pytest

from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment
from uplift.eval.qini import (
    analytic_max_area,
    qini_area,
    qini_coefficient,
    qini_curve,
    qini_with_ci,
    random_area,
)


def _experiment_arrays(seed: int = 5, n: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    experiment = make_experiment(n=n, seed=seed)
    frame = experiment.frame
    y = frame["conversion"].to_numpy(dtype=float)
    t = HILLSTROM.is_treated(frame).to_numpy(dtype=float)
    return y, t, experiment.true_uplift


def test_qini_of_perfect_ranking_equals_analytic_max() -> None:
    y = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=float)
    t = np.array([1, 1, 0, 1, 0, 1, 0, 0], dtype=float)
    # treated converters first, zero-increment rows in the middle, control converters last
    perfect_score = y * (2.0 * t - 1.0)

    assert qini_area(y, t, perfect_score) == pytest.approx(analytic_max_area(y, t), rel=1e-12)
    assert qini_coefficient(y, t, perfect_score) == pytest.approx(1.0, abs=1e-12)
    assert analytic_max_area(y, t) > random_area(y, t)


def test_analytic_max_area_matches_brute_force_oracle() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 60
        t = rng.binomial(1, 0.4, n).astype(float)
        y = rng.binomial(1, 0.3, n).astype(float)
        if t.sum() in (0, n):
            continue
        # oracle: treated converters, then zero-increment rows, then control converters
        oracle_score = np.where(y * t == 1, 2.0, np.where(y * (1 - t) == 1, -1.0, 0.0))
        assert analytic_max_area(y, t) == pytest.approx(qini_area(y, t, oracle_score), rel=1e-12)


def test_qini_of_random_ranking_is_near_zero() -> None:
    y, t, _ = _experiment_arrays()
    rng = np.random.default_rng(42)
    result = qini_with_ci(y, t, rng.normal(size=len(y)), resamples=400, seed=7)
    assert abs(result.value) < 0.1
    assert result.ci_low <= 0.0 <= result.ci_high


def test_qini_prefers_the_oracle_ranking() -> None:
    experiment = make_experiment(n=4000, seed=5, heterogeneity=1.0)
    frame = experiment.frame
    y = frame["conversion"].to_numpy(dtype=float)
    t = HILLSTROM.is_treated(frame).to_numpy(dtype=float)
    rng = np.random.default_rng(11)
    oracle = qini_coefficient(y, t, experiment.true_uplift)
    random = qini_coefficient(y, t, rng.normal(size=len(y)))
    assert 0.0 < random + 0.05 < oracle <= 1.0


def test_qini_curve_has_one_point_per_row() -> None:
    y, t, true_uplift = _experiment_arrays(n=500)
    curve = qini_curve(y, t, true_uplift)
    assert len(curve.fractions) == len(curve.qini) == 500
    assert curve.fractions[-1] == pytest.approx(1.0)
    assert np.all(np.diff(curve.fractions) > 0)
    assert curve.qini[-1] == pytest.approx(
        y[t == 1].sum() * len(y) / t.sum() - y[t == 0].sum() * len(y) / (len(t) - t.sum())
    )


def test_qini_interval_is_deterministic_for_a_seed() -> None:
    y, t, true_uplift = _experiment_arrays(n=800)
    first = qini_with_ci(y, t, true_uplift, resamples=200, seed=99)
    second = qini_with_ci(y, t, true_uplift, resamples=200, seed=99)
    assert first == second
    assert first.ci_low <= first.value <= first.ci_high


def test_qini_rejects_single_arm() -> None:
    y = np.array([0, 1, 0, 0], dtype=float)
    with pytest.raises(ValueError, match="both treatment arms"):
        qini_coefficient(y, np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError, match="both treatment arms"):
        analytic_max_area(y, np.zeros(4))
