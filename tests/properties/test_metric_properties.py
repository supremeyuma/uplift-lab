"""Properties the metrics must hold: coverage, determinism, null behaviour."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment
from uplift.eval.qini import qini_with_ci
from uplift.eval.uplift_at_k import uplift_at_k_with_ci


def test_uplift_at_k_bootstrap_ci_covers_truth() -> None:
    draws = 60
    covered = 0
    for seed in range(draws):
        experiment = make_experiment(n=1500, seed=seed, logit_effect=0.5)
        frame = experiment.frame
        t = HILLSTROM.is_treated(frame).to_numpy(dtype=float)
        y = frame["conversion"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed + 10_000)
        score = experiment.true_uplift + rng.normal(0.0, 0.005, len(y))

        result = uplift_at_k_with_ci(y, t, score, 0.1, resamples=1000, seed=seed)
        if result.ci_low <= experiment.ate <= result.ci_high:
            covered += 1

    coverage = covered / draws
    assert coverage >= 0.95, f"bootstrap coverage {coverage:.2%} below 95%"


def test_qini_of_random_is_zero_within_ci() -> None:
    covered = 0
    seeds = 12
    for seed in range(seeds):
        experiment = make_experiment(n=8000, seed=seed, logit_effect=0.0)
        frame = experiment.frame
        t = HILLSTROM.is_treated(frame).to_numpy(dtype=float)
        y = frame["conversion"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed + 500)
        result = qini_with_ci(y, t, rng.normal(size=len(y)), resamples=500, seed=seed)
        if result.ci_low <= 0.0 <= result.ci_high:
            covered += 1
    assert covered >= seeds - 1, f"null Qini covered {covered}/{seeds}"


@given(seed=st.integers(min_value=0, max_value=500))
@settings(max_examples=10, deadline=None)
def test_metrics_are_deterministic_with_seed(seed: int) -> None:
    experiment = make_experiment(n=600, seed=seed)
    frame = experiment.frame
    t = HILLSTROM.is_treated(frame).to_numpy(dtype=float)
    y = frame["conversion"].to_numpy(dtype=float)
    score = experiment.true_uplift

    first = uplift_at_k_with_ci(y, t, score, 0.1, resamples=100, seed=seed)
    second = uplift_at_k_with_ci(y, t, score, 0.1, resamples=100, seed=seed)
    assert first == second

    first_qini = qini_with_ci(y, t, score, resamples=100, seed=seed)
    second_qini = qini_with_ci(y, t, score, resamples=100, seed=seed)
    assert first_qini == second_qini
