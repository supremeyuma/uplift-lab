from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from uplift.decision.cost import (
    CostMatrix,
    find_excluded_segment,
    format_naira,
    measured_value,
    policy_value,
    segment_uplift,
    sensitivity_grid,
    summarize_policy,
    target_mask,
    value_range,
)


def test_expected_value_matches_hand_computed_example() -> None:
    # margin 5000, cost 40 -> net value = uplift*5000 - 40
    cost = CostMatrix(margin_per_conversion=5000.0, treatment_cost=40.0)
    uplift = np.array([0.001, 0.010, 0.020, -0.010])
    assert cost.net_value(uplift).tolist() == pytest.approx([-35.0, 10.0, 60.0, -90.0])
    assert target_mask(uplift, cost).tolist() == [False, True, True, False]
    assert policy_value(uplift, cost) == pytest.approx(70.0)

    # observed: the two targeted rows are one treated conversion and one control non-conversion
    y = np.array([0.0, 1.0, 0.0, 0.0])
    t = np.array([0.0, 1.0, 0.0, 0.0])
    mask = np.array([False, True, True, False])
    incremental, value = measured_value(y, t, mask, cost)
    assert incremental == pytest.approx(2.0)
    assert value == pytest.approx(2.0 * 5000.0 - 2.0 * 40.0)


@given(
    margin=st.floats(min_value=100.0, max_value=50_000.0),
    other_margin=st.floats(min_value=100.0, max_value=50_000.0),
)
@settings(max_examples=50, deadline=None)
def test_threshold_is_monotone_in_margin(margin: float, other_margin: float) -> None:
    cost = CostMatrix(margin_per_conversion=margin)
    other = CostMatrix(margin_per_conversion=other_margin)
    if margin < other_margin:
        assert cost.break_even_uplift() > other.break_even_uplift()
    elif margin > other_margin:
        assert cost.break_even_uplift() < other.break_even_uplift()
    else:
        assert cost.break_even_uplift() == pytest.approx(other.break_even_uplift())


def test_targeting_threshold_rejects_negative_uplift_segment() -> None:
    rng = np.random.default_rng(3)
    n = 400
    groups = np.array(["weak"] * (n // 2) + ["strong"] * (n // 2))
    t = rng.binomial(1, 0.5, n).astype(float)
    y = np.zeros(n)
    y[groups == "strong"] = t[groups == "strong"]
    y[groups == "weak"] = 1.0 - t[groups == "weak"]  # negative effect

    uplift = np.where(groups == "strong", 0.05, -0.05)
    rows = segment_uplift(y, t, uplift, groups, resamples=200, seed=3)
    excluded = find_excluded_segment(rows)

    assert excluded is not None
    assert excluded["segment"] == "weak"
    assert excluded["ci_high"] < 0.0
    mask = target_mask(uplift, CostMatrix())
    assert not mask[groups == "weak"].any()
    assert mask[groups == "strong"].all()


def test_policy_summary_reports_interval_and_assumptions() -> None:
    rng = np.random.default_rng(4)
    n = 2000
    t = rng.binomial(1, 0.5, n)
    uplift = rng.normal(0.01, 0.01, n)
    y = rng.binomial(1, np.clip(0.1 + t * np.maximum(uplift, 0.0), 0, 1))
    summary = summarize_policy(y, t, uplift, CostMatrix(), resamples=100, seed=4)
    assert summary.ci_low <= summary.measured_value <= summary.ci_high
    assert summary.threshold == pytest.approx(40.0 / 5000.0)
    assert summary.n_targeted > 0
    assert summary.assumptions["currency"] == "₦"


def test_sensitivity_grid_covers_range_and_scales_cost() -> None:
    rng = np.random.default_rng(5)
    n = 1500
    t = rng.binomial(1, 0.5, n)
    uplift = rng.normal(0.02, 0.01, n)
    y = rng.binomial(1, np.clip(0.1 + t * np.maximum(uplift, 0), 0, 1))
    grid = sensitivity_grid(y, t, uplift)
    assert len(grid) == 15
    low, high = value_range(grid)
    assert low <= high
    # raising the margin can only increase the value of the same observed uplift
    base = [row for row in grid if row["treatment_cost"] == 40.0]
    values = [row["measured_value"] for row in base]
    assert values == sorted(values)


def test_format_naira() -> None:
    assert format_naira(1234.6) == "₦1,235"
    assert format_naira(-50.0) == "-₦50"


def test_cost_matrix_validation() -> None:
    with pytest.raises(ValueError, match="margin"):
        CostMatrix(margin_per_conversion=0.0)
    with pytest.raises(ValueError, match="treatment_cost"):
        CostMatrix(treatment_cost=-1.0)
