"""Property: MDE shrinks monotonically as either arm grows."""

from hypothesis import given, settings
from hypothesis import strategies as st

from uplift.design.power import mde


@given(
    baseline=st.floats(min_value=0.01, max_value=0.40),
    n=st.integers(min_value=200, max_value=5_000),
)
@settings(max_examples=40, deadline=None)
def test_power_mde_shrinks_with_sample_size(baseline: float, n: int) -> None:
    small = mde(baseline, n, n)
    large = mde(baseline, 4 * n, 4 * n)
    assert 0.0 < large < small


@given(
    baseline=st.floats(min_value=0.01, max_value=0.30),
    n=st.integers(min_value=200, max_value=3_000),
    ratio=st.floats(min_value=0.25, max_value=4.0),
)
@settings(max_examples=30, deadline=None)
def test_mde_shrinks_with_the_smaller_arm(baseline: float, n: int, ratio: float) -> None:
    n_treatment = max(1, int(n * ratio))
    balanced = mde(baseline, n, n_treatment)
    doubled = mde(baseline, 2 * n, 2 * n_treatment)
    assert doubled <= balanced
