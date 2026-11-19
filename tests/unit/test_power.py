import math

import pytest
from scipy.stats import norm

from uplift.design.power import analyse, mde, required_n, two_proportion_power


def test_realised_power_matches_reference_for_known_case() -> None:
    """Reference value worked by hand from the normal approximation.

    p0=0.10, p1=0.11, n=5000 per arm, alpha=0.05:
      se_alt  = sqrt(0.1*0.9/5000 + 0.11*0.89/5000) = 0.0061300
      se_null = sqrt(0.105*0.895*(2/5000))           = 0.0061311
      power   = Phi((0.01 - 1.959964*0.0061311)/0.0061300) = 0.37109
    """
    power = two_proportion_power(0.10, 0.11, 5000, 5000)
    assert power == pytest.approx(0.37109, abs=1e-4)

    # Independently recompute the same formula, to pin the convention.
    se_alt = math.sqrt(0.1 * 0.9 / 5000 + 0.11 * 0.89 / 5000)
    se_null = math.sqrt(0.105 * 0.895 * (2 / 5000))
    expected = norm.cdf((0.01 - norm.ppf(0.975) * se_null) / se_alt)
    assert power == pytest.approx(float(expected), abs=1e-12)


def test_power_increases_with_effect_and_sample_size() -> None:
    small = two_proportion_power(0.10, 0.11, 5000, 5000)
    big_effect = two_proportion_power(0.10, 0.13, 5000, 5000)
    big_n = two_proportion_power(0.10, 0.11, 20000, 20000)
    assert small < big_effect
    assert small < big_n


def test_mde_is_reachable_at_target_power() -> None:
    baseline = 0.05
    effect = mde(baseline, 100_000, 100_000, power=0.80)
    achieved = two_proportion_power(baseline, baseline + effect, 100_000, 100_000)
    assert achieved == pytest.approx(0.80, abs=1e-6)


def test_required_n_achieves_target_power() -> None:
    n_c, n_t = required_n(0.05, 0.005, power=0.80)
    achieved = two_proportion_power(0.05, 0.055, n_c, n_t)
    assert achieved >= 0.80
    assert achieved < 0.85


def test_required_n_grows_for_smaller_effects() -> None:
    coarse = required_n(0.05, 0.01)
    fine = required_n(0.05, 0.002)
    assert fine[0] > coarse[0]


def test_analyse_reports_observed_effect_power() -> None:
    report = analyse(0.09, 0.10, 20_000, 20_000)
    assert report.mde > 0
    assert 0.0 < report.realised_power < 1.0
    assert report.realised_power == pytest.approx(
        two_proportion_power(0.09, 0.10, 20_000, 20_000), abs=1e-12
    )


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((0.0, 0.01, 100, 100), "p_control"),
        ((0.1, 1.0, 100, 100), "p_treatment"),
        ((0.1, 0.2, 0, 100), "positive"),
    ],
)
def test_power_rejects_invalid_inputs(args: tuple[float, float, int, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        two_proportion_power(*args)


def test_power_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        two_proportion_power(0.1, 0.2, 100, 100, alpha=1.5)


def test_mde_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="power"):
        mde(0.1, 100, 100, power=1.5)
    with pytest.raises(ValueError, match="positive"):
        mde(0.1, 0, 100)


def test_required_n_rejects_non_positive_effect() -> None:
    with pytest.raises(ValueError, match="effect"):
        required_n(0.1, 0.0)
