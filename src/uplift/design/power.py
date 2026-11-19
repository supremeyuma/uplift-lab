"""Experiment design maths: minimum detectable effect, required n, power.

All routines use the two-sided normal approximation for a difference in
proportions. The alternative standard error is unpooled, the null standard
error is pooled; that is the convention used by the sample-size calculators
most analysts check against.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.stats import norm

from uplift.config import ALPHA, SPLIT_FRACTIONS


@dataclass(frozen=True)
class PowerReport:
    baseline_rate: float
    treatment_rate: float
    n_control: int
    n_treatment: int
    alpha: float
    mde: float
    realised_power: float


def _check_rate(rate: float, name: str) -> None:
    if not 0.0 < rate < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {rate!r}")


def two_proportion_power(
    p_control: float,
    p_treatment: float,
    n_control: int,
    n_treatment: int,
    alpha: float = ALPHA,
) -> float:
    """Power of a two-sided two-proportion z-test at the given effect."""
    _check_rate(p_control, "p_control")
    _check_rate(p_treatment, "p_treatment")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("sample sizes must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    delta = abs(p_treatment - p_control)
    pooled = (p_control * n_control + p_treatment * n_treatment) / (n_control + n_treatment)
    se_null = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_control + 1.0 / n_treatment))
    se_alt = math.sqrt(
        p_control * (1.0 - p_control) / n_control
        + p_treatment * (1.0 - p_treatment) / n_treatment
    )
    if se_alt == 0.0:
        return 1.0
    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    return float(norm.cdf((delta - z_alpha * se_null) / se_alt))


def mde(
    baseline_rate: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = ALPHA,
    power: float = 0.80,
    upper: float = 0.99,
) -> float:
    """Smallest absolute effect detectable with the requested power."""
    _check_rate(baseline_rate, "baseline_rate")
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("sample sizes must be positive")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    cap = min(upper, 1.0 - baseline_rate - 1e-6)

    def gap(delta: float) -> float:
        return two_proportion_power(baseline_rate, baseline_rate + delta, n_control, n_treatment, alpha) - power

    if gap(cap) < 0.0:
        return cap
    return float(brentq(gap, 1e-9, cap, xtol=1e-12, rtol=1e-12))


def required_n(
    baseline_rate: float,
    effect: float,
    *,
    alpha: float = ALPHA,
    power: float = 0.80,
    ratio: float = 1.0,
) -> tuple[int, int]:
    """Per-arm sample sizes needed to detect ``effect`` (absolute)."""
    _check_rate(baseline_rate, "baseline_rate")
    if effect <= 0.0:
        raise ValueError("effect must be positive")
    _check_rate(baseline_rate + effect, "baseline_rate + effect")
    if ratio <= 0.0:
        raise ValueError("ratio must be positive")

    def gap(n_control: float) -> float:
        n_treatment = max(1, round(n_control * ratio))
        return (
            two_proportion_power(
                baseline_rate, baseline_rate + effect, round(n_control), n_treatment, alpha
            )
            - power
        )

    n_control = brentq(gap, 1.0, 5_000_000.0, xtol=1.0, rtol=1e-6)
    n_c = math.ceil(n_control)
    n_t = math.ceil(n_c * ratio)
    return n_c, n_t


def analyse(
    baseline_rate: float,
    treatment_rate: float,
    n_control: int,
    n_treatment: int,
    *,
    alpha: float = ALPHA,
    power: float = 0.80,
) -> PowerReport:
    return PowerReport(
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
        n_control=n_control,
        n_treatment=n_treatment,
        alpha=alpha,
        mde=mde(baseline_rate, n_control, n_treatment, alpha=alpha, power=power),
        realised_power=two_proportion_power(
            baseline_rate, treatment_rate, n_control, n_treatment, alpha
        ),
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MDE and realised power for the Hillstrom experiment")
    parser.add_argument("--data", default="hillstrom", choices=["hillstrom"])
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--power", type=float, default=0.80)
    args = parser.parse_args(argv)

    from uplift.data.loader import load_frame
    from uplift.data.schema import HILLSTROM

    frame = load_frame(args.data)
    treated = HILLSTROM.is_treated(frame)
    control = frame.loc[~treated, "conversion"]
    treatment = frame.loc[treated, "conversion"]

    report = analyse(
        float(control.mean()),
        float(treatment.mean()),
        len(control),
        len(treatment),
        alpha=args.alpha,
        power=args.power,
    )
    test_share = SPLIT_FRACTIONS["test"]
    print(f"dataset            : {args.data} (all rows)")
    print(f"n control          : {report.n_control:,}")
    print(f"n treated          : {report.n_treatment:,}")
    print(f"baseline conversion: {report.baseline_rate:.4%}")
    print(f"treated conversion : {report.treatment_rate:.4%}")
    print(f"observed effect    : {report.treatment_rate - report.baseline_rate:+.4%}")
    print(f"MDE @ {args.power:.0%} power  : {report.mde:.4%} (absolute)")
    print(f"realised power     : {report.realised_power:.4f}")
    print(f"test-split share   : {test_share:.0%} of rows is never used before M6")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
