"""Cost-based targeting: expected incremental value per customer and threshold.

All monetary values are explicit assumptions (naira), never measured from the
dataset. The sensitivity grid is the answer to "what if the assumption is
wrong", not a single confident number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from uplift.config import ALPHA, BOOTSTRAP_RESAMPLES, SEED
from uplift.eval.bootstrap import bootstrap_indices, percentile_interval

DEFAULT_MARGIN_NAIRA = 5000.0
DEFAULT_TREATMENT_COST_NAIRA = 40.0
CURRENCY = "₦"


@dataclass(frozen=True)
class CostMatrix:
    margin_per_conversion: float = DEFAULT_MARGIN_NAIRA
    treatment_cost: float = DEFAULT_TREATMENT_COST_NAIRA
    currency: str = CURRENCY

    def __post_init__(self) -> None:
        if self.margin_per_conversion <= 0:
            raise ValueError("margin_per_conversion must be positive")
        if self.treatment_cost < 0:
            raise ValueError("treatment_cost must be non-negative")

    def break_even_uplift(self) -> float:
        """Smallest uplift that pays for treating one customer."""
        return self.treatment_cost / self.margin_per_conversion

    def net_value(self, uplift: np.ndarray) -> np.ndarray:
        return np.asarray(uplift, dtype=float) * self.margin_per_conversion - self.treatment_cost

    def as_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "currency_code": "NGN",
            "margin_per_conversion": self.margin_per_conversion,
            "treatment_cost": self.treatment_cost,
            "break_even_uplift": self.break_even_uplift(),
        }


@dataclass(frozen=True)
class PolicySummary:
    threshold: float
    n_targeted: int
    share_targeted: float
    predicted_value: float
    measured_incremental_conversions: float
    measured_value: float
    ci_low: float
    ci_high: float
    roi: float
    assumptions: dict[str, Any]


def target_mask(uplift: np.ndarray, cost: CostMatrix) -> np.ndarray:
    return cost.net_value(uplift) > 0.0


def policy_value(uplift: np.ndarray, cost: CostMatrix) -> float:
    return float(np.sum(np.maximum(cost.net_value(uplift), 0.0)))


def measured_value(
    y: np.ndarray, t: np.ndarray, mask: np.ndarray, cost: CostMatrix
) -> tuple[float, float]:
    """Observed value of the policy: measured uplift x margin x n - cost x n."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n_targeted = int(mask.sum())
    if n_targeted == 0:
        return 0.0, 0.0
    n_treated = float(t[mask].sum())
    n_control = n_targeted - n_treated
    if n_treated == 0 or n_control == 0:
        raise ValueError("targeted group contains a single arm")
    observed_uplift = float(y[mask & (t == 1)].mean() - y[mask & (t == 0)].mean())
    incremental = observed_uplift * n_targeted
    value = incremental * cost.margin_per_conversion - cost.treatment_cost * n_targeted
    return incremental, float(value)


def summarize_policy(
    y: np.ndarray,
    t: np.ndarray,
    uplift: np.ndarray,
    cost: CostMatrix = CostMatrix(),
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> PolicySummary:
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    uplift = np.asarray(uplift, dtype=float)
    mask = target_mask(uplift, cost)
    n_targeted = int(mask.sum())
    if n_targeted == 0:
        incremental, value = 0.0, 0.0
        low = high = 0.0
    else:
        incremental, value = measured_value(y, t, mask, cost)
        samples: list[float] = []
        for idx in bootstrap_indices(len(y), resamples, seed):
            y_b, t_b, mask_b = y[idx], t[idx], mask[idx]
            if mask_b.sum() == 0:
                samples.append(0.0)
                continue
            n_t = t_b[mask_b].sum()
            if n_t == 0 or n_t == mask_b.sum():
                continue
            observed = y_b[mask_b & (t_b == 1)].mean() - y_b[mask_b & (t_b == 0)].mean()
            samples.append(
                float(
                    observed * mask_b.sum() * cost.margin_per_conversion
                    - cost.treatment_cost * mask_b.sum()
                )
            )
        low, high = percentile_interval(samples, alpha)

    spend = cost.treatment_cost * n_targeted
    return PolicySummary(
        threshold=cost.break_even_uplift(),
        n_targeted=n_targeted,
        share_targeted=float(n_targeted / len(y)),
        predicted_value=policy_value(uplift, cost),
        measured_incremental_conversions=incremental,
        measured_value=value,
        ci_low=low,
        ci_high=high,
        roi=float(value / spend) if spend > 0 else 0.0,
        assumptions=cost.as_dict(),
    )


def sensitivity_grid(
    y: np.ndarray,
    t: np.ndarray,
    uplift: np.ndarray,
    *,
    margins: tuple[float, ...] | None = None,
    costs: tuple[float, ...] | None = None,
) -> list[dict[str, Any]]:
    base = CostMatrix()
    margins = margins or tuple(base.margin_per_conversion * f for f in (0.5, 0.75, 1.0, 1.25, 1.5))
    costs = costs or tuple(base.treatment_cost * f for f in (0.5, 1.0, 2.0))
    rows: list[dict[str, Any]] = []
    for margin in margins:
        for treatment_cost in costs:
            matrix = CostMatrix(margin_per_conversion=margin, treatment_cost=treatment_cost)
            mask = target_mask(uplift, matrix)
            if mask.sum() == 0:
                incremental, value = 0.0, 0.0
            else:
                incremental, value = measured_value(y, t, mask, matrix)
            rows.append(
                {
                    "margin_per_conversion": float(margin),
                    "treatment_cost": float(treatment_cost),
                    "break_even_uplift": matrix.break_even_uplift(),
                    "n_targeted": int(mask.sum()),
                    "predicted_value": policy_value(uplift, matrix),
                    "measured_value": float(value),
                    "measured_incremental_conversions": float(incremental),
                }
            )
    return rows


def value_range(grid: list[dict[str, Any]]) -> tuple[float, float]:
    values = [float(row["measured_value"]) for row in grid]
    return (min(values), max(values))


def segment_uplift(
    y: np.ndarray,
    t: np.ndarray,
    uplift: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int = 300,
    seed: int = SEED,
    alpha: float = ALPHA,
) -> list[dict[str, Any]]:
    """Measured and predicted uplift per segment, with an interval on the measured value."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    uplift = np.asarray(uplift, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    for group in sorted(np.unique(groups).tolist()):
        mask = groups == group
        if mask.sum() < 50:
            continue
        n_t = t[mask].sum()
        if n_t == 0 or n_t == mask.sum():
            continue
        observed = float(y[mask & (t == 1)].mean() - y[mask & (t == 0)].mean())
        samples: list[float] = []
        for _ in range(resamples):
            idx = rng.integers(0, mask.sum(), size=int(mask.sum()))
            t_b = t[mask][idx]
            y_b = y[mask][idx]
            if t_b.sum() == 0 or t_b.sum() == len(t_b):
                continue
            samples.append(float(y_b[t_b == 1].mean() - y_b[t_b == 0].mean()))
        low, high = percentile_interval(samples, alpha)
        rows.append(
            {
                "segment": str(group),
                "n": int(mask.sum()),
                "predicted_uplift": float(uplift[mask].mean()),
                "observed_uplift": observed,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def find_excluded_segment(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The worst segment whose measured uplift interval is entirely below zero."""
    negative = [row for row in rows if row["ci_high"] < 0.0]
    if not negative:
        return None
    return min(negative, key=lambda row: row["observed_uplift"])


def format_naira(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}₦{abs(value):,.0f}"
