"""Render the decision memo from the metrics payload. No new numbers are computed here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uplift.config import ensure_dir
from uplift.decision.cost import format_naira

REQUIRED_HEADINGS = (
    "## Recommendation",
    "## The decision",
    "## Economics",
    "## Uncertainty",
    "## Excluded segment",
    "## What this test cannot support",
    "## Sensitivity",
    "## Reproduce",
)


def _pct(value: float) -> str:
    return f"{value:.2%}"


def render_memo(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    decision = payload["decision"]
    assumptions = decision["assumptions"]
    segment = payload["segments"]

    excluded = segment.get("excluded")
    if excluded:
        excluded_text = (
            f"**{excluded['segment']}** is excluded: measured uplift "
            f"{_pct(excluded['observed_uplift'])} "
            f"[{_pct(excluded['ci_low'])}, {_pct(excluded['ci_high'])}], "
            f"n={excluded['n']:,}. The interval lies entirely below zero, so the optimal "
            "policy never targets it at the assumptions below."
        )
    else:
        lowest = segment["lowest"]
        excluded_text = (
            "No segment has a measured uplift interval entirely below zero. The weakest "
            f"segment is **{lowest['segment']}** at {_pct(lowest['observed_uplift'])} "
            f"[{_pct(lowest['ci_low'])}, {_pct(lowest['ci_high'])}], n={lowest['n']:,}. "
            "It is not excluded by the cost rule, but it is the first place to look when "
            "the campaign is re-tested."
        )

    if decision["n_targeted"] == 0:
        recommendation = (
            "Do not run the targeted campaign at these assumptions: no customer has a "
            "predicted uplift above the break-even threshold."
        )
    elif decision["measured_value"] > 0:
        recommendation = (
            "Target the customers whose predicted uplift clears the break-even threshold."
        )
    elif decision["ci_high"] <= 0:
        recommendation = (
            "Do not run the targeted campaign at these assumptions: the measured value of "
            "the policy is negative and its interval lies below zero."
        )
    else:
        recommendation = (
            "Do not run the campaign on this evidence. The policy's measured value is "
            "negative, although its interval includes zero; the predicted uplift is not "
            "large enough to cover the assumed treatment cost. Raise the threshold or "
            "lower the cost before committing spend."
        )

    uplift10 = metrics["uplift_at_10"]
    qini = metrics["qini"]
    margin = format_naira(assumptions["margin_per_conversion"])
    cost = format_naira(assumptions["treatment_cost"])
    value = format_naira(decision["measured_value"])
    low = format_naira(decision["ci_low"])
    high = format_naira(decision["ci_high"])
    grid_low = format_naira(decision["range"]["low"])
    grid_high = format_naira(decision["range"]["high"])

    lines = [
        "# Decision memo — UpLift",
        "",
        f"Dataset: `{payload['dataset']}` ({payload['source']}), evaluation split "
        f"`{payload['split']}`, n={payload['n']:,}, seed {payload['seed']}.",
        "",
        "## Recommendation",
        "",
        recommendation,
        f"The policy targets **{decision['n_targeted']:,} customers "
        f"({_pct(decision['share_targeted'])} of the split)** with a break-even uplift of "
        f"{_pct(decision['threshold'])}. Modelled value at these assumptions is "
        f"{format_naira(decision['predicted_value'])}; the value measured in the experiment "
        f"is {value} with a 95% bootstrap interval of [{low}, {high}].",
        "",
        "## The decision",
        "",
        f"- Treatment cost: {cost} per customer.",
        f"- Margin per conversion: {margin}.",
        f"- Break-even uplift: {cost} / {margin} = {_pct(decision['threshold'])}.",
        f"- Customers above the threshold: {decision['n_targeted']:,}.",
        f"- Return on treatment spend: {decision['roi']:.2f}x.",
        "",
        "## Economics",
        "",
        f"Measured incremental conversions in the targeted group: "
        f"{decision['measured_incremental_conversions']:.1f}. At {margin} per conversion "
        f"minus {cost} per treated customer, the measured incremental margin is "
        f"**{value}** (95% CI {low} to {high}).",
        "",
        "## Uncertainty",
        "",
        f"- Qini coefficient: {qini['value']:.4f} "
        f"[{qini['ci_low']:.4f}, {qini['ci_high']:.4f}].",
        f"- Uplift@10%: {_pct(uplift10['value'])} "
        f"[{_pct(uplift10['ci_low'])}, {_pct(uplift10['ci_high'])}].",
        f"- Incremental margin interval: {low} to {high} (bootstrap, "
        f"{decision['resamples']} resamples, seed {payload['seed']}).",
        "- The interval covers sampling variation in the evaluation split, not model risk "
        "or the possibility that the next campaign behaves differently.",
        "",
        "## Excluded segment",
        "",
        excluded_text,
        "",
        "## What this test cannot support",
        "",
        "- It cannot rank channels or creatives: Hillstrom has one binary e-mail treatment.",
        "- It cannot claim long-run effects: spend and conversion are measured in a short "
        "window and repeat purchases are not observed.",
        "- It cannot separate the e-mail effect from the offer it contained; the treated arms "
        "differ in message, not in incentive.",
        "- It cannot justify a margin outside the sensitivity grid below.",
        "",
        "## Sensitivity",
        "",
        "| Margin | Treatment cost | Break-even uplift | Targeted | Measured value |",
        "|---|---|---|---|---|",
    ]
    for row in decision["sensitivity"]:
        lines.append(
            f"| {format_naira(row['margin_per_conversion'])} | "
            f"{format_naira(row['treatment_cost'])} | "
            f"{_pct(row['break_even_uplift'])} | {row['n_targeted']:,} | "
            f"{format_naira(row['measured_value'])} |"
        )
    lines += [
        "",
        f"Across the grid the measured value ranges from {grid_low} to {grid_high}. A "
        "decision that only survives one cell of this table is not a decision.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "make setup && make report",
        "python scripts/check_readme_numbers.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_memo(payload: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    path.write_text(render_memo(payload), encoding="utf-8")
    return path
