from __future__ import annotations

from typing import Any

import pytest

from uplift.decision.memo import REQUIRED_HEADINGS, render_memo, write_memo


def _payload(include_excluded: bool = False) -> dict[str, Any]:
    excluded = (
        {
            "segment": "1) $0 - $100",
            "n": 400,
            "predicted_uplift": -0.004,
            "observed_uplift": -0.006,
            "ci_low": -0.010,
            "ci_high": -0.001,
        }
        if include_excluded
        else None
    )
    lowest = {
        "segment": "1) $0 - $100",
        "n": 400,
        "predicted_uplift": 0.001,
        "observed_uplift": 0.001,
        "ci_low": -0.002,
        "ci_high": 0.004,
    }
    return {
        "dataset": "hillstrom",
        "source": "unit-test",
        "split": "test",
        "seed": 7,
        "n": 2000,
        "metrics": {
            "qini": {"value": 0.08, "ci_low": -0.02, "ci_high": 0.18},
            "uplift_at_10": {"value": 0.012, "ci_low": 0.0004, "ci_high": 0.024},
        },
        "decision": {
            "assumptions": {
                "currency": "₦",
                "margin_per_conversion": 5000.0,
                "treatment_cost": 40.0,
                "break_even_uplift": 0.008,
            },
            "threshold": 0.008,
            "n_targeted": 250,
            "share_targeted": 0.125,
            "predicted_value": 15000.0,
            "measured_incremental_conversions": 3.2,
            "measured_value": 6000.0,
            "ci_low": -2000.0,
            "ci_high": 14000.0,
            "roi": 0.6,
            "resamples": 2000,
            "range": {"low": -5000.0, "high": 20000.0},
            "sensitivity": [
                {
                    "margin_per_conversion": 5000.0,
                    "treatment_cost": 40.0,
                    "break_even_uplift": 0.008,
                    "n_targeted": 250,
                    "measured_value": 6000.0,
                }
            ],
        },
        "segments": {"by": "history_segment", "excluded": excluded, "lowest": lowest},
    }


def test_memo_renders_without_missing_fields() -> None:
    memo = render_memo(_payload())
    assert memo.startswith("# Decision memo — UpLift")
    for heading in REQUIRED_HEADINGS:
        assert heading in memo
    assert "₦6,000" in memo
    assert "₦-2,000" in memo or "-₦2,000" in memo
    assert "Uplift@10%" in memo
    assert "95%" in memo
    assert "250" in memo
    assert "None" not in memo


def test_memo_names_the_excluded_segment() -> None:
    memo = render_memo(_payload(include_excluded=True))
    section = memo.split("## Excluded segment")[1].split("## What this test cannot support")[0]
    assert "1) $0 - $100" in section
    assert "excluded" in section.lower()


def test_memo_states_limits() -> None:
    memo = render_memo(_payload())
    limits = memo.split("## What this test cannot support")[1].split("## Sensitivity")[0]
    assert "binary e-mail treatment" in limits
    assert "long-run effects" in limits


def test_memo_writes_file(tmp_path: object) -> None:
    from pathlib import Path

    path = write_memo(_payload(), Path(str(tmp_path)) / "nested" / "decision-memo.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == render_memo(_payload())


def test_memo_requires_decision_fields() -> None:
    payload = _payload()
    del payload["decision"]["measured_value"]
    with pytest.raises(KeyError):
        render_memo(payload)
