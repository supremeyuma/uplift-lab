from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from scripts import check_metrics_schema, check_readme_numbers


def _metrics() -> dict[str, Any]:
    return {
        "dataset": "hillstrom",
        "split": "test",
        "seed": 20240501,
        "n": 12891,
        "model": {"selected": "tlearner-lightgbm"},
        "metrics": {
            "qini": {
                "value": 0.0838,
                "ci_low": -0.0263,
                "ci_high": 0.2034,
                "baselines": {"response_model": -0.0496},
            },
            "uplift_at_5": {"value": 0.0101, "ci_low": -0.0071, "ci_high": 0.0275},
            "uplift_at_10": {"value": 0.0121, "ci_low": 0.0004, "ci_high": 0.0242},
            "uplift_at_20": {"value": 0.0049, "ci_low": -0.0034, "ci_high": 0.0136},
            "calibration_error": 0.0097,
            "incremental_margin": {
                "value": -24862.0,
                "ci_low": -169969.0,
                "ci_high": 116933.0,
                "baselines": {"untargeted_campaign": -308889.0},
            },
        },
        "decision": {
            "assumptions": {
                "currency": "₦",
                "margin_per_conversion": 5000.0,
                "treatment_cost": 40.0,
            },
            "ci_low": -169969.0,
            "ci_high": 116933.0,
            "sensitivity": [{"margin_per_conversion": 5000.0, "treatment_cost": 40.0}],
        },
    }


def _readme(payload: dict[str, Any]) -> str:
    rows = check_readme_numbers.expected_rows(payload)
    lines = ["# UpLift", "", "| Metric | Baseline | UpLift | 95% interval |", "|---|---|---|---|"]
    for label, cells in rows.items():
        lines.append(f"| {label} | {' | '.join(cells)} |")
    return "\n".join(lines) + "\n"


def test_readme_gate_accepts_matching_numbers() -> None:
    payload = _metrics()
    assert check_readme_numbers.check(_readme(payload), payload) == []


def test_readme_numbers_gate_catches_drift(tmp_path: Path) -> None:
    payload = _metrics()
    good = _readme(payload)
    bad = good.replace("1.21%", "1.31%", 1)
    assert bad != good

    errors = check_readme_numbers.check(bad, payload)
    assert any("Uplift@10%" in error for error in errors)

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    good_path = tmp_path / "README.md"
    good_path.write_text(good, encoding="utf-8")
    bad_path = tmp_path / "README-bad.md"
    bad_path.write_text(bad, encoding="utf-8")

    assert check_readme_numbers.main(["--readme", str(good_path), "--metrics", str(metrics_path)]) == 0
    assert check_readme_numbers.main(["--readme", str(bad_path), "--metrics", str(metrics_path)]) == 1


def test_readme_gate_reports_missing_row() -> None:
    payload = _metrics()
    readme = _readme(payload).replace("| Qini coefficient |", "| Qini |", 1)
    errors = check_readme_numbers.check(readme, payload)
    assert errors
    assert "missing a 'Qini coefficient' row" in errors[0]


def test_readme_gate_requires_metrics_file(tmp_path: Path) -> None:
    assert check_readme_numbers.main(["--metrics", str(tmp_path / "absent.json")]) == 2


def test_metrics_schema_accepts_valid_payload() -> None:
    assert check_metrics_schema.validate(_metrics()) == []


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda p: p["metrics"].pop("qini"), "missing metric: qini"),
        (lambda p: p["metrics"]["qini"].update(ci_low=0.5), "ci_low exceeds ci_high"),
        (
            lambda p: p["metrics"]["uplift_at_10"].update(value=0.5),
            "uplift_at_10: value outside its interval",
        ),
        (lambda p: p["metrics"].update(calibration_error=-0.1), "calibration_error"),
        (lambda p: p["model"].update(selected="magic"), "unexpected model"),
        (lambda p: p["decision"].pop("sensitivity"), "sensitivity grid is empty"),
        (
            lambda p: p["decision"]["assumptions"].update(margin_per_conversion=0),
            "margin must be positive",
        ),
        (lambda p: p.pop("split"), "missing top-level key: split"),
    ],
)
def test_metrics_schema_catches_drift(mutate: Any, needle: str) -> None:
    payload = _metrics()
    mutate(payload)
    errors = check_metrics_schema.validate(payload)
    assert any(needle in error for error in errors), errors
