"""Assert that reports/metrics.json has the shape and ranges the README relies on."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "reports" / "metrics.json"
REQUIRED_METRICS = ("qini", "uplift_at_5", "uplift_at_10", "uplift_at_20", "incremental_margin")


def load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def validate(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("dataset", "split", "seed", "n", "model", "metrics", "decision"):
        if key not in payload:
            errors.append(f"missing top-level key: {key}")
    if errors:
        return errors

    if payload["split"] not in {"train", "validation", "test"}:
        errors.append(f"unexpected split: {payload['split']!r}")
    if not isinstance(payload["n"], int) or payload["n"] <= 0:
        errors.append("n must be a positive integer")
    if payload["model"].get("selected") not in {
        "tlearner-logistic",
        "tlearner-lightgbm",
    }:
        errors.append(f"unexpected model: {payload['model'].get('selected')!r}")

    metrics = payload["metrics"]
    for key in REQUIRED_METRICS:
        if key not in metrics:
            errors.append(f"missing metric: {key}")
            continue
        block = metrics[key]
        for field in ("value", "ci_low", "ci_high"):
            if not isinstance(block.get(field), (int, float)):
                errors.append(f"{key}.{field} must be numeric")
        if isinstance(block.get("ci_low"), (int, float)) and isinstance(
            block.get("ci_high"), (int, float)
        ):
            if block["ci_low"] > block["ci_high"]:
                errors.append(f"{key}: ci_low exceeds ci_high")
            if not block["ci_low"] <= block["value"] <= block["ci_high"]:
                errors.append(f"{key}: value outside its interval")

    for key in ("qini", "uplift_at_5", "uplift_at_10", "uplift_at_20"):
        if key in metrics and not -1.0 <= float(metrics[key]["value"]) <= 1.0:
            errors.append(f"{key}: value outside [-1, 1]")

    calibration = metrics.get("calibration_error")
    if not isinstance(calibration, (int, float)) or calibration < 0:
        errors.append("calibration_error must be non-negative")

    decision = payload["decision"]
    assumptions = decision.get("assumptions", {})
    if assumptions.get("margin_per_conversion", 0) <= 0:
        errors.append("decision margin must be positive")
    if assumptions.get("treatment_cost", -1) < 0:
        errors.append("decision treatment cost must be non-negative")
    if not decision.get("sensitivity"):
        errors.append("decision sensitivity grid is empty")
    if decision.get("ci_low", 1.0) > decision.get("ci_high", -1.0):
        errors.append("decision interval is inverted")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args(argv)

    if not args.metrics.exists():
        print(f"missing metrics file: {args.metrics}; run `make report`", file=sys.stderr)
        return 2
    errors = validate(load(args.metrics))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"metrics schema OK: {args.metrics}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
