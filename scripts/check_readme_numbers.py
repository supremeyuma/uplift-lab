"""Fail when the README results table disagrees with reports/metrics.json.

The README table is the only place numbers are copied by hand, so a machine
check keeps the copy honest. Formatting is fixed here; a mismatch is a gate
failure, not a warning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_README = ROOT / "README.md"
DEFAULT_METRICS = ROOT / "reports" / "metrics.json"


def load_metrics(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def naira(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}₦{abs(value):,.0f}"


def expected_rows(payload: dict[str, Any]) -> dict[str, list[str]]:
    metrics = payload["metrics"]
    uplift = metrics["uplift_at_10"]
    qini = metrics["qini"]
    margin = metrics["incremental_margin"]
    return {
        "Uplift@10%": [
            "0.00%",
            f"{uplift['value']:.2%}",
            f"95% CI {uplift['ci_low']:.2%} \u2013 {uplift['ci_high']:.2%}",
        ],
        "Qini coefficient": [
            f"{qini['baselines']['response_model']:.4f}",
            f"{qini['value']:.4f}",
            f"95% CI {qini['ci_low']:.4f} \u2013 {qini['ci_high']:.4f}",
        ],
        "Incremental margin": [
            naira(margin["baselines"]["untargeted_campaign"]),
            naira(margin["value"]),
            f"95% CI {naira(margin['ci_low'])} \u2013 {naira(margin['ci_high'])}",
        ],
    }


def parse_table(readme: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in readme.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or "---" in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        rows[cells[0]] = cells[1:]
    return rows


def check(readme: str, payload: dict[str, Any]) -> list[str]:
    expected = expected_rows(payload)
    actual = parse_table(readme)
    errors: list[str] = []
    for label, cells in expected.items():
        row = actual.get(label)
        if row is None:
            errors.append(f"README table is missing a {label!r} row")
            continue
        for index, value in enumerate(cells):
            if index >= len(row):
                errors.append(f"{label}: README row is too short")
                break
            if row[index] != value:
                errors.append(
                    f"{label}: expected {value!r} in column {index + 1}, found {row[index]!r}"
                )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args(argv)

    if not args.metrics.exists():
        print(f"missing metrics file: {args.metrics}; run `make report`", file=sys.stderr)
        return 2
    errors = check(args.readme.read_text(encoding="utf-8"), load_metrics(args.metrics))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("README numbers match reports/metrics.json")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
