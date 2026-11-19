"""The report pipeline must run offline on the fixture and be deterministic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from uplift import report


@pytest.fixture(scope="module")
def payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    out = tmp_path_factory.mktemp("report")
    return report.build_report(fixture=True, resamples=60, output_dir=out)


def test_report_writes_metrics_and_figures(payload: dict[str, object], tmp_path: Path) -> None:
    result = report.build_report(fixture=True, resamples=40, output_dir=tmp_path)
    for name in (
        "metrics.json",
        "qini.png",
        "uplift-at-k.png",
        "calibration.png",
        "decision-memo.md",
    ):
        assert (tmp_path / name).exists(), name
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    for key in (
        "qini",
        "uplift_at_5",
        "uplift_at_10",
        "uplift_at_20",
        "calibration_error",
        "incremental_margin",
    ):
        assert key in metrics


def test_metrics_payload_has_intervals(payload: dict[str, object]) -> None:
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    qini = metrics["qini"]
    assert isinstance(qini, dict)
    assert qini["ci_low"] <= qini["value"] <= qini["ci_high"]
    assert qini["resamples"] == 60
    assert "response_model" in qini["baselines"]

    for key in ("uplift_at_5", "uplift_at_10", "uplift_at_20"):
        block = metrics[key]
        assert isinstance(block, dict)
        assert block["ci_low"] <= block["value"] <= block["ci_high"]
        assert block["n_targeted"] > 0
        assert 0.0 < block["treated_share"] < 1.0
        assert "baselines" in block

    calibration = payload["calibration"]
    assert isinstance(calibration, dict)
    assert len(calibration["deciles"]) == 10
    assert calibration["error"] >= 0.0


def test_report_is_deterministic(payload: dict[str, object], tmp_path: Path) -> None:
    again = report.build_report(fixture=True, resamples=60, output_dir=tmp_path)
    assert again["metrics"] == payload["metrics"]
    assert again["model"] == payload["model"]
    assert again["decision"] == payload["decision"]


def test_decision_memo_is_written_with_naira_and_ci(payload: dict[str, object], tmp_path: Path) -> None:
    report.build_report(fixture=True, resamples=20, output_dir=tmp_path)
    memo = (tmp_path / "decision-memo.md").read_text(encoding="utf-8")
    assert "# Decision memo" in memo
    assert "₦" in memo
    assert "95%" in memo
    assert "## Excluded segment" in memo
    decision = payload["decision"]
    assert isinstance(decision, dict)
    assert decision["ci_low"] <= decision["measured_value"] <= decision["ci_high"]
    assert len(decision["sensitivity"]) == 15


def test_metrics_json_matches_returned_payload(payload: dict[str, object], tmp_path: Path) -> None:
    report.build_report(fixture=True, resamples=20, output_dir=tmp_path)
    stored = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert stored["split"] == "test"
    assert stored["dataset"] == "hillstrom"
    assert stored["n"] > 0


def test_build_report_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="unknown split"):
        report.build_report(split="holdout", fixture=True, resamples=10)


def test_report_cli_prints_intervals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert report._main(["--fixture", "--resamples", "20", "--output-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "selected model" in out
    assert "qini" in out
    assert "uplift_at_10" in out
    assert (tmp_path / "metrics.json").exists()


def test_fixture_loader_validates_and_caches(tmp_path: Path) -> None:
    path = tmp_path / "fixture.csv"
    frame = report.load_fixture(path)
    assert len(frame) == report.FIXTURE_ROWS
    assert path.exists()
    columns = set(frame.columns)
    assert {"segment", "conversion", "spend"} <= columns
    assert frame["segment"].nunique() >= 2
    assert np.isfinite(frame["spend"]).all()


def test_split_frames_are_disjoint() -> None:
    frames = report.load_splits(fixture=True)
    total = len(frames.train) + len(frames.validation) + len(frames.test)
    assert total == len(report.load_fixture())
    assert len(frames.train) > len(frames.test)
