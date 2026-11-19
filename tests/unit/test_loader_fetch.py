"""Offline tests for the fetch path: urllib is replaced by an in-memory response."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from uplift.data import loader
from uplift.data.synthetic import make_experiment


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class FetchHarness:
    payload: bytes
    calls: list[str] = field(default_factory=list)

    def urlopen(self, url: str, timeout: int = 0) -> FakeResponse:
        self.calls.append(url)
        return FakeResponse(self.payload)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FetchHarness:
    frame = make_experiment(n=120, seed=21).frame
    harness = FetchHarness(payload=frame.to_csv(index=False).encode("utf-8"))
    monkeypatch.setenv("UPLIFT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("urllib.request.urlopen", harness.urlopen)
    return harness


def test_fetch_downloads_verifies_and_is_idempotent(harness: FetchHarness) -> None:
    payload = loader.fetch("hillstrom")
    assert len(harness.calls) == 1
    assert payload["row_count"] == 120
    assert loader.raw_path().exists()
    assert loader.manifest_path().exists()

    again = loader.fetch("hillstrom")
    assert len(harness.calls) == 1, "cached data must not be re-downloaded"
    assert again["sha256"] == payload["sha256"]

    forced = loader.fetch("hillstrom", force=True)
    assert len(harness.calls) == 2
    assert forced["sha256"] == payload["sha256"]


def test_fetch_rejects_and_replaces_corrupt_cache(harness: FetchHarness) -> None:
    first = loader.fetch("hillstrom")
    loader.raw_path().write_text("recency\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing treatment column"):
        loader.fetch("hillstrom")

    frame = make_experiment(n=90, seed=99).frame
    loader.raw_path().write_text(frame.to_csv(index=False), encoding="utf-8")
    second = loader.fetch("hillstrom")
    assert second["sha256"] != first["sha256"]


def test_main_fetch_prints_summary(
    harness: FetchHarness, capsys: pytest.CaptureFixture[str]
) -> None:
    assert loader.main(["--fetch"]) == 0
    out = capsys.readouterr().out
    assert "sha256" in out
    assert "split counts" in out
    assert loader.splits_path().exists()


def test_main_without_fetch_verifies_cached_data(harness: FetchHarness) -> None:
    loader.fetch("hillstrom")
    assert loader.main([]) == 0


def test_download_reports_http_failure(
    harness: FetchHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(url: str, timeout: int = 0) -> FakeResponse:
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(OSError, match="connection refused"):
        loader.download(loader.HILLSTROM_URL, loader.raw_path())


def test_manifest_shape_is_json_serialisable(harness: FetchHarness) -> None:
    loader.fetch("hillstrom")
    frame = pd.read_csv(loader.raw_path())
    manifest = loader.read_manifest("hillstrom")
    assert manifest["bytes"] == len(harness.payload)
    assert set(manifest["columns"]) == set(frame.columns)


def test_ensure_dataset_fetches_once_and_regenerates_missing_splits(
    harness: FetchHarness,
) -> None:
    loader.ensure_dataset()
    assert loader.raw_path().exists()
    assert loader.manifest_path().exists()
    first = loader.load_split_indices()
    assert len(harness.calls) == 1

    loader.splits_path().unlink()
    loader.ensure_dataset()
    second = loader.load_split_indices()
    assert len(harness.calls) == 1, "cached raw data must not be re-downloaded"
    for name in first:
        assert (first[name] == second[name]).all()

    loader.manifest_path().unlink()
    loader.ensure_dataset()
    assert loader.manifest_path().exists()
    assert len(harness.calls) == 1
