from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift.data import loader
from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment


@pytest.fixture
def tmp_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("UPLIFT_DATA_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_experiment(n=300, seed=11).frame


def _write_raw(frame: pd.DataFrame) -> Path:
    path = loader.raw_path("hillstrom")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_manifest_hash_matches_file(tmp_data: Path, frame: pd.DataFrame) -> None:
    path = _write_raw(frame)
    manifest = loader.write_manifest("hillstrom", path, frame)

    assert manifest["sha256"] == loader.sha256_file(path)
    assert manifest["row_count"] == len(frame)
    assert manifest["columns"]["segment"] in {"object", "str"}
    assert loader.verify_manifest("hillstrom")["sha256"] == manifest["sha256"]

    payload = path.read_bytes() + b"\n"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        loader.verify_manifest("hillstrom")


def test_verify_manifest_reports_missing_raw_file(tmp_data: Path, frame: pd.DataFrame) -> None:
    path = _write_raw(frame)
    loader.write_manifest("hillstrom", path, frame)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="raw data missing"):
        loader.verify_manifest("hillstrom")


def test_load_split_indices_roundtrip(tmp_data: Path, frame: pd.DataFrame) -> None:
    path = _write_raw(frame)
    loader.write_manifest("hillstrom", path, frame)
    loader.persist_splits(frame, "hillstrom")

    indices = loader.load_split_indices("hillstrom")
    assert set(indices) == {"train", "validation", "test"}
    all_idx = np.concatenate([indices[k] for k in ("train", "validation", "test")])
    assert sorted(all_idx.tolist()) == list(range(len(frame)))
    assert len(set(indices["train"].tolist())) == len(indices["train"])

    loaded = loader.load_splits("hillstrom")
    assert len(loaded["test"]) == len(indices["test"])
    HILLSTROM.validate(loaded["train"])


def test_split_counts_are_close_to_fractions(frame: pd.DataFrame) -> None:
    labels = loader.row_splits(frame)
    shares = {name: float((labels == name).mean()) for name in ("train", "validation", "test")}
    assert 0.52 <= shares["train"] <= 0.68
    assert 0.14 <= shares["validation"] <= 0.26
    assert 0.14 <= shares["test"] <= 0.26


def test_no_row_leaks_across_splits(frame: pd.DataFrame) -> None:
    splits = loader.split_frame(frame)
    keys = {
        name: {loader.canonical_row_key(row, HILLSTROM.columns) for _, row in part.iterrows()}
        for name, part in splits.items()
    }
    assert keys["train"].isdisjoint(keys["validation"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["validation"].isdisjoint(keys["test"])
    assert len(keys["train"] | keys["validation"] | keys["test"]) == len(
        {loader.canonical_row_key(row, HILLSTROM.columns) for _, row in frame.iterrows()}
    )


def test_split_labels_are_stable_across_calls(frame: pd.DataFrame) -> None:
    first = loader.row_splits(frame)
    second = loader.row_splits(frame.copy())
    assert (first == second).all()


def test_missing_manifest_is_actionable(tmp_data: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        loader.read_manifest("hillstrom")
