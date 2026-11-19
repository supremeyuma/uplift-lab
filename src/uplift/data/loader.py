"""Fetch, verify, cache and split the Hillstrom e-mail experiment.

The raw CSV is never committed. A manifest with its sha256, shape and column
dtypes is committed instead, and every load verifies the file against it.

Splitting is a deterministic function of each row's stable content hash, so it
does not depend on row order, file timestamps or the machine. Test rows are
therefore never seen during training or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from uplift.config import SPLIT_FRACTIONS, data_dir, ensure_dir, manifest_dir, raw_dir, split_dir
from uplift.data.schema import HILLSTROM, DatasetSchema, get_schema

HILLSTROM_URL: Final = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
SPLIT_VERSION: Final = "v1"
CHUNK: Final = 1 << 20


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(name: str = "hillstrom") -> Path:
    return raw_dir() / f"{name}.csv"


def manifest_path(name: str = "hillstrom") -> Path:
    return manifest_dir() / f"{name}.json"


def _manifest_payload(csv_path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": "hillstrom",
        "source_url": HILLSTROM_URL,
        "path": str(csv_path.relative_to(data_dir().parent)).replace("\\", "/"),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sha256": sha256_file(csv_path),
        "bytes": csv_path.stat().st_size,
        "row_count": len(frame),
        "columns": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "outcomes": list(HILLSTROM.outcomes),
        "treatment": HILLSTROM.treatment,
        "control_label": HILLSTROM.control_label,
    }


def write_manifest(name: str, csv_path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    payload = _manifest_payload(csv_path, frame)
    path = manifest_path(name)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def download(url: str, dest: Path) -> Path:
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)
    return dest


def fetch(name: str = "hillstrom", *, force: bool = False) -> dict[str, Any]:
    """Download the dataset once and write/normalise its manifest."""
    schema = get_schema(name)
    dest = raw_path(name)
    if force or not dest.exists():
        url = HILLSTROM_URL if name == "hillstrom" else ""
        if not url:
            raise ValueError(f"no download URL for {name!r}")
        download(url, dest)

    frame = pd.read_csv(dest)
    schema.validate(frame)
    payload = _manifest_payload(dest, frame)

    existing = manifest_path(name)
    if existing.exists() and not force:
        old: dict[str, Any] = json.loads(existing.read_text(encoding="utf-8"))
        if old.get("sha256") == payload["sha256"]:
            return old

    ensure_dir(existing.parent)
    existing.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def read_manifest(name: str = "hillstrom") -> dict[str, Any]:
    path = manifest_path(name)
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}; run `make data`")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def verify_manifest(name: str = "hillstrom") -> dict[str, Any]:
    """Raise if the cached CSV no longer matches its committed manifest."""
    payload = read_manifest(name)
    path = raw_path(name)
    if not path.exists():
        raise FileNotFoundError(f"raw data missing: {path}; run `make data`")
    actual = sha256_file(path)
    expected = str(payload["sha256"])
    if actual != expected:
        raise ValueError(
            f"{name}: sha256 mismatch (expected {expected}, got {actual}); "
            "delete data/raw and run `make data`"
        )
    return payload


def load_frame(name: str = "hillstrom") -> pd.DataFrame:
    verify_manifest(name)
    frame = pd.read_csv(raw_path(name))
    get_schema(name).validate(frame)
    return frame


def canonical_row_key(row: pd.Series, columns: tuple[str, ...]) -> str:
    parts: list[str] = []
    for col in columns:
        value = row[col]
        if isinstance(value, (float, np.floating)):
            parts.append(f"{float(value):.6f}")
        else:
            parts.append(str(value))
    return "|".join(parts)


def split_of_key(key: str) -> str:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    u = int.from_bytes(digest, "big") / float(1 << 64)
    train_cut = SPLIT_FRACTIONS["train"]
    validation_cut = train_cut + SPLIT_FRACTIONS["validation"]
    if u < train_cut:
        return "train"
    if u < validation_cut:
        return "validation"
    return "test"


def row_splits(frame: pd.DataFrame, schema: DatasetSchema = HILLSTROM) -> np.ndarray:
    columns = schema.columns
    return np.array(
        [split_of_key(canonical_row_key(row, columns)) for _, row in frame.iterrows()],
        dtype=object,
    )


def split_frame(frame: pd.DataFrame, schema: DatasetSchema = HILLSTROM) -> dict[str, pd.DataFrame]:
    labels = row_splits(frame, schema)
    return {name: frame.loc[labels == name].reset_index(drop=True) for name in SPLIT_FRACTIONS}


def splits_path(name: str = "hillstrom") -> Path:
    return split_dir() / f"{name}-{SPLIT_VERSION}.json"


def persist_splits(frame: pd.DataFrame, name: str = "hillstrom") -> dict[str, Any]:
    """Persist the validation/test row indices; train is the complement."""
    labels = row_splits(frame)
    manifest = read_manifest(name)
    payload: dict[str, Any] = {
        "dataset": name,
        "method": "blake2b64(content key) mapped to [0,1) with 60/20/20 cut points",
        "split_version": SPLIT_VERSION,
        "row_count": len(frame),
        "row_sha256": manifest["sha256"],
        "fractions": SPLIT_FRACTIONS,
        "validation": np.flatnonzero(labels == "validation").tolist(),
        "test": np.flatnonzero(labels == "test").tolist(),
        "train": "complement of validation and test",
        "counts": {kind: int((labels == kind).sum()) for kind in SPLIT_FRACTIONS},
    }
    path = splits_path(name)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def load_split_indices(name: str = "hillstrom") -> dict[str, np.ndarray]:
    path = splits_path(name)
    if not path.exists():
        raise FileNotFoundError(f"split file missing: {path}; run `make data`")
    payload = json.loads(path.read_text(encoding="utf-8"))
    n = int(payload["row_count"])
    validation = np.asarray(payload["validation"], dtype=int)
    test = np.asarray(payload["test"], dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[validation] = False
    mask[test] = False
    return {"train": np.flatnonzero(mask), "validation": validation, "test": test}


def load_splits(name: str = "hillstrom") -> dict[str, pd.DataFrame]:
    frame = load_frame(name)
    indices = load_split_indices(name)
    return {kind: frame.iloc[idx].reset_index(drop=True) for kind, idx in indices.items()}


def ensure_dataset(name: str = "hillstrom") -> None:
    """Make the cached raw file, manifest and splits available.

    A fresh clone has no ``data/raw`` (gitignored): fetch once, then regenerate
    the committed split file only if it is missing.
    """
    if not raw_path(name).exists() or not manifest_path(name).exists():
        fetch(name)
    verify_manifest(name)
    if not splits_path(name).exists():
        persist_splits(load_frame(name), name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download and verify the dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    payload = fetch("hillstrom", force=args.force) if args.fetch else verify_manifest("hillstrom")
    frame = load_frame("hillstrom")
    splits = persist_splits(frame)
    print(f"dataset      : {payload['dataset']}")
    print(f"sha256       : {payload['sha256']}")
    print(f"rows         : {payload['row_count']:,}  ({_human_size(int(payload['bytes']))})")
    print(f"columns      : {len(payload['columns'])}")
    print(f"split counts : {splits['counts']}")
    print(f"manifest     : {manifest_path()}")
    print(f"splits       : {splits_path()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
