"""Filesystem locations and shared constants.

Every path is derived from the repository root so that a fresh clone behaves
identically. ``UPLIFT_DATA_DIR`` and ``UPLIFT_REPORTS_DIR`` can point at a
different location (used by the fixture pipeline in CI).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Master seed. Every stochastic component derives its seed from this value.
SEED = 20240501

#: Bootstrap resamples used for every interval reported by the project.
BOOTSTRAP_RESAMPLES = 2000

ALPHA = 0.05

SPLIT_FRACTIONS: dict[str, float] = {"train": 0.60, "validation": 0.20, "test": 0.20}


def _dir_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default


def data_dir() -> Path:
    return _dir_from_env("UPLIFT_DATA_DIR", ROOT / "data")


def raw_dir() -> Path:
    return data_dir() / "raw"


def manifest_dir() -> Path:
    return data_dir() / "manifests"


def split_dir() -> Path:
    return data_dir() / "splits"


def reports_dir() -> Path:
    return _dir_from_env("UPLIFT_REPORTS_DIR", ROOT / "reports")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
