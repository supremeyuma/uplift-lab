"""Write metric artefacts (JSON + one figure per metric) to ``reports/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uplift.config import ensure_dir


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload
