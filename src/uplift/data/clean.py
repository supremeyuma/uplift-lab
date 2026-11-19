"""Remove generated artefacts; keep the fetched raw data and committed manifests."""

from __future__ import annotations

import shutil

from uplift.config import ROOT, reports_dir


def main() -> int:
    removed: list[str] = []
    for directory in (ROOT / ".pytest_cache", ROOT / ".mypy_cache", ROOT / ".ruff_cache"):
        if directory.exists():
            shutil.rmtree(directory)
            removed.append(directory.name)
    report_dir = reports_dir()
    if report_dir.exists():
        for item in report_dir.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        removed.append(f"{report_dir.name}/*")
    print("removed: " + (", ".join(removed) if removed else "nothing"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
