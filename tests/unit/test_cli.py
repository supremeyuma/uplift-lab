from pathlib import Path

import pytest

from uplift.data import clean, loader
from uplift.data.synthetic import make_experiment
from uplift.design import power


@pytest.fixture
def prepared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    reports = tmp_path / "reports"
    monkeypatch.setenv("UPLIFT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("UPLIFT_REPORTS_DIR", str(reports))
    frame = make_experiment(n=1500, seed=5).frame
    path = loader.raw_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    loader.write_manifest("hillstrom", path, frame)
    loader.persist_splits(frame)
    return tmp_path


def test_power_cli_prints_mde_and_realised_power(
    prepared: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert power._main(["--data", "hillstrom"]) == 0
    out = capsys.readouterr().out
    assert "MDE @ 80% power" in out
    assert "realised power" in out
    assert "baseline conversion" in out


def test_power_cli_rejects_unknown_dataset(prepared: Path) -> None:
    with pytest.raises(SystemExit):
        power._main(["--data", "criteo"])


def test_clean_removes_generated_artefacts(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(clean, "ROOT", prepared)
    (prepared / ".pytest_cache").mkdir()
    (prepared / ".mypy_cache").mkdir()
    reports = prepared / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text("{}", encoding="utf-8")
    (reports / ".gitkeep").write_text("", encoding="utf-8")

    assert clean.main() == 0
    out = capsys.readouterr().out
    assert "removed" in out
    assert not (prepared / ".pytest_cache").exists()
    assert not (reports / "metrics.json").exists()
    assert (reports / ".gitkeep").exists()
    assert (prepared / "data" / "raw" / "hillstrom.csv").exists()


def test_clean_is_safe_when_nothing_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(clean, "ROOT", tmp_path)
    monkeypatch.setenv("UPLIFT_REPORTS_DIR", str(tmp_path / "absent"))
    assert clean.main() == 0
    assert "nothing" in capsys.readouterr().out
