from __future__ import annotations

import numpy as np
import pytest

from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment
from uplift.eval.calibration import calibration_error, calibration_table, is_overconfident


def _arrays(seed: int = 12, n: int = 6000) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    experiment = make_experiment(n=n, seed=seed)
    frame = experiment.frame
    t = HILLSTROM.is_treated(frame).to_numpy(dtype=float)
    y = frame["conversion"].to_numpy(dtype=float)
    return y, t, experiment.true_uplift, experiment.ate


def test_calibration_table_covers_every_row_once() -> None:
    y, t, score, _ = _arrays()
    table = calibration_table(y, t, score, bins=10, resamples=50, seed=1)
    assert len(table) == 10
    assert sum(row.n for row in table) == len(y)
    assert [row.decile for row in table] == list(range(10))
    assert table[0].predicted_mean >= table[-1].predicted_mean


def test_calibration_flags_overconfident_scores() -> None:
    y, t, score, ate = _arrays()
    true_error = calibration_error(y, t, score)
    overconfident = score * 3.0
    over_error = calibration_error(y, t, overconfident)

    assert is_overconfident(y, t, overconfident)
    assert not is_overconfident(y, t, score)
    assert over_error > true_error
    assert over_error > ate


def test_calibration_interval_is_deterministic() -> None:
    y, t, score, _ = _arrays(n=2000)
    first = calibration_table(y, t, score, bins=5, resamples=100, seed=8)
    second = calibration_table(y, t, score, bins=5, resamples=100, seed=8)
    assert first == second
    for row in first:
        assert row.ci_low <= row.observed_uplift <= row.ci_high


def test_calibration_rejects_invalid_bins() -> None:
    y, t, score, _ = _arrays(n=1000)
    with pytest.raises(ValueError, match="bins"):
        calibration_table(y, t, score, bins=0)
