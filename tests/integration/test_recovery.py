"""Synthetic recovery: the only honest way to check an uplift estimator."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment
from uplift.models.causal_forest import CausalForestUplift
from uplift.models.features import covariates
from uplift.models.tlearner import make_tlearner


def test_tlearner_recovers_constant_effect_within_tolerance() -> None:
    experiment = make_experiment(n=20000, seed=101)
    X = covariates(experiment.frame, HILLSTROM)
    t = HILLSTROM.is_treated(experiment.frame).to_numpy(dtype=int)
    y = experiment.frame["conversion"].to_numpy(dtype=int)

    model = make_tlearner("logistic", seed=101)
    model.fit(X, t, y)
    recovered = float(np.mean(model.oof_predictions_))

    # 15k+ rows: the sampling standard error of the recovered ATE is ~0.005,
    # so 0.015 is a 3-sigma tolerance, stated rather than tuned after the fact.
    assert abs(recovered - experiment.ate) < 0.015


def test_tlearner_recovers_known_heterogeneity_by_segment() -> None:
    experiment = make_experiment(n=8000, seed=202, heterogeneity=0.5)
    frame = experiment.frame
    X = covariates(frame, HILLSTROM)
    t = HILLSTROM.is_treated(frame).to_numpy(dtype=int)
    y = frame["conversion"].to_numpy(dtype=int)

    model = make_tlearner("logistic", seed=202)
    model.fit(X, t, y)

    above = (frame["history"] > frame["history"].median()).to_numpy()
    true_gap = float(experiment.true_uplift[above].mean() - experiment.true_uplift[~above].mean())
    recovered_gap = float(model.oof_predictions_[above].mean() - model.oof_predictions_[~above].mean())

    assert true_gap > 0.02, "fixture must contain a real subgroup difference"
    assert recovered_gap > 0
    assert abs(recovered_gap - true_gap) < 0.025
    assert abs(float(np.mean(model.oof_predictions_)) - experiment.ate) < 0.015


@pytest.mark.skipif(
    importlib.util.find_spec("econml") is None,
    reason="optional econml extra is not installed",
)
def test_causal_forest_optional_backend_recovers_positive_effect() -> None:  # pragma: no cover
    experiment = make_experiment(n=1500, seed=303)
    X = covariates(experiment.frame, HILLSTROM)
    t = HILLSTROM.is_treated(experiment.frame).to_numpy(dtype=int)
    y = experiment.frame["conversion"].to_numpy(dtype=int)

    model = CausalForestUplift(seed=303, n_estimators=100)
    model.fit(X, t, y)
    recovered = float(np.mean(model.predict_uplift(X)))
    assert 0.0 < recovered < 0.15
