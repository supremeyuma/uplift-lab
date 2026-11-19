from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment
from uplift.models.base import ConstantLearner, UpliftModel, positive_class_proba
from uplift.models.causal_forest import CausalForestUplift
from uplift.models.features import covariates
from uplift.models.response import ResponseModel, logistic_learner
from uplift.models.tlearner import TLearner, _fit_arm, make_tlearner

FRAME = make_experiment(n=900, seed=17).frame
X = covariates(FRAME, HILLSTROM)
T = HILLSTROM.is_treated(FRAME).to_numpy(dtype=int)
Y = FRAME["conversion"].to_numpy(dtype=int)


@dataclass
class StubForest:
    effect_value: float = 0.03
    fitted: bool = False

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> StubForest:
        assert X.ndim == 2
        assert len(X) == len(t) == len(y)
        self.fitted = True
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted
        return np.full(len(X), self.effect_value)


def _factories() -> list[tuple[str, object]]:
    return [
        ("t-learner-logistic", lambda: make_tlearner("logistic", seed=17)),
        ("t-learner-lightgbm", lambda: make_tlearner("lightgbm", seed=17, n_folds=3)),
        ("response-logistic", lambda: ResponseModel(seed=17)),
        ("causal-forest-stub", lambda: CausalForestUplift(StubForest(), seed=17)),
    ]


@pytest.mark.parametrize(("label", "factory"), _factories(), ids=[f[0] for f in _factories()])
def test_all_models_share_interface(label: str, factory: object) -> None:
    model = factory()  # type: ignore[operator]
    assert isinstance(model, UpliftModel), label
    model.fit(X, T, Y)
    predictions = np.asarray(model.predict_uplift(X.head(25)))
    assert predictions.shape == (25,), label
    assert np.isfinite(predictions).all(), label


def test_predictions_are_out_of_fold() -> None:
    model = TLearner(lambda: logistic_learner(seed=17), n_folds=4, seed=17)
    model.fit(X, T, Y)

    assert len(model.oof_predictions_) == len(X)
    assert np.isfinite(model.oof_predictions_).all()

    seen = np.zeros(len(X), dtype=int)
    for fold in model.folds:
        train, validation = set(fold.train_index.tolist()), fold.validation_index
        assert train.isdisjoint(validation.tolist()), "a fold must not train on its own rows"
        recomputed = positive_class_proba(fold.treated, X.iloc[validation]) - positive_class_proba(
            fold.control, X.iloc[validation]
        )
        np.testing.assert_allclose(model.oof_predictions_[validation], recomputed, rtol=0, atol=1e-12)
        seen[validation] += 1

    assert (seen == 1).all(), "every training row is predicted exactly once, out of fold"


def test_final_models_see_the_whole_training_set() -> None:
    model = TLearner(lambda: logistic_learner(seed=17), n_folds=3, seed=17)
    model.fit(X, T, Y)
    mu0, mu1 = model.predict_arms(X.head(10))
    assert (mu1 >= 0).all()
    assert (mu1 <= 1).all()
    assert mu0.shape == mu1.shape == (10,)


def test_fit_arm_falls_back_to_constant_for_single_class() -> None:
    learner = _fit_arm(lambda: logistic_learner(seed=1), X.head(40), np.zeros(40))
    assert isinstance(learner, ConstantLearner)
    assert np.allclose(positive_class_proba(learner, X.head(3)), 0.0)


def test_constant_learner_returns_two_columns() -> None:
    proba = ConstantLearner(0.25).predict_proba(X.head(4))
    assert proba.shape == (4, 2)
    assert np.allclose(proba[:, 1], 0.25)


def test_tlearner_requires_fit_before_predict() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        TLearner().predict_uplift(X)
    with pytest.raises(RuntimeError, match="before fit"):
        CausalForestUplift(StubForest()).predict_uplift(X)
    with pytest.raises(RuntimeError, match="before fit"):
        ResponseModel().predict_proba(X)


def test_fit_rejects_single_arm_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="single arm"):
        TLearner().fit(X, np.zeros(len(X), dtype=int), Y)
    with pytest.raises(ValueError, match="equal length"):
        TLearner().fit(X, T[:-1], Y)
    broken = X.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="null"):
        TLearner().fit(broken, T, Y)
    with pytest.raises(ValueError, match="empty"):
        TLearner().fit(X.iloc[0:0], T[0:0], Y[0:0])


def test_unknown_learner_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown learner"):
        make_tlearner("transformer")


def test_causal_forest_reports_missing_econml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "econml", None)
    model = CausalForestUplift()
    with pytest.raises(ImportError, match="causal-forest"):
        model.fit(X, T, Y)


def test_causal_forest_backend_receives_a_numeric_matrix() -> None:
    backend = StubForest(effect_value=0.07)
    model = CausalForestUplift(backend)
    model.fit(X, T, Y)
    predictions = model.predict_uplift(X.head(5))
    assert np.allclose(predictions, 0.07)


def test_response_model_is_not_leaking_treatment() -> None:
    model = ResponseModel(seed=3)
    model.fit(X, T, Y)
    assert "segment" not in model.columns
    assert "conversion" not in model.columns


def test_predict_rejects_missing_covariates() -> None:
    model = TLearner(lambda: logistic_learner(seed=2), n_folds=3, seed=2)
    model.fit(X, T, Y)
    with pytest.raises(ValueError, match="missing covariates"):
        model.predict_uplift(X.drop(columns=["history"]))
