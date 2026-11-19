"""Two-model (T-learner) uplift estimator with cross-fitting.

Two outcome models are fitted per fold: one on treated rows, one on control
rows. The uplift for a row is mu1(x) - mu0(x). Training rows only ever receive
out-of-fold predictions, so the scores used for calibration and threshold
selection are not contaminated by the row's own outcome.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from uplift.config import SEED
from uplift.data.schema import HILLSTROM, DatasetSchema
from uplift.models.base import (
    ConstantLearner,
    Learner,
    check_fit_inputs,
    check_predict_inputs,
    positive_class_proba,
)
from uplift.models.features import covariates
from uplift.models.response import lightgbm_learner, logistic_learner


@dataclass(frozen=True)
class FoldModels:
    """Everything needed to audit where an out-of-fold prediction came from."""

    train_index: np.ndarray
    validation_index: np.ndarray
    control: Learner
    treated: Learner


def _fit_arm(
    factory: Callable[[], Learner], X: pd.DataFrame, y: np.ndarray
) -> Learner:
    if np.unique(y).size < 2:
        return ConstantLearner(float(np.mean(y)) if len(y) else 0.0)
    learner = factory()
    learner.fit(X, y)
    return learner


class TLearner:
    """Separate treated/control outcome models, difference of predictions."""

    def __init__(
        self,
        learner_factory: Callable[[], Learner] | None = None,
        *,
        schema: DatasetSchema = HILLSTROM,
        n_folds: int = 5,
        seed: int = SEED,
    ) -> None:
        if n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        self.schema = schema
        self.seed = seed
        self.n_folds = n_folds
        self.learner_factory: Callable[[], Learner] = learner_factory or (
            lambda: logistic_learner(seed)
        )
        self.folds: list[FoldModels] = []
        self.oof_predictions_: np.ndarray = np.array([], dtype=float)
        self._control: Learner | None = None
        self._treated: Learner | None = None
        self.columns: list[str] = []

    @property
    def name(self) -> str:
        return "t-learner"

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> None:
        check_fit_inputs(X, t, y)
        t = np.asarray(t, dtype=int)
        y = np.asarray(y, dtype=float)
        self.columns = list(X.columns)

        oof = np.full(len(X), np.nan, dtype=float)
        splitter = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self.folds = []
        for train_idx, val_idx in splitter.split(X, t):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            t_train, y_train = t[train_idx], y[train_idx]

            control = _fit_arm(self.learner_factory, X_train.loc[t_train == 0], y_train[t_train == 0])
            treated = _fit_arm(self.learner_factory, X_train.loc[t_train == 1], y_train[t_train == 1])
            mu0 = positive_class_proba(control, X_val)
            mu1 = positive_class_proba(treated, X_val)
            oof[val_idx] = mu1 - mu0
            self.folds.append(
                FoldModels(
                    train_index=np.asarray(train_idx),
                    validation_index=np.asarray(val_idx),
                    control=control,
                    treated=treated,
                )
            )

        if np.isnan(oof).any():  # pragma: no cover - StratifiedKFold covers every row
            raise RuntimeError("cross-fitting left rows without an out-of-fold prediction")
        self.oof_predictions_ = oof

        treated_mask = t == 1
        self._control = _fit_arm(self.learner_factory, X.loc[~treated_mask], y[~treated_mask])
        self._treated = _fit_arm(self.learner_factory, X.loc[treated_mask], y[treated_mask])

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self._control is None or self._treated is None:
            raise RuntimeError("TLearner.predict_uplift called before fit")
        check_predict_inputs(X, self.columns)
        return np.asarray(
            positive_class_proba(self._treated, X) - positive_class_proba(self._control, X)
        )

    def predict_arms(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(mu0, mu1) from the models fitted on all training rows."""
        if self._control is None or self._treated is None:
            raise RuntimeError("TLearner.predict_arms called before fit")
        check_predict_inputs(X, self.columns)
        return positive_class_proba(self._control, X), positive_class_proba(self._treated, X)


def make_tlearner(
    kind: str = "logistic", *, seed: int = SEED, n_folds: int = 5
) -> TLearner:
    if kind == "logistic":
        return TLearner(lambda: logistic_learner(seed), seed=seed, n_folds=n_folds)
    if kind == "lightgbm":
        return TLearner(lambda: lightgbm_learner(seed), seed=seed, n_folds=n_folds)
    raise ValueError(f"unknown learner kind {kind!r}")


def _check_recovery(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover a known ATE on a synthetic fixture")
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--logit-effect", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--learner", default="logistic", choices=["logistic", "lightgbm"])
    args = parser.parse_args(argv)

    from uplift.data.synthetic import make_experiment

    experiment = make_experiment(n=args.n, seed=args.seed, logit_effect=args.logit_effect)
    X = covariates(experiment.frame, HILLSTROM)
    t = HILLSTROM.is_treated(experiment.frame).to_numpy(dtype=int)
    y = experiment.frame["conversion"].to_numpy(dtype=int)

    model = make_tlearner(args.learner, seed=args.seed)
    model.fit(X, t, y)
    recovered = float(np.mean(model.oof_predictions_))
    print(f"fixture            : synthetic, n={args.n}, seed={args.seed}")
    print(f"true ATE           : {experiment.ate:.4f}")
    print(f"recovered ATE (OOF): {recovered:.4f}")
    print(f"absolute error     : {abs(recovered - experiment.ate):.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    arguments = sys.argv[1:]
    if "--check-recovery" in arguments:
        arguments.remove("--check-recovery")
        raise SystemExit(_check_recovery(arguments))
    raise SystemExit("usage: python -m uplift.models.tlearner --check-recovery")
