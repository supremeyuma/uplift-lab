"""Outcome-model baseline p(y | x), ignoring treatment.

This is deliberately strong: if a response model ranks customers as well as an
uplift model, the incremental targeting claim is not supported. It is also the
control learner inside the T-learner, so both baselines share preprocessing.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from uplift.config import SEED
from uplift.data.schema import HILLSTROM, DatasetSchema
from uplift.models.base import (
    Learner,
    check_fit_inputs,
    check_predict_inputs,
    positive_class_proba,
)
from uplift.models.features import covariates, learner_pipeline


def logistic_learner(seed: int = SEED, schema: DatasetSchema = HILLSTROM) -> Learner:
    return cast(
        Learner,
        learner_pipeline(
            LogisticRegression(max_iter=2000, random_state=seed, solver="lbfgs"), schema
        ),
    )


def lightgbm_learner(seed: int = SEED, schema: DatasetSchema = HILLSTROM) -> Learner:
    return cast(
        Learner,
        learner_pipeline(
            LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=15,
                min_child_samples=40,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            ),
            schema,
        ),
    )


class ResponseModel:
    """Ranking baseline: P(conversion | x) fitted on both arms pooled."""

    def __init__(
        self,
        learner: Learner | None = None,
        *,
        schema: DatasetSchema = HILLSTROM,
        seed: int = SEED,
    ) -> None:
        self.schema = schema
        self.seed = seed
        self.learner: Learner = learner if learner is not None else logistic_learner(seed)
        self.columns: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> None:
        check_fit_inputs(X, t, y)
        self.columns = list(X.columns)
        self.learner.fit(X, np.asarray(y))
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ResponseModel.predict_proba called before fit")
        check_predict_inputs(X, self.columns)
        return positive_class_proba(self.learner, X)

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Response-model targeting score (higher = more likely to convert)."""
        return self.predict_proba(X)


def response_model_from_frame(
    frame: pd.DataFrame, *, schema: DatasetSchema = HILLSTROM, seed: int = SEED
) -> ResponseModel:
    model = ResponseModel(schema=schema, seed=seed)
    X = covariates(frame, schema)
    t = schema.is_treated(frame).to_numpy(dtype=int)
    y = frame["conversion"].to_numpy(dtype=int)
    model.fit(X, t, y)
    return model
