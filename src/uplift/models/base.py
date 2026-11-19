"""Shared model contracts and input guards."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


class Learner(Protocol):
    """Minimal sklearn-classifier surface used by the two-model estimator."""

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Any: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class UpliftModel(Protocol):
    """Everything downstream (evaluation, decision, report) is allowed to assume."""

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> None: ...

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray: ...


class ConstantLearner:
    """Fallback for a fold whose arm contains a single outcome class."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ConstantLearner:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        positive = np.full(len(X), self.value, dtype=float)
        return np.column_stack([1.0 - positive, positive])


def check_fit_inputs(X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> None:
    n = len(X)
    if len(t) != n or len(y) != n:
        raise ValueError(f"X, t and y must have equal length; got {n}, {len(t)}, {len(y)}")
    if n == 0:
        raise ValueError("cannot fit on an empty design matrix")
    if X.isna().to_numpy().any():
        raise ValueError("design matrix contains nulls")
    values = np.unique(t)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"treatment must be binary 0/1; got {values.tolist()}")
    if len(values) < 2:
        raise ValueError("treatment has a single arm; contrast is not identified")
    if not np.isfinite(y).all():
        raise ValueError("outcome contains non-finite values")


def check_predict_inputs(X: pd.DataFrame, expected_columns: list[str]) -> None:
    missing = [col for col in expected_columns if col not in X.columns]
    if missing:
        raise ValueError(f"design matrix missing covariates: {missing}")


def positive_class_proba(learner: Learner, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(learner.predict_proba(X))
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError(f"learner must return a two-column predict_proba matrix, got {proba.shape}")
    return proba[:, 1]
