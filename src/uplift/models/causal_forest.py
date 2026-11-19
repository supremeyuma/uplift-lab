"""Optional econml CausalForestDML path behind the same protocol.

The backend is injected so the wrapper (preprocessing, validation, shape
handling) can be tested offline even when econml is not installed.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from uplift.config import SEED
from uplift.data.schema import HILLSTROM, DatasetSchema
from uplift.models.base import check_fit_inputs, check_predict_inputs
from uplift.models.features import preprocessing


class ForestBackend(Protocol):
    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> Any: ...

    def effect(self, X: np.ndarray) -> np.ndarray: ...


def _build_econml_backend(seed: int, n_estimators: int) -> ForestBackend:
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "econml is not installed; run `uv sync --extra causal-forest` "
            "or use the T-learner"
        ) from exc

    return CausalForestDML(  # type: ignore[no-any-return]
        model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=seed),
        model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=20, random_state=seed),
        discrete_treatment=True,
        n_estimators=n_estimators,
        cv=3,
        random_state=seed,
        n_jobs=1,
    )


class CausalForestUplift:
    """econml CausalForestDML wrapped with the shared preprocessing."""

    def __init__(
        self,
        backend: ForestBackend | None = None,
        *,
        schema: DatasetSchema = HILLSTROM,
        seed: int = SEED,
        n_estimators: int = 300,
    ) -> None:
        self.schema = schema
        self.seed = seed
        self.n_estimators = n_estimators
        self._backend = backend
        self._prep = preprocessing(schema)
        self.columns: list[str] = []
        self._fitted = False

    @property
    def name(self) -> str:
        return "causal-forest"

    def fit(self, X: pd.DataFrame, t: np.ndarray, y: np.ndarray) -> None:
        check_fit_inputs(X, t, y)
        self.columns = list(X.columns)
        if self._backend is None:
            self._backend = _build_econml_backend(self.seed, self.n_estimators)
        matrix = np.asarray(self._prep.fit_transform(X), dtype=float)
        self._backend.fit(matrix, np.asarray(t, dtype=int), np.asarray(y, dtype=float))
        self._fitted = True

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._backend is None:
            raise RuntimeError("CausalForestUplift.predict_uplift called before fit")
        check_predict_inputs(X, self.columns)
        matrix = np.asarray(self._prep.transform(X), dtype=float)
        effect = np.asarray(self._backend.effect(matrix), dtype=float).reshape(-1)
        if len(effect) != len(X):
            raise ValueError(f"backend returned {len(effect)} effects for {len(X)} rows")
        return effect
