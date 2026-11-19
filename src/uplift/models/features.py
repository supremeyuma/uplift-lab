"""Feature preparation shared by every estimator."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from uplift.data.schema import DatasetSchema


def covariates(frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Return the covariate block, never the treatment or an outcome."""
    forbidden = {schema.treatment, *schema.outcomes}
    leaked = [col for col in schema.covariates if col in forbidden]
    if leaked:  # pragma: no cover - schema guard
        raise ValueError(f"covariates must not include treatment/outcome columns: {leaked}")
    return frame.loc[:, list(schema.covariates)].reset_index(drop=True)


def preprocessing(schema: DatasetSchema) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(schema.numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(schema.categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def learner_pipeline(estimator: object, schema: DatasetSchema) -> Pipeline:
    return Pipeline([("prep", preprocessing(schema)), ("model", estimator)])
