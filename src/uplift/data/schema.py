"""Column contract for the Hillstrom e-mail experiment.

The schema is a frozen description of what the rest of the code is allowed to
assume. ``validate`` is the single place that turns schema drift into an
exception instead of a silently wrong estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import pandas as pd


class SchemaError(ValueError):
    """Raised when a dataframe does not satisfy the declared contract."""


@dataclass(frozen=True)
class DatasetSchema:
    name: str
    treatment: str
    control_label: str
    treatment_labels: tuple[str, ...]
    outcomes: tuple[str, ...]
    covariates: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    extra: tuple[str, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> tuple[str, ...]:
        return (self.treatment, *self.outcomes, *self.covariates, *self.extra)

    def treated_label(self) -> str:
        """Label used when the multi-arm experiment is collapsed to treated/control."""
        return self.treatment_labels[0]

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [c for c in self.columns if c not in frame.columns]
        if self.treatment in missing:
            raise SchemaError(f"missing treatment column: {self.treatment!r}")
        if missing:
            raise SchemaError(f"missing columns: {missing}")
        if frame.empty:
            raise SchemaError(f"{self.name}: dataframe is empty")

        labels = set(frame[self.treatment].dropna().unique())
        if self.control_label not in labels:
            raise SchemaError(
                f"{self.treatment!r} has no control label {self.control_label!r}; saw {sorted(map(str, labels))}"
            )
        if not labels.intersection(self.treatment_labels):
            raise SchemaError(
                f"{self.treatment!r} has no treatment label from {self.treatment_labels!r}; "
                f"saw {sorted(map(str, labels))}"
            )
        unexpected = labels - {self.control_label, *self.treatment_labels}
        if unexpected:
            raise SchemaError(f"unexpected {self.treatment!r} labels: {sorted(map(str, unexpected))}")

        for col in self.numeric:
            if not pd.api.types.is_numeric_dtype(frame[col]):
                raise SchemaError(f"column {col!r} must be numeric, got {frame[col].dtype}")
            if frame[col].isna().any():
                raise SchemaError(f"column {col!r} contains nulls")
        for col in self.categorical:
            if frame[col].isna().any():
                raise SchemaError(f"column {col!r} contains nulls")
        for col in self.outcomes:
            if frame[col].isna().any():
                raise SchemaError(f"outcome {col!r} contains nulls")
        if "conversion" in frame.columns:
            conv = frame["conversion"]
            if not conv.isin([0, 1]).all():
                raise SchemaError("conversion must be binary (0/1)")
        if "spend" in frame.columns and (frame["spend"] < 0).any():
            raise SchemaError("spend must be non-negative")

    def is_treated(self, frame: pd.DataFrame) -> pd.Series:
        return frame[self.treatment].isin(self.treatment_labels)


HILLSTROM: Final[DatasetSchema] = DatasetSchema(
    name="hillstrom",
    treatment="segment",
    control_label="No E-Mail",
    treatment_labels=("Mens E-Mail", "Womens E-Mail"),
    outcomes=("conversion", "spend"),
    covariates=(
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
    ),
    numeric=("recency", "history", "mens", "womens", "newbie"),
    categorical=("history_segment", "zip_code", "channel"),
    extra=("visit",),
)

SCHEMAS: Final[dict[str, DatasetSchema]] = {HILLSTROM.name: HILLSTROM}


def get_schema(name: str) -> DatasetSchema:
    try:
        return SCHEMAS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise SchemaError(f"unknown dataset {name!r}; known: {sorted(SCHEMAS)}") from exc
