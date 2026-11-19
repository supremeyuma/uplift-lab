import numpy as np
import pandas as pd
import pytest

from uplift.data.schema import HILLSTROM, DatasetSchema, SchemaError, get_schema
from uplift.data.synthetic import make_experiment


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_experiment(n=200, seed=7).frame


def test_validate_accepts_synthetic_hillstrom_frame(frame: pd.DataFrame) -> None:
    HILLSTROM.validate(frame)
    assert HILLSTROM.columns == (
        "segment",
        "conversion",
        "spend",
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "visit",
    )


def test_validate_rejects_missing_treatment_column(frame: pd.DataFrame) -> None:
    broken = frame.drop(columns=["segment"])
    with pytest.raises(SchemaError, match="missing treatment column"):
        HILLSTROM.validate(broken)


def test_validate_rejects_missing_covariate(frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        HILLSTROM.validate(frame.drop(columns=["history"]))


def test_validate_rejects_unknown_treatment_label(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[broken.index[0], "segment"] = "Carrier Pigeon"
    with pytest.raises(SchemaError, match="unexpected"):
        HILLSTROM.validate(broken)


def test_validate_rejects_missing_control_arm(frame: pd.DataFrame) -> None:
    broken = frame[frame["segment"] != HILLSTROM.control_label]
    with pytest.raises(SchemaError, match="no control label"):
        HILLSTROM.validate(broken)


def test_validate_rejects_non_binary_conversion(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[broken.index[0], "conversion"] = 2
    with pytest.raises(SchemaError, match="binary"):
        HILLSTROM.validate(broken)


def test_validate_rejects_null_covariate(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[broken.index[0], "history"] = np.nan
    with pytest.raises(SchemaError, match="nulls"):
        HILLSTROM.validate(broken)


def test_is_treated_marks_both_email_arms() -> None:
    frame = pd.DataFrame({"segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]})
    assert HILLSTROM.is_treated(frame).tolist() == [False, True, True]


def test_get_schema_rejects_unknown_dataset() -> None:
    with pytest.raises(SchemaError, match="unknown dataset"):
        get_schema("criteo")


def test_schema_is_frozen() -> None:
    schema: DatasetSchema = HILLSTROM
    with pytest.raises(AttributeError):
        schema.treatment = "other"  # type: ignore[misc]


def test_treated_label_and_columns_are_derived() -> None:
    assert HILLSTROM.treated_label() == "Mens E-Mail"
    assert HILLSTROM.columns[0] == "segment"
    assert set(HILLSTROM.outcomes) <= set(HILLSTROM.columns)


def test_validate_rejects_empty_frame(frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="empty"):
        HILLSTROM.validate(frame.iloc[0:0])


def test_validate_rejects_negative_spend(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[broken.index[0], "spend"] = -1.0
    with pytest.raises(SchemaError, match="non-negative"):
        HILLSTROM.validate(broken)


def test_validate_rejects_non_numeric_covariate(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken["history"] = broken["history"].astype(str)
    with pytest.raises(SchemaError, match="must be numeric"):
        HILLSTROM.validate(broken)


def test_validate_rejects_null_outcome(frame: pd.DataFrame) -> None:
    broken = frame.copy()
    broken.loc[broken.index[0], "spend"] = np.nan
    with pytest.raises(SchemaError, match="spend"):
        HILLSTROM.validate(broken)
