"""Tests for raw dataset schema validation."""

import numpy as np
import pandas as pd
import pytest

from industrial_ai.config import load_config
from industrial_ai.data.schema import SchemaError, validate_raw_frame


def _columns():
    return load_config().columns


def test_valid_frame_passes(raw_frame: pd.DataFrame) -> None:
    validate_raw_frame(raw_frame, _columns())


def test_empty_frame_rejected(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="no rows"):
        validate_raw_frame(raw_frame.iloc[0:0], _columns())


def test_missing_column_named_in_error(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match=r"Missing columns.*Torque \[Nm\]"):
        validate_raw_frame(raw_frame.drop(columns=["Torque [Nm]"]), _columns())


def test_unexpected_column_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame["Surprise"] = 1
    with pytest.raises(SchemaError, match="Unexpected columns.*Surprise"):
        validate_raw_frame(raw_frame, _columns())


def test_missing_values_reported(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[3, "Air temperature [K]"] = np.nan
    with pytest.raises(SchemaError, match=r"'Air temperature \[K\]' has 1 missing"):
        validate_raw_frame(raw_frame, _columns())


def test_non_numeric_feature_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame["Torque [Nm]"] = "high"
    with pytest.raises(SchemaError, match=r"'Torque \[Nm\]' must be numeric"):
        validate_raw_frame(raw_frame, _columns())


def test_infinite_values_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, "Rotational speed [rpm]"] = np.inf
    with pytest.raises(SchemaError, match="infinite"):
        validate_raw_frame(raw_frame, _columns())


def test_non_binary_label_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, "Machine failure"] = 2
    with pytest.raises(SchemaError, match=r"'Machine failure' must contain only 0/1"):
        validate_raw_frame(raw_frame, _columns())


def test_non_binary_failure_mode_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, "HDF"] = -1
    with pytest.raises(SchemaError, match="'HDF' must contain only 0/1"):
        validate_raw_frame(raw_frame, _columns())


def test_unknown_type_category_rejected(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, "Type"] = "X"
    with pytest.raises(SchemaError, match="unexpected categories"):
        validate_raw_frame(raw_frame, _columns())


def test_duplicate_rows_rejected(raw_frame: pd.DataFrame) -> None:
    doubled = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicated"):
        validate_raw_frame(doubled, _columns())


def test_all_problems_reported_together(raw_frame: pd.DataFrame) -> None:
    raw_frame.loc[0, "Type"] = "X"
    raw_frame.loc[1, "Machine failure"] = 5
    with pytest.raises(SchemaError) as info:
        validate_raw_frame(raw_frame, _columns())
    assert len(info.value.problems) == 2
