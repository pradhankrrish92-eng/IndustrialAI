"""Tests for loading the raw dataset from disk."""

from pathlib import Path

import pandas as pd
import pytest

from industrial_ai.data.loader import DatasetNotFoundError, load_dataset, load_raw_dataset
from industrial_ai.data.schema import SchemaError


def _write(cfg, frame: pd.DataFrame) -> Path:
    path = cfg.paths.raw_data_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_load_returns_clean_column_names(cfg, raw_frame: pd.DataFrame) -> None:
    _write(cfg, raw_frame)
    df = load_dataset(cfg)
    assert list(df.columns) == list(cfg.columns.rename.values())
    assert len(df) == len(raw_frame)
    assert set(cfg.columns.feature_columns) <= set(df.columns)


def test_missing_file_gives_actionable_message(cfg) -> None:
    with pytest.raises(DatasetNotFoundError, match="download_data.py"):
        load_dataset(cfg)


def test_empty_file_raises_schema_error(cfg) -> None:
    path = cfg.paths.raw_data_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    with pytest.raises(SchemaError, match="Could not parse"):
        load_raw_dataset(path, cfg.columns)


def test_wrong_columns_raise_schema_error(cfg, raw_frame: pd.DataFrame) -> None:
    path = _write(cfg, raw_frame.drop(columns=["Type"]))
    with pytest.raises(SchemaError, match="Missing columns"):
        load_raw_dataset(path, cfg.columns)
