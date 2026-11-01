"""Unit tests for configuration loading and validation."""

from dataclasses import replace
from pathlib import Path

import pytest

from industrial_ai import __version__
from industrial_ai.config import (
    ColumnsConfig,
    ConfigError,
    SplitConfig,
    load_config,
)


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg.columns.target == "machine_failure"
    assert cfg.split.seed == 42
    assert cfg.paths.data_dir.is_absolute()
    assert cfg.paths.raw_data_path.name == "ai4i2020.csv"


def test_features_exclude_target_identifiers_and_leakage() -> None:
    cols = load_config().columns
    features = set(cols.feature_columns)
    assert cols.target not in features
    assert not features & set(cols.identifier_columns)
    assert not features & set(cols.leakage_columns)
    assert set(cols.leakage_columns) == {"twf", "hdf", "pwf", "osf", "rnf"}


def test_split_sizes_must_sum_to_one() -> None:
    with pytest.raises(ConfigError, match="sum to 1.0"):
        SplitConfig(seed=1, train_size=0.6, val_size=0.3, test_size=0.3)


def test_split_sizes_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="positive"):
        SplitConfig(seed=1, train_size=1.0, val_size=0.0, test_size=0.0)


def test_leakage_column_as_feature_is_rejected() -> None:
    cols = load_config().columns
    with pytest.raises(ConfigError, match="leakage"):
        replace(cols, numeric_features=(*cols.numeric_features, "hdf"))


def test_target_as_feature_is_rejected() -> None:
    cols = load_config().columns
    with pytest.raises(ConfigError, match="target"):
        ColumnsConfig(
            target=cols.target,
            numeric_features=(*cols.numeric_features, cols.target),
            categorical_features=cols.categorical_features,
            identifier_columns=cols.identifier_columns,
            leakage_columns=cols.leakage_columns,
            rename=cols.rename,
        )


def test_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_invalid_toml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(bad)


def test_missing_section_raises(tmp_path: Path) -> None:
    partial = tmp_path / "partial.toml"
    partial.write_text("[paths]\ndata_dir='data'\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(partial)
