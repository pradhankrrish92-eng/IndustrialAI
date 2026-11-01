"""Typed, validated project configuration loaded from ``configs/default.toml``.

All paths, seeds, split ratios and column roles live here so that no other module
needs magic constants. Validation happens at load time so misconfiguration (for
example a leakage column listed as a feature) fails immediately and loudly.
"""

from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ConfigError(ValueError):
    """Raised when the configuration file is missing, incomplete or inconsistent."""


@dataclass(frozen=True)
class PathsConfig:
    """Absolute filesystem locations used by the project."""

    data_dir: Path
    raw_filename: str
    artifacts_dir: Path
    reports_dir: Path

    @property
    def raw_data_path(self) -> Path:
        """Location of the raw dataset CSV."""
        return self.data_dir / "raw" / self.raw_filename


@dataclass(frozen=True)
class DataConfig:
    """Where the dataset comes from and how to verify it."""

    dataset_page_url: str
    download_url: str
    expected_sha256: str  # lowercase hex of the CSV; empty string disables the check

    def __post_init__(self) -> None:
        if not self.download_url.startswith("https://"):
            raise ConfigError("[data] download_url must use https.")
        if self.expected_sha256 and not _SHA256_RE.fullmatch(self.expected_sha256):
            raise ConfigError(
                "[data] expected_sha256 must be 64 lowercase hex characters or empty."
            )


@dataclass(frozen=True)
class SplitConfig:
    """Stratified train/validation/test split settings."""

    seed: int
    train_size: float
    val_size: float
    test_size: float

    def __post_init__(self) -> None:
        sizes = (self.train_size, self.val_size, self.test_size)
        if any(s <= 0 for s in sizes):
            raise ConfigError(f"Split sizes must all be positive, got {sizes}.")
        if not math.isclose(sum(sizes), 1.0, abs_tol=1e-9):
            raise ConfigError(f"Split sizes must sum to 1.0, got {sum(sizes)}.")


@dataclass(frozen=True)
class ColumnsConfig:
    """Roles of dataset columns, using cleaned (snake_case) names."""

    target: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    leakage_columns: tuple[str, ...]
    rename: dict[str, str]
    allowed_categories: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = self.feature_columns
        if len(set(features)) != len(features):
            raise ConfigError("A column is listed more than once among the features.")

        bad_keys = set(self.allowed_categories) - set(self.categorical_features)
        if bad_keys:
            raise ConfigError(
                f"allowed_categories refers to non-categorical columns: {sorted(bad_keys)}"
            )

        forbidden = {
            "target": {self.target},
            "identifier": set(self.identifier_columns),
            "leakage": set(self.leakage_columns),
        }
        for role, names in forbidden.items():
            overlap = names & set(features)
            if overlap:
                raise ConfigError(
                    f"Feature columns must not include {role} columns: {sorted(overlap)}"
                )

        known = set(self.rename.values())
        referenced = {self.target, *features, *self.identifier_columns, *self.leakage_columns}
        unknown = referenced - known
        if unknown:
            raise ConfigError(f"Columns not defined in [columns.rename]: {sorted(unknown)}")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Model input columns: numeric first, then categorical."""
        return (*self.numeric_features, *self.categorical_features)


@dataclass(frozen=True)
class Config:
    """Top-level project configuration."""

    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    columns: ColumnsConfig


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        section = raw[name]
    except KeyError as exc:
        raise ConfigError(f"Missing section [{name}] in configuration.") from exc
    if not isinstance(section, dict):
        raise ConfigError(f"[{name}] must be a table.")
    return section


def _get(section: dict[str, Any], key: str, name: str) -> Any:
    try:
        return section[key]
    except KeyError as exc:
        raise ConfigError(f"Missing key '{key}' in [{name}].") from exc


def load_config(path: Path | None = None) -> Config:
    """Load and validate configuration.

    Args:
        path: TOML file to read. Defaults to ``configs/default.toml`` in the repo.

    Raises:
        ConfigError: If the file is missing, malformed or internally inconsistent.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        with config_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    p = _section(raw, "paths")
    d = _section(raw, "data")
    s = _section(raw, "split")
    c = _section(raw, "columns")

    paths = PathsConfig(
        data_dir=PROJECT_ROOT / _get(p, "data_dir", "paths"),
        raw_filename=_get(p, "raw_filename", "paths"),
        artifacts_dir=PROJECT_ROOT / _get(p, "artifacts_dir", "paths"),
        reports_dir=PROJECT_ROOT / _get(p, "reports_dir", "paths"),
    )
    data = DataConfig(
        dataset_page_url=_get(d, "dataset_page_url", "data"),
        download_url=_get(d, "download_url", "data"),
        expected_sha256=_get(d, "expected_sha256", "data"),
    )
    split = SplitConfig(
        seed=int(_get(s, "seed", "split")),
        train_size=float(_get(s, "train_size", "split")),
        val_size=float(_get(s, "val_size", "split")),
        test_size=float(_get(s, "test_size", "split")),
    )
    columns = ColumnsConfig(
        target=_get(c, "target", "columns"),
        numeric_features=tuple(_get(c, "numeric_features", "columns")),
        categorical_features=tuple(_get(c, "categorical_features", "columns")),
        identifier_columns=tuple(_get(c, "identifier_columns", "columns")),
        leakage_columns=tuple(_get(c, "leakage_columns", "columns")),
        rename=dict(_get(c, "rename", "columns")),
        allowed_categories={k: tuple(v) for k, v in c.get("allowed_categories", {}).items()},
    )
    return Config(paths=paths, data=data, split=split, columns=columns)
