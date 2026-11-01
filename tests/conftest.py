"""Shared fixtures: a config pointing at a temp dir and a small valid synthetic raw frame."""

import io
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from industrial_ai.config import Config, load_config


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Default config with data/ redirected into a temporary directory."""
    base = load_config()
    return replace(base, paths=replace(base.paths, data_dir=tmp_path / "data"))


def make_raw_frame(n: int = 60) -> pd.DataFrame:
    """Build a valid synthetic frame using the raw UCI column names (not real data)."""
    rng = np.random.default_rng(0)
    types = np.array(["L", "M", "H"])[np.arange(n) % 3]
    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"{t}{i:05d}" for i, t in enumerate(types)],
            "Type": types,
            "Air temperature [K]": rng.normal(300, 2, n),
            "Process temperature [K]": rng.normal(310, 1, n),
            "Rotational speed [rpm]": rng.normal(1500, 100, n),
            "Torque [Nm]": rng.normal(40, 10, n),
            "Tool wear [min]": rng.integers(0, 250, n),
            "Machine failure": (np.arange(n) % 10 == 0).astype(int),
            "TWF": np.zeros(n, dtype=int),
            "HDF": np.zeros(n, dtype=int),
            "PWF": np.zeros(n, dtype=int),
            "OSF": np.zeros(n, dtype=int),
            "RNF": np.zeros(n, dtype=int),
        }
    )


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


def make_zip(files: dict[str, bytes]) -> bytes:
    """Build an in-memory zip archive from name -> content."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buffer.getvalue()
