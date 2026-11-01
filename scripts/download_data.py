"""Download (or stage) the UCI AI4I 2020 dataset and validate it.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force
    python scripts/download_data.py --from-file C:\\path\\to\\ai4i2020.zip
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from industrial_ai.config import ConfigError, load_config
from industrial_ai.data.download import DownloadError, ensure_raw_dataset, file_sha256
from industrial_ai.data.loader import DatasetNotFoundError, load_raw_dataset
from industrial_ai.data.schema import SchemaError

logger = logging.getLogger("download_data")


def main(argv: list[str] | None = None) -> int:
    """Run the download-and-validate workflow. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--from-file", type=Path, help="stage a local .zip or .csv instead")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config()
        path = ensure_raw_dataset(config, force=args.force, from_file=args.from_file)
        df = load_raw_dataset(path, config.columns)
    except (ConfigError, DownloadError, SchemaError, DatasetNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    target = config.columns.target
    digest = file_sha256(path)
    print(f"Validated dataset: {path}")
    print(f"  rows: {len(df)}, columns: {df.shape[1]}")
    print(f"  {target} = 1: {int(df[target].sum())} ({df[target].mean():.2%})")
    print(f"  SHA-256: {digest}")
    if not config.data.expected_sha256:
        print("  To pin this exact file, set expected_sha256 in configs/default.toml.")
    print(f"Source and license: {config.data.dataset_page_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
