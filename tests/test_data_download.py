"""Tests for dataset download and staging. No network access is used."""

import hashlib
from dataclasses import replace

import pandas as pd
import pytest

from industrial_ai.config import ConfigError, DataConfig
from industrial_ai.data.download import (
    DownloadError,
    csv_bytes_from_zip,
    ensure_raw_dataset,
    fetch_bytes,
    file_sha256,
)
from tests.conftest import make_zip


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _forbidden_fetch(url: str) -> bytes:
    raise AssertionError("fetch must not be called")


def test_download_extracts_and_writes_csv(cfg, raw_frame) -> None:
    csv = _csv_bytes(raw_frame)
    archive = make_zip({"ai4i2020.csv": csv})
    path = ensure_raw_dataset(cfg, fetch=lambda url: archive)
    assert path == cfg.paths.raw_data_path
    assert path.read_bytes() == csv


def test_existing_file_is_not_redownloaded(cfg, raw_frame) -> None:
    path = cfg.paths.raw_data_path
    path.parent.mkdir(parents=True)
    path.write_bytes(_csv_bytes(raw_frame))
    assert ensure_raw_dataset(cfg, fetch=_forbidden_fetch) == path


def test_force_redownloads(cfg, raw_frame) -> None:
    path = cfg.paths.raw_data_path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"old")
    csv = _csv_bytes(raw_frame)
    ensure_raw_dataset(cfg, force=True, fetch=lambda url: make_zip({"a.csv": csv}))
    assert path.read_bytes() == csv


def test_archive_member_path_cannot_escape(cfg, raw_frame, tmp_path) -> None:
    csv = _csv_bytes(raw_frame)
    archive = make_zip({"../../evil.csv": csv})
    path = ensure_raw_dataset(cfg, fetch=lambda url: archive)
    assert path == cfg.paths.raw_data_path
    assert not (tmp_path.parent / "evil.csv").exists()


def test_multiple_csv_members_rejected() -> None:
    with pytest.raises(DownloadError, match="exactly one CSV"):
        csv_bytes_from_zip(make_zip({"a.csv": b"x", "b.csv": b"y"}))


def test_no_csv_member_rejected() -> None:
    with pytest.raises(DownloadError, match="exactly one CSV"):
        csv_bytes_from_zip(make_zip({"readme.txt": b"hi"}))


def test_non_zip_payload_rejected(cfg) -> None:
    with pytest.raises(DownloadError, match="not a valid zip"):
        ensure_raw_dataset(cfg, fetch=lambda url: b"<html>not a zip</html>")


def test_checksum_mismatch_rejected_and_nothing_written(cfg, raw_frame) -> None:
    pinned = replace(cfg, data=replace(cfg.data, expected_sha256="0" * 64))
    archive = make_zip({"a.csv": _csv_bytes(raw_frame)})
    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        ensure_raw_dataset(pinned, fetch=lambda url: archive)
    assert not pinned.paths.raw_data_path.exists()


def test_matching_checksum_accepted(cfg, raw_frame) -> None:
    csv = _csv_bytes(raw_frame)
    digest = hashlib.sha256(csv).hexdigest()
    pinned = replace(cfg, data=replace(cfg.data, expected_sha256=digest))
    path = ensure_raw_dataset(pinned, fetch=lambda url: make_zip({"a.csv": csv}))
    assert file_sha256(path) == digest


def test_existing_file_with_wrong_checksum_rejected(cfg, raw_frame) -> None:
    pinned = replace(cfg, data=replace(cfg.data, expected_sha256="1" * 64))
    path = pinned.paths.raw_data_path
    path.parent.mkdir(parents=True)
    path.write_bytes(_csv_bytes(raw_frame))
    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        ensure_raw_dataset(pinned, fetch=_forbidden_fetch)


def test_from_file_csv(cfg, raw_frame, tmp_path) -> None:
    source = tmp_path / "manual.csv"
    source.write_bytes(_csv_bytes(raw_frame))
    path = ensure_raw_dataset(cfg, from_file=source, fetch=_forbidden_fetch)
    assert path.read_bytes() == source.read_bytes()


def test_from_file_zip(cfg, raw_frame, tmp_path) -> None:
    csv = _csv_bytes(raw_frame)
    source = tmp_path / "manual.zip"
    source.write_bytes(make_zip({"ai4i2020.csv": csv}))
    path = ensure_raw_dataset(cfg, from_file=source, fetch=_forbidden_fetch)
    assert path.read_bytes() == csv


def test_from_file_missing_rejected(cfg, tmp_path) -> None:
    with pytest.raises(DownloadError, match="File not found"):
        ensure_raw_dataset(cfg, from_file=tmp_path / "nope.zip")


def test_fetch_rejects_non_https_without_network() -> None:
    with pytest.raises(DownloadError, match="non-https"):
        fetch_bytes("http://example.com/data.zip")


def test_config_rejects_non_https_download_url() -> None:
    with pytest.raises(ConfigError, match="https"):
        DataConfig(dataset_page_url="https://x", download_url="http://x/y.zip", expected_sha256="")


def test_config_rejects_malformed_checksum() -> None:
    with pytest.raises(ConfigError, match="64 lowercase hex"):
        DataConfig(
            dataset_page_url="https://x", download_url="https://x/y.zip", expected_sha256="abc"
        )
