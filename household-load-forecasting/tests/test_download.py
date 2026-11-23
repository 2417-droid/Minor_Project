"""
Tests for data download module.

Run with::

    pytest tests/test_download.py -v
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import RAW_CSV, ZIP_PATH
from src.data_download import download_dataset, verify_dataset


class TestVerifyDataset:
    def test_verify_returns_false_when_missing(self, tmp_path):
        """verify_dataset should return False when the file doesn't exist."""
        # Use a non-existent path
        with patch("src.data_download.RAW_CSV", tmp_path / "nonexistent.txt"):
            assert verify_dataset() is False

    def test_verify_returns_false_when_too_small(self, tmp_path):
        """verify_dataset should return False when the file is too small."""
        small_file = tmp_path / "small.txt"
        small_file.write_text("tiny")
        with patch("src.data_download.RAW_CSV", small_file):
            assert verify_dataset() is False


class TestDownloadDataset:
    def test_download_returns_existing_file(self, tmp_path):
        """If the raw file already exists, download_dataset should return it."""
        # Create a fake raw file
        fake_raw = tmp_path / "household_power_consumption.txt"
        fake_raw.write_text("x" * 60_000_000)  # > MIN_TXT_BYTES

        with patch("src.data_download.RAW_CSV", fake_raw):
            with patch("src.data_download.ZIP_PATH", tmp_path / "fake.zip"):
                result = download_dataset()
                assert result == fake_raw

    def test_download_handles_missing_zip(self, tmp_path):
        """download_dataset should handle missing ZIP gracefully."""
        with patch("src.data_download.RAW_CSV", tmp_path / "missing.txt"):
            with patch("src.data_download.ZIP_PATH", tmp_path / "missing.zip"):
                with patch("src.data_download._download_zip",
                          side_effect=Exception("Network error")):
                    with pytest.raises(Exception, match="Network error"):
                        download_dataset()
