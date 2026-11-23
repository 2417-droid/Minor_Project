"""
Tests for data loading, preprocessing, and timestamp handling.

Run with::

    pytest tests/test_data.py -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.config import NUMERIC_COLUMNS, RAW_COLUMNS, TARGET
from src.data_preprocessing import (
    check_duplicates,
    convert_dtypes,
    handle_duplicates,
    impute_missing,
    load_hourly_data,
    parse_datetime,
    report_missing,
    resample_hourly,
    run_preprocessing,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_raw_df():
    """Create a small raw DataFrame mimicking the UCI format."""
    data = {
        "Date": ["16/12/2006", "16/12/2006", "16/12/2006", "16/12/2006"],
        "Time": ["17:24:00", "17:25:00", "17:26:00", "17:27:00"],
        "Global_active_power": ["4.216", "5.373", "5.488", "3.810"],
        "Global_reactive_power": ["0.418", "0.436", "0.468", "0.400"],
        "Voltage": ["234.84", "233.63", "233.29", "235.06"],
        "Global_intensity": ["18.4", "23.0", "23.0", "16.0"],
        "Sub_metering_1": ["0", "0", "0", "0"],
        "Sub_metering_2": ["1", "1", "2", "0"],
        "Sub_metering_3": ["17", "16", "17", "16"],
    }
    df = pd.DataFrame(data, columns=RAW_COLUMNS)
    return df


@pytest.fixture
def sample_with_missing():
    """Raw data with missing values encoded as '?'."""
    data = {
        "Date": ["16/12/2006", "16/12/2006", "16/12/2006"],
        "Time": ["17:24:00", "17:25:00", "17:26:00"],
        "Global_active_power": ["4.216", "?", "5.488"],
        "Global_reactive_power": ["0.418", "?", "0.468"],
        "Voltage": ["234.84", "?", "233.29"],
        "Global_intensity": ["18.4", "?", "23.0"],
        "Sub_metering_1": ["0", "?", "0"],
        "Sub_metering_2": ["1", "?", "2"],
        "Sub_metering_3": ["17", "?", "17"],
    }
    df = pd.DataFrame(data, columns=RAW_COLUMNS)
    return df


@pytest.fixture
def sample_with_duplicates():
    """Raw data with duplicate timestamps."""
    data = {
        "Date": ["16/12/2006", "16/12/2006", "16/12/2006"],
        "Time": ["17:24:00", "17:24:00", "17:25:00"],
        "Global_active_power": ["4.216", "4.220", "5.373"],
        "Global_reactive_power": ["0.418", "0.420", "0.436"],
        "Voltage": ["234.84", "235.00", "233.63"],
        "Global_intensity": ["18.4", "18.5", "23.0"],
        "Sub_metering_1": ["0", "0", "0"],
        "Sub_metering_2": ["1", "1", "1"],
        "Sub_metering_3": ["17", "18", "16"],
    }
    df = pd.DataFrame(data, columns=RAW_COLUMNS)
    return df


# ── Datetime parsing tests ────────────────────────────────────────────────────
class TestParseDatetime:
    def test_datetime_parsed_correctly(self, sample_raw_df):
        """Verify that Date + Time are combined into a proper datetime index."""
        df = parse_datetime(sample_raw_df)
        assert df.index.name == "datetime"
        assert df.index[0] == pd.Timestamp("2006-12-16 17:24:00")
        assert df.index[-1] == pd.Timestamp("2006-12-16 17:27:00")

    def test_date_time_columns_dropped(self, sample_raw_df):
        """Date and Time columns should be removed after parsing."""
        df = parse_datetime(sample_raw_df)
        assert "Date" not in df.columns
        assert "Time" not in df.columns

    def test_index_is_monotonic_after_parse(self, sample_raw_df):
        """The parsed index should preserve chronological order."""
        df = parse_datetime(sample_raw_df)
        assert df.index.is_monotonic_increasing

    def test_numeric_columns_preserved(self, sample_raw_df):
        """All 7 numeric columns should remain after datetime parsing."""
        df = parse_datetime(sample_raw_df)
        for col in NUMERIC_COLUMNS:
            assert col in df.columns


# ── Data type conversion tests ────────────────────────────────────────────────
class TestConvertDtypes:
    def test_conversion_to_numeric(self, sample_raw_df):
        """Numeric columns should be numeric (float or int) after conversion."""
        df = parse_datetime(sample_raw_df)
        df = convert_dtypes(df)
        for col in NUMERIC_COLUMNS:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"{col} is {df[col].dtype}, expected numeric"

    def test_missing_values_become_nan(self, sample_with_missing):
        """'?' entries should become NaN after conversion."""
        df = parse_datetime(sample_with_missing)
        df = convert_dtypes(df)
        assert df["Global_active_power"].isna().sum() == 1
        assert df["Voltage"].isna().sum() == 1


# ── Missing value handling tests ──────────────────────────────────────────────
class TestMissingValues:
    def test_report_missing(self, sample_with_missing):
        """report_missing should identify columns with NaN."""
        df = parse_datetime(sample_with_missing)
        df = convert_dtypes(df)
        report = report_missing(df)
        assert len(report) > 0
        assert "Global_active_power" in report.index

    def test_impute_missing_no_nan_left(self, sample_with_missing):
        """After imputation, no NaN should remain."""
        df = parse_datetime(sample_with_missing)
        df = convert_dtypes(df)
        df = impute_missing(df)
        assert df.isna().sum().sum() == 0

    def test_impute_uses_forward_fill(self, sample_with_missing):
        """ffill should carry the last known value forward."""
        df = parse_datetime(sample_with_missing)
        df = convert_dtypes(df)
        df = impute_missing(df)
        # The second row was '?' (NaN), should be filled with the first row's value
        assert df["Global_active_power"].iloc[1] == df["Global_active_power"].iloc[0]


# ── Duplicate handling tests ──────────────────────────────────────────────────
class TestDuplicates:
    def test_duplicate_detection(self, sample_with_duplicates):
        """check_duplicates should detect duplicate timestamps."""
        df = parse_datetime(sample_with_duplicates)
        n_dup = check_duplicates(df)
        assert n_dup == 1  # 17:24:00 appears twice

    def test_duplicate_resolution(self, sample_with_duplicates):
        """handle_duplicates should average duplicate timestamps."""
        df = parse_datetime(sample_with_duplicates)
        df = convert_dtypes(df)  # convert to numeric before grouping
        df = handle_duplicates(df)
        assert df.index.duplicated().sum() == 0
        # Averaged value: (4.216 + 4.220) / 2 = 4.218
        assert abs(df["Global_active_power"].iloc[0] - 4.218) < 0.001


# ── Resampling tests ──────────────────────────────────────────────────────────
class TestResampling:
    def test_resample_hourly_shape(self):
        """Resampling 120 minutes (2 hours) should yield 2 hourly rows."""
        # Create 120 minutes of data (2 full hours)
        idx = pd.date_range("2023-01-01 00:00", periods=120, freq="min")
        df = pd.DataFrame({
            "Global_active_power": np.linspace(1.0, 3.0, 120),
            "Global_reactive_power": np.linspace(0.1, 0.3, 120),
            "Voltage": np.linspace(230, 240, 120),
            "Global_intensity": np.linspace(5, 15, 120),
            "Sub_metering_1": np.ones(120),
            "Sub_metering_2": np.ones(120) * 2,
            "Sub_metering_3": np.ones(120) * 10,
        }, index=idx)
        df.index.name = "datetime"

        hourly = resample_hourly(df)
        assert len(hourly) >= 2
        assert "Sub_metering_1_Wh" in hourly.columns
        assert "Sub_metering_2_Wh" in hourly.columns
        assert "Sub_metering_3_Wh" in hourly.columns

    def test_resample_power_uses_mean(self):
        """Global_active_power should be averaged, not summed."""
        idx = pd.date_range("2023-01-01 00:00", periods=60, freq="min")
        df = pd.DataFrame({
            "Global_active_power": np.full(60, 2.0),
            "Global_reactive_power": np.full(60, 0.2),
            "Voltage": np.full(60, 235.0),
            "Global_intensity": np.full(60, 10.0),
            "Sub_metering_1": np.zeros(60),
            "Sub_metering_2": np.zeros(60),
            "Sub_metering_3": np.zeros(60),
        }, index=idx)
        df.index.name = "datetime"

        hourly = resample_hourly(df)
        # Mean of constant 2.0 should still be 2.0
        assert abs(hourly["Global_active_power"].iloc[0] - 2.0) < 0.001

    def test_resample_submetering_uses_sum(self):
        """Sub-metering should be summed (total Wh in the hour)."""
        idx = pd.date_range("2023-01-01 00:00", periods=60, freq="min")
        df = pd.DataFrame({
            "Global_active_power": np.full(60, 2.0),
            "Global_reactive_power": np.full(60, 0.2),
            "Voltage": np.full(60, 235.0),
            "Global_intensity": np.full(60, 10.0),
            "Sub_metering_1": np.full(60, 1.0),
            "Sub_metering_2": np.full(60, 2.0),
            "Sub_metering_3": np.full(60, 10.0),
        }, index=idx)
        df.index.name = "datetime"

        hourly = resample_hourly(df)
        # Sum of 60 values of 1.0 = 60
        assert abs(hourly["Sub_metering_1_Wh"].iloc[0] - 60.0) < 0.001
        assert abs(hourly["Sub_metering_2_Wh"].iloc[0] - 120.0) < 0.001
        assert abs(hourly["Sub_metering_3_Wh"].iloc[0] - 600.0) < 0.001

    def test_hourly_index_freq(self):
        """Hourly resampled data should have 1-hour spacing."""
        idx = pd.date_range("2023-01-01 00:00", periods=180, freq="min")
        df = pd.DataFrame({
            "Global_active_power": np.ones(180),
            "Global_reactive_power": np.ones(180),
            "Voltage": np.ones(180) * 235,
            "Global_intensity": np.ones(180) * 10,
            "Sub_metering_1": np.ones(180),
            "Sub_metering_2": np.ones(180),
            "Sub_metering_3": np.ones(180),
        }, index=idx)
        df.index.name = "datetime"

        hourly = resample_hourly(df)
        time_diffs = hourly.index.to_series().diff().dropna()
        assert (time_diffs == pd.Timedelta(hours=1)).all()


# ── Full preprocessing pipeline tests ─────────────────────────────────────────
class TestPreprocessingPipeline:
    def test_full_pipeline_no_nan(self):
        """The full preprocessing pipeline should produce no NaN values."""
        # Create 200 minutes of data (enough for at least 3 hours)
        idx = pd.date_range("2023-01-01 00:00", periods=200, freq="min")
        df = pd.DataFrame({
            "Global_active_power": np.linspace(1, 5, 200),
            "Global_reactive_power": np.linspace(0.1, 0.5, 200),
            "Voltage": np.linspace(230, 240, 200),
            "Global_intensity": np.linspace(5, 20, 200),
            "Sub_metering_1": np.linspace(0, 5, 200),
            "Sub_metering_2": np.linspace(0, 5, 200),
            "Sub_metering_3": np.linspace(0, 20, 200),
        }, index=idx)
        df.index.name = "datetime"

        # Impute (no missing) and resample
        df = impute_missing(df)
        hourly = resample_hourly(df)
        hourly = impute_missing(hourly)

        assert hourly.isna().sum().sum() == 0
        assert TARGET in hourly.columns
        assert len(hourly) >= 3
