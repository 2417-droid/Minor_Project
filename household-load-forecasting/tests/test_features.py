"""
Tests for feature engineering and leakage prevention.

Run with::

    pytest tests/test_features.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.config import HORIZONS, LAG_HOURS, ROLLING_WINDOWS, TARGET
from src.feature_engineering import (
    add_exogenous_features,
    add_lag_features,
    add_rolling_features,
    add_targets,
    add_time_features,
    build_features,
    get_feature_columns,
    get_target_columns,
    split_features_targets,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_hourly():
    """Create 300 hours of synthetic hourly data for feature testing."""
    idx = pd.date_range("2023-01-01 00:00", periods=300, freq="h")
    np.random.seed(42)
    df = pd.DataFrame({
        "Global_active_power": np.random.uniform(0.5, 5.0, 300),
        "Global_reactive_power": np.random.uniform(0.0, 0.5, 300),
        "Voltage": np.random.uniform(230, 245, 300),
        "Global_intensity": np.random.uniform(2, 25, 300),
        "Sub_metering_1_Wh": np.random.uniform(0, 10, 300),
        "Sub_metering_2_Wh": np.random.uniform(0, 15, 300),
        "Sub_metering_3_Wh": np.random.uniform(0, 30, 300),
    }, index=idx)
    df.index.name = "datetime"
    return df


# ── Time feature tests ─────────────────────────────────────────────────────────
class TestTimeFeatures:
    def test_hour_column(self, sample_hourly):
        df = add_time_features(sample_hourly)
        assert "hour" in df.columns
        assert df["hour"].min() == 0
        assert df["hour"].max() == 23

    def test_dayofweek_column(self, sample_hourly):
        df = add_time_features(sample_hourly)
        assert "dayofweek" in df.columns
        assert df["dayofweek"].min() == 0
        assert df["dayofweek"].max() == 6

    def test_is_weekend(self, sample_hourly):
        df = add_time_features(sample_hourly)
        assert "is_weekend" in df.columns
        assert set(df["is_weekend"].unique()) == {0, 1}
        # 2023-01-01 is a Sunday, so is_weekend should be 1
        assert df["is_weekend"].iloc[0] == 1

    def test_month_column(self, sample_hourly):
        df = add_time_features(sample_hourly)
        assert "month" in df.columns
        assert df["month"].min() >= 1
        assert df["month"].max() <= 12

    def test_cyclical_encodings(self, sample_hourly):
        """sin/cos encodings should be in [-1, 1]."""
        df = add_time_features(sample_hourly)
        assert "sin_hour" in df.columns
        assert "cos_hour" in df.columns
        assert "sin_dow" in df.columns
        assert "cos_dow" in df.columns
        assert df["sin_hour"].between(-1, 1).all()
        assert df["cos_hour"].between(-1, 1).all()


# ── Lag feature tests ──────────────────────────────────────────────────────────
class TestLagFeatures:
    def test_lag_columns_present(self, sample_hourly):
        df = add_lag_features(sample_hourly)
        for lag in LAG_HOURS:
            assert f"lag_{lag}" in df.columns

    def test_lag_values_correct(self, sample_hourly):
        """lag_1 at row i should equal target at row i-1."""
        df = add_lag_features(sample_hourly, lags=[1])
        # lag_1 at index 1 should equal target at index 0
        assert abs(df["lag_1"].iloc[1] - df[TARGET].iloc[0]) < 1e-10

    def test_lag_24_correct(self, sample_hourly):
        """lag_24 at row i should equal target at row i-24."""
        df = add_lag_features(sample_hourly, lags=[24])
        assert abs(df["lag_24"].iloc[24] - df[TARGET].iloc[0]) < 1e-10

    def test_lag_168_correct(self, sample_hourly):
        """lag_168 (1 week) at row i should equal target at row i-168."""
        df = add_lag_features(sample_hourly, lags=[168])
        assert abs(df["lag_168"].iloc[168] - df[TARGET].iloc[0]) < 1e-10

    def test_lag_first_n_rows_are_nan(self, sample_hourly):
        """The first `lag` rows should be NaN (no earlier data available)."""
        df = add_lag_features(sample_hourly, lags=[3])
        assert df["lag_3"].iloc[:3].isna().all()
        assert df["lag_3"].iloc[3:].notna().all()


# ── Rolling feature tests (leakage prevention) ────────────────────────────────
class TestRollingFeatures:
    def test_rolling_columns_present(self, sample_hourly):
        df = add_rolling_features(sample_hourly)
        for w in ROLLING_WINDOWS:
            assert f"roll_mean_{w}" in df.columns
            assert f"roll_std_{w}" in df.columns

    def test_rolling_excludes_current_observation(self, sample_hourly):
        """
        CRITICAL: Rolling mean at row i must NOT include the value at row i.
        If it did, the model would see the target it is trying to predict.

        We test this by creating a series where one value is an extreme outlier.
        If the rolling window includes the current observation, the mean at
        that row would be affected.  With shift(1), it should NOT be affected.
        """
        # Create data with an outlier at position 10
        values = np.ones(50)
        values[10] = 1000.0  # extreme spike
        idx = pd.date_range("2023-01-01", periods=50, freq="h")
        df = pd.DataFrame({TARGET: values}, index=idx)
        df.index.name = "datetime"

        df = add_rolling_features(df, windows=[3])
        # roll_mean_3 at row 10 = mean(values[7], values[8], values[9])
        #   = mean(1, 1, 1) = 1.0 (NOT including the spike at row 10)
        # If leakage existed: mean(values[8], values[9], values[10])
        #   = mean(1, 1, 1000) = 334.0
        assert abs(df["roll_mean_3"].iloc[10] - 1.0) < 1e-10, \
            "Rolling mean includes current observation - LEAKAGE!"

        # The spike DOES affect roll_mean_3 at row 11 (one step later)
        # roll_mean_3 at row 11 = mean(values[8], values[9], values[10])
        #   = mean(1, 1, 1000) = 334.0
        assert abs(df["roll_mean_3"].iloc[11] - (1 + 1 + 1000) / 3) < 1e-10

    def test_rolling_mean_matches_manual_calculation(self, sample_hourly):
        """Verify roll_mean_3 at row i = mean(target[i-3], target[i-2], target[i-1])."""
        df = add_rolling_features(sample_hourly, windows=[3])
        i = 10
        expected = sample_hourly[TARGET].iloc[i-3:i].mean()
        assert abs(df["roll_mean_3"].iloc[i] - expected) < 1e-10


# ── Exogenous feature tests ───────────────────────────────────────────────────
class TestExogenousFeatures:
    def test_exog_lag_columns(self, sample_hourly):
        df = add_exogenous_features(sample_hourly)
        assert "Global_reactive_power_lag1" in df.columns
        assert "Voltage_lag1" in df.columns
        assert "Global_intensity_lag1" in df.columns
        assert "Sub_metering_1_Wh_lag1" in df.columns
        assert "Sub_metering_2_Wh_lag1" in df.columns
        assert "Sub_metering_3_Wh_lag1" in df.columns

    def test_exog_lag_values(self, sample_hourly):
        """Exogenous lag1 at row i should equal original at row i-1."""
        df = add_exogenous_features(sample_hourly)
        assert abs(df["Voltage_lag1"].iloc[1] - sample_hourly["Voltage"].iloc[0]) < 1e-10


# ── Target (multi-horizon) tests ───────────────────────────────────────────────
class TestTargets:
    def test_target_columns_present(self, sample_hourly):
        df = add_targets(sample_hourly)
        for h in HORIZONS:
            assert f"target_t+{h}" in df.columns

    def test_target_t_plus_1_correct(self, sample_hourly):
        """target_t+1 at row i should equal target at row i+1."""
        df = add_targets(sample_hourly, horizons=[1])
        assert abs(df["target_t+1"].iloc[0] - sample_hourly[TARGET].iloc[1]) < 1e-10

    def test_target_t_plus_6_correct(self, sample_hourly):
        """target_t+6 at row i should equal target at row i+6."""
        df = add_targets(sample_hourly, horizons=[6])
        assert abs(df["target_t+6"].iloc[0] - sample_hourly[TARGET].iloc[6]) < 1e-10

    def test_target_t_plus_24_correct(self, sample_hourly):
        """target_t+24 at row i should equal target at row i+24."""
        df = add_targets(sample_hourly, horizons=[24])
        assert abs(df["target_t+24"].iloc[0] - sample_hourly[TARGET].iloc[24]) < 1e-10

    def test_last_n_targets_are_nan(self, sample_hourly):
        """The last H rows of target_t+H should be NaN (no future data)."""
        df = add_targets(sample_hourly, horizons=[1])
        assert df["target_t+1"].iloc[-1] is np.nan or \
               pd.isna(df["target_t+1"].iloc[-1])

    def test_target_shifts_forward(self, sample_hourly):
        """
        CRITICAL: target_t+H is created by shifting backward, so the model
        learns from past features to predict future consumption.
        The target at row i is the actual value at row i+H (future).
        """
        df = add_targets(sample_hourly, horizons=[3])
        # target_t+3 at row 0 = actual power at row 3
        assert df["target_t+3"].iloc[0] == sample_hourly[TARGET].iloc[3]


# ── Full feature pipeline tests ────────────────────────────────────────────────
class TestBuildFeatures:
    def test_build_features_shape(self, sample_hourly):
        df = build_features(sample_hourly)
        assert df.shape[0] > 0
        assert df.shape[1] > 20  # many features

    def test_build_features_no_nan(self, sample_hourly):
        """After dropna, there should be no NaN values."""
        df = build_features(sample_hourly)
        assert df.isna().sum().sum() == 0

    def test_feature_columns_exclude_target(self, sample_hourly):
        """Feature columns should not include the raw target or target_t+ columns."""
        df = build_features(sample_hourly)
        feat_cols = get_feature_columns(df)
        assert TARGET not in feat_cols
        assert all(not c.startswith("target_t+") for c in feat_cols)

    def test_target_columns_present(self, sample_hourly):
        df = build_features(sample_hourly)
        target_cols = get_target_columns(df)
        for h in HORIZONS:
            assert f"target_t+{h}" in target_cols

    def test_split_features_targets(self, sample_hourly):
        """split_features_targets should return X and y with matching indices."""
        df = build_features(sample_hourly)
        X, y = split_features_targets(df, horizon=1)
        assert len(X) == len(y)
        assert (X.index == y.index).all()
        assert f"target_t+1" not in X.columns


# ── Leakage prevention integration test ──────────────────────────────────────
class TestLeakagePrevention:
    def test_no_future_data_in_features(self, sample_hourly):
        """
        CRITICAL TEST: Ensure no feature at row i contains information from
        any row j > i (future data). This is the fundamental leakage check.

        We verify this by:
        1. Building features
        2. Modifying the target value at a future row
        3. Checking that no feature at the current row changed
        """
        df_original = build_features(sample_hourly, drop_na=True)
        feat_cols = get_feature_columns(df_original)

        # Pick a row in the middle
        i = len(df_original) // 2

        # Record all feature values at row i
        original_features = df_original[feat_cols].iloc[i].copy()

        # Now modify the target at a future row (i + 50)
        modified_hourly = sample_hourly.copy()
        modified_hourly.iloc[i + 50, modified_hourly.columns.get_loc(TARGET)] = 999.0

        # Rebuild features with modified data
        df_modified = build_features(modified_hourly, drop_na=True)
        modified_features = df_modified[feat_cols].iloc[i]

        # Features at row i should NOT be affected by changes at row i+50
        # (which is in the future relative to row i)
        for col in feat_cols:
            # Allow for floating point differences only if both are very small
            diff = abs(original_features[col] - modified_features[col])
            assert diff < 1e-10, (
                f"Feature '{col}' at row {i} changed by {diff} when future "
                f"row {i+50} was modified — possible LEAKAGE!"
            )

    def test_rolling_uses_shift1(self, sample_hourly):
        """
        Verify that rolling features use shift(1) by checking that the
        rolling mean at row i does not depend on the target at row i.
        """
        df = add_rolling_features(sample_hourly, windows=[6])

        # Change the target at row 100
        modified = sample_hourly.copy()
        modified.iloc[100, modified.columns.get_loc(TARGET)] = 999.0

        df_mod = add_rolling_features(modified, windows=[6])

        # roll_mean_6 at row 100 should be the same in both (uses shift(1))
        assert abs(df["roll_mean_6"].iloc[100] - df_mod["roll_mean_6"].iloc[100]) < 1e-10

        # roll_mean_6 at row 101 should be different (now includes the modified value)
        assert abs(df["roll_mean_6"].iloc[101] - df_mod["roll_mean_6"].iloc[101]) > 1e-6
