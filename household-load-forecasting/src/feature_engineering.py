"""
Feature engineering for household electricity load forecasting.

Design principles (leakage prevention)
--------------------------------------
1. **Lag features** use ``shift(h)`` — the value from *h* hours ago.  For a
   forecast at time *t* for horizon *H*, we only use lags ≥ *H* for the target
   itself, so the model never sees the value it is trying to predict.

2. **Rolling features** use ``shift(1).rolling(window)`` — we first shift by 1
   so the window ends at the *previous* timestamp.  This guarantees the
   current observation is never included in its own rolling statistic.

3. **Time features** (hour, day-of-week, etc.) are deterministic functions of
   the timestamp and are always known at prediction time.

4. **Multi-horizon targets** are created by shifting the target forward.
   ``target_t+H`` = ``target.shift(-H)``.  Rows where the shifted target is
   NaN (end of dataset) are dropped.

Forecasting strategy
~~~~~~~~~~~~~~~~~~~~~
We use **direct multi-horizon prediction**: a separate model is trained for
each horizon (1 h, 6 h, 24 h).  This avoids error accumulation that would occur
with recursive prediction and is simpler to implement correctly without
leakage.

Usage::

    python -m src.feature_engineering
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    FEATURE_CSV,
    HORIZONS,
    LAG_HOURS,
    ROLLING_WINDOWS,
    TARGET,
    ensure_dirs,
)

FEATURE_PREFIX = "lag_"
ROLLING_MEAN_PREFIX = "roll_mean_"
ROLLING_STD_PREFIX = "roll_std_"


# ── Calendar / time features ──────────────────────────────────────────────────
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features derived from the datetime index.

    Features:
    - hour: 0-23
    - dayofweek: 0=Monday ... 6=Sunday
    - is_weekend: 1 if Saturday/Sunday, else 0
    - month: 1-12
    - sin_hour, cos_hour: cyclical encoding of hour (captures continuity
      between 23:00 and 00:00)
    - sin_dow, cos_dow: cyclical encoding of day-of-week
    """
    out = df.copy()
    idx = out.index

    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["month"] = idx.month

    # Cyclical encodings - helps linear models understand that hour 23 is
    # close to hour 0, not far away.
    out["sin_hour"] = np.sin(2 * np.pi * idx.hour / 24)
    out["cos_hour"] = np.cos(2 * np.pi * idx.hour / 24)
    out["sin_dow"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["cos_dow"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    return out


# ── Lag features ──────────────────────────────────────────────────────────────
def add_lag_features(
    df: pd.DataFrame,
    target: str = TARGET,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add lagged values of the target variable.

    ``lag_H`` = target value *H* hours ago.  These are always known at
    prediction time because they are in the past.
    """
    if lags is None:
        lags = LAG_HOURS

    out = df.copy()
    for lag in lags:
        out[f"{FEATURE_PREFIX}{lag}"] = out[target].shift(lag)
    return out


# ── Rolling features ──────────────────────────────────────────────────────────
def add_rolling_features(
    df: pd.DataFrame,
    target: str = TARGET,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean and std of the target.

    **Leakage prevention**: We use ``shift(1).rolling(window)`` so that the
    current observation is excluded from its own rolling statistic.  The
    window spans ``[t - window, t - 1]`` (inclusive), never including *t*.
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    out = df.copy()
    shifted = out[target].shift(1)
    for w in windows:
        out[f"{ROLLING_MEAN_PREFIX}{w}"] = shifted.rolling(w).mean()
        out[f"{ROLLING_STD_PREFIX}{w}"] = shifted.rolling(w).std()
    return out


# ── Additional feature columns ───────────────────────────────────────────────
def add_exogenous_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged versions of exogenous variables (reactive power, voltage,
    intensity, sub-metering) that are known at prediction time.

    For forecasting at time *t*, only past values are available, so we lag
    by 1 hour minimum.
    """
    out = df.copy()
    exog_cols = [
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1_Wh",
        "Sub_metering_2_Wh",
        "Sub_metering_3_Wh",
    ]
    for col in exog_cols:
        if col in out.columns:
            out[f"{col}_lag1"] = out[col].shift(1)
    return out


# ── Multi-horizon targets ─────────────────────────────────────────────────────
def add_targets(df: pd.DataFrame, target: str = TARGET,
                horizons: list[int] | None = None) -> pd.DataFrame:
    """
    Add shifted target columns for each forecast horizon.

    ``target_t+H`` = target value *H* hours into the future.  This is what
    the model learns to predict.  Rows where any future target is NaN (at
    the end of the dataset) will be dropped later.
    """
    if horizons is None:
        horizons = HORIZONS

    out = df.copy()
    for h in horizons:
        out[f"target_t+{h}"] = out[target].shift(-h)
    return out


# ── Full pipeline ─────────────────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    target: str = TARGET,
    horizons: list[int] | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly resampled data (output of preprocessing).
    target : str
        Name of the primary target column.
    horizons : list[int]
        Forecast horizons in hours.
    lags : list[int]
        Lag hours to include.
    windows : list[int]
        Rolling window sizes.
    drop_na : bool
        Drop rows with NaN features/targets (caused by lags / shifts).

    Returns
    -------
    pd.DataFrame
        Feature matrix with target columns appended.
    """
    out = df.copy()

    # Time features (always safe - no leakage possible)
    out = add_time_features(out)

    # Lag features
    out = add_lag_features(out, target=target, lags=lags)

    # Rolling features (shift(1) prevents leakage)
    out = add_rolling_features(out, target=target, windows=windows)

    # Lagged exogenous features
    out = add_exogenous_features(out)

    # Multi-horizon targets
    out = add_targets(out, target=target, horizons=horizons)

    if drop_na:
        n_before = len(out)
        out = out.dropna()
        n_after = len(out)
        print(f"[features] Dropped {n_before - n_after:,} rows with NaN "
              f"(lags / rolling / future targets). {n_after:,} rows remain.")

    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excluding targets and raw target)."""
    target_cols = {c for c in df.columns if c.startswith("target_t+")}
    # Exclude the raw target column and all target_t+ columns
    exclude = target_cols | {TARGET}
    feature_cols = [c for c in df.columns if c not in exclude]
    return feature_cols


def get_target_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of target column names."""
    return [c for c in df.columns if c.startswith("target_t+")]


def split_features_targets(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the feature DataFrame into X (features) and y (target) for a
    given horizon.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_features``.
    horizon : int
        Forecast horizon (must be in HORIZONS).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target for this horizon.
    """
    target_col = f"target_t+{horizon}"
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found. "
                         f"Available: {get_target_columns(df)}")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


if __name__ == "__main__":
    from .data_preprocessing import load_hourly_data

    ensure_dirs()

    print("[features] Loading hourly data ...")
    hourly = load_hourly_data()

    print("[features] Building features ...")
    features = build_features(hourly)

    print(f"[features] Shape: {features.shape}")
    print(f"[features] Feature columns: {get_feature_columns(features)}")
    print(f"[features] Target columns: {get_target_columns(features)}")

    features.to_csv(FEATURE_CSV)
    print(f"[features] Saved to {FEATURE_CSV}")
