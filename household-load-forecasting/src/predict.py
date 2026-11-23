"""
Inference module for household electricity load forecasting.

Loads the saved best models and feature columns, constructs a feature row
from recent historical data, and produces forecasts for the selected horizon.

Usage (programmatic)::

    from src.predict import load_models, forecast
    models, feature_cols = load_models()
    forecast_df = forecast(models, feature_cols, hourly_data, horizon=1)

Usage (CLI)::

    python -m src.predict --horizon 1
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from .config import (
    HORIZONS,
    MODEL_PATH,
    MODELS_DIR,
    MODELS_PATH,
    TARGET,
)
from .data_preprocessing import load_hourly_data
from .feature_engineering import build_features, get_feature_columns


def load_models() -> tuple[dict, list[str]]:
    """
    Load the saved best models and feature column list.

    Returns
    -------
    models : dict
        {horizon: fitted_model}
    feature_cols : list[str]
        Ordered list of feature column names.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python -m src.train` first."
        )
    models = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    return models, feature_cols


def load_all_models() -> tuple[dict, list[str]]:
    """Load all saved models (not just best) for comparison."""
    if not MODELS_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODELS_PATH}. "
            "Run `python -m src.train` first."
        )
    all_models = joblib.load(MODELS_PATH)
    feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    return all_models, feature_cols


def prepare_recent_features(
    hourly_data: pd.DataFrame,
    feature_cols: list[str],
    n_recent: int = 500,
) -> pd.DataFrame:
    """
    Build features from recent hourly data and return the most recent
    feature row(s) ready for prediction.

    Parameters
    ----------
    hourly_data : pd.DataFrame
        Hourly resampled consumption data.
    feature_cols : list[str]
        Feature column names the model expects.
    n_recent : int
        Number of recent hours to use for building lag/rolling features.

    Returns
    -------
    pd.DataFrame
        Feature rows (most recent at the end) with exactly ``feature_cols``.
    """
    # Build full features from recent data
    features = build_features(hourly_data, drop_na=True)

    # Select only the feature columns the model was trained on
    available_cols = [c for c in feature_cols if c in features.columns]
    X = features[available_cols].copy()

    # If some feature columns are missing, fill with 0 (shouldn't happen
    # if preprocessing is consistent, but guards against edge cases)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    X = X[feature_cols]  # enforce order

    # Return the last n_recent rows (most recent at the end)
    return X.tail(n_recent)


def forecast(
    models: dict,
    feature_cols: list[str],
    hourly_data: pd.DataFrame,
    horizon: int = 1,
    n_recent: int = 168,
) -> pd.DataFrame:
    """
    Generate a forecast for the selected horizon.

    Parameters
    ----------
    models : dict
        Loaded best models: {horizon: model}
    feature_cols : list[str]
        Feature column names.
    hourly_data : pd.DataFrame
        Recent hourly consumption data.
    horizon : int
        Forecast horizon in hours (1, 6, or 24).
    n_recent : int
        Number of recent hours to include in the output.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datetime, actual, predicted.
        The last row's predicted value is the forecast for ``horizon``
        hours ahead.
    """
    if horizon not in models:
        raise ValueError(f"No model found for horizon {horizon}. "
                         f"Available: {list(models.keys())}")

    # Build features from the full hourly data
    X = prepare_recent_features(hourly_data, feature_cols, n_recent=n_recent + 200)

    model = models[horizon]
    predictions = model.predict(X)

    # Build features to get the correct actual target (target_t+H)
    # The model was trained to predict target_t+H = Global_active_power at t+H
    # So "actual" must also be target_t+H, not Global_active_power at t
    features = build_features(hourly_data, drop_na=True)
    target_col = f"target_t+{horizon}"
    dt_index = features.index[-len(X):]

    # Build output DataFrame — actual = what really happened H hours later
    result = pd.DataFrame({
        "datetime": dt_index[-n_recent:],
        "actual": features[target_col].values[-len(X):][-n_recent:],
        "predicted": predictions[-n_recent:],
    })

    return result


def forecast_all_models(
    all_models: dict,
    feature_cols: list[str],
    hourly_data: pd.DataFrame,
    horizon: int = 1,
    n_recent: int = 168,
) -> dict:
    """
    Generate forecasts from ALL models for comparison.

    Returns
    -------
    dict
        {model_name: pd.DataFrame with datetime, actual, predicted}
    """
    # Build features from the full hourly data
    features = build_features(hourly_data, drop_na=True)
    available_cols = [c for c in feature_cols if c in features.columns]
    X = features[available_cols].copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols]
    X_recent = X.tail(n_recent)
    dt_index = features.index[-len(X):][-n_recent:]
    # Use target_t+H as actual (matches what each model was trained to predict)
    target_col = f"target_t+{horizon}"
    actuals = features[target_col].values[-len(X):][-n_recent:]

    results = {}
    model_key = f"h{horizon}_"
    for key, model in all_models.items():
        if key.startswith(model_key):
            name = key.replace(model_key, "")
            preds = model.predict(X_recent)
            results[name] = pd.DataFrame({
                "datetime": dt_index,
                "actual": actuals,
                "predicted": preds,
            })

    return results


def get_latest_forecast(hourly_data: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Convenience function: load models and produce a single forecast value
    for ``horizon`` hours from the most recent data point.

    Returns
    -------
    dict
        {"horizon": int, "forecast_kW": float, "datetime": Timestamp,
         "target_datetime": Timestamp}
    """
    models, feature_cols = load_models()
    X = prepare_recent_features(hourly_data, feature_cols, n_recent=200)
    latest_features = X.iloc[[-1]]  # most recent row

    if horizon not in models:
        raise ValueError(f"No model for horizon {horizon}")

    pred = models[horizon].predict(latest_features)[0]
    latest_dt = hourly_data.index[-1]
    target_dt = latest_dt + pd.Timedelta(hours=horizon)

    return {
        "horizon": horizon,
        "forecast_kW": float(pred),
        "datetime": latest_dt,
        "target_datetime": target_dt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate load forecasts")
    parser.add_argument("--horizon", type=int, default=1, choices=HORIZONS,
                        help="Forecast horizon in hours")
    parser.add_argument("--n-recent", type=int, default=168,
                        help="Number of recent hours to display")
    args = parser.parse_args()

    try:
        print(f"[predict] Loading models ...")
        models, feature_cols = load_models()
        print(f"[predict] Loading hourly data ...")
        hourly = load_hourly_data()
        print(f"[predict] Generating forecast for t+{args.horizon}h ...")
        result = forecast(models, feature_cols, hourly, horizon=args.horizon,
                          n_recent=args.n_recent)
        print(result.tail(10))

        # Single point forecast
        latest = get_latest_forecast(hourly, horizon=args.horizon)
        print(f"\n[predict] Latest forecast:")
        print(f"  Forecast for: {latest['target_datetime']}")
        print(f"  Predicted power: {latest['forecast_kW']:.4f} kW")
    except FileNotFoundError as e:
        print(f"[predict] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
