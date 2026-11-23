"""
Model training for household electricity load forecasting.

Models implemented
-------------------
1. **Persistence baseline** - forecast = last observed value.
2. **Seasonal naive baseline** - forecast = value from same hour yesterday.
3. **Ridge Regression** - regularised linear model.
4. **Decision Tree Regressor** - single tree with depth control.
5. **Random Forest Regressor** - ensemble of trees.

Split strategy
---------------
Chronological (temporal) split into train / validation / test:
    train 70% | validation 15% | test 15%

The validation set is used for model selection.  The test set is used **once**
for final evaluation after the best model is chosen.

Multi-horizon
-------------
A separate model is trained for each horizon (1 h, 6 h, 24 h) using the
**direct** prediction strategy.  The best model per horizon is selected on
validation data, then evaluated on the test set.

Usage::

    python -m src.train
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from .config import (
    HORIZONS,
    MODELS_DIR,
    MODEL_PARAMS,
    MODEL_PATH,
    MODELS_PATH,
    RANDOM_SEED,
    TARGET,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    ensure_dirs,
)
from .data_preprocessing import load_hourly_data
from .feature_engineering import build_features, get_feature_columns


# ── Chronological split ───────────────────────────────────────────────────────
def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-indexed DataFrame chronologically.

    Returns (train_df, val_df, test_df) in temporal order.
    No shuffling - preserves temporal sequence to prevent leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"[train] Split: train={len(train_df):,} | "
          f"val={len(val_df):,} | test={len(test_df):,}")
    print(f"[train] Train: {train_df.index.min()} -> {train_df.index.max()}")
    print(f"[train] Val:   {val_df.index.min()} -> {val_df.index.max()}")
    print(f"[train] Test:  {test_df.index.min()} -> {test_df.index.max()}")

    return train_df, val_df, test_df


# ── Baseline models ───────────────────────────────────────────────────────────
class PersistenceModel:
    """
    Persistence baseline: forecast = last observed value.

    For horizon H, the forecast at time t is target(t), i.e. the current
    consumption.  This is the simplest possible forecast.
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.last_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceModel":
        self.last_value = y.iloc[-1] if len(y) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # For persistence: predict the current target value (lag_1)
        # lag_1 in X is the target value 1 hour ago, which is the most
        # recent observation available at prediction time.
        if "lag_1" in X.columns:
            return X["lag_1"].values
        # Fallback: constant
        return np.full(len(X), self.last_value)


class SeasonalNaiveModel:
    """
    Seasonal naive baseline: forecast = value from same hour last week
    (168 hours ago) if available, otherwise yesterday same hour (24 h ago).

    For horizon H: at time t, forecast = target(t - 168 + H)
    Simplified: use lag_168 as the prediction (the value from the same hour
    one week ago).
    """

    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.fallback_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveModel":
        self.fallback_value = y.mean() if len(y) > 0 else 0.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "lag_168" in X.columns:
            preds = X["lag_168"].values
            # Replace NaN with fallback
            preds = np.where(np.isnan(preds), self.fallback_value, preds)
            return preds
        # Fallback: use lag_24 (yesterday same hour)
        if "lag_24" in X.columns:
            preds = X["lag_24"].values
            preds = np.where(np.isnan(preds), self.fallback_value, preds)
            return preds
        return np.full(len(X), self.fallback_value)


# ── Model factory ─────────────────────────────────────────────────────────────
def create_models(horizon: int) -> dict:
    """Create all model instances for a given horizon."""
    models = {
        "persistence": PersistenceModel(horizon=horizon),
        "seasonal_naive": SeasonalNaiveModel(horizon=horizon),
        "ridge": Ridge(**MODEL_PARAMS["ridge"]),
        "decision_tree": DecisionTreeRegressor(**MODEL_PARAMS["decision_tree"]),
        "random_forest": RandomForestRegressor(**MODEL_PARAMS["random_forest"]),
    }
    return models


# ── Training ──────────────────────────────────────────────────────────────────
def train_all_models(
    features_df: pd.DataFrame | None = None,
    save: bool = True,
) -> dict:
    """
    Train all models for all horizons.

    Returns a nested dict: results[horizon][model_name] = {
        "model": fitted_model,
        "val_predictions": np.ndarray,
        "val_actuals": np.ndarray,
        "test_predictions": np.ndarray,
        "test_actuals": np.ndarray,
        "feature_columns": list[str],
    }
    """
    ensure_dirs()

    # Load / build features
    if features_df is None:
        print("[train] Loading hourly data ...")
        hourly = load_hourly_data()
        print("[train] Building features ...")
        features_df = build_features(hourly)

    # Chronological split
    print("[train] Chronological split ...")
    train_df, val_df, test_df = chronological_split(features_df)

    feature_cols = get_feature_columns(train_df)

    all_results = {}

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"[train] Training models for horizon t+{horizon}h")
        print(f"{'='*60}")

        target_col = f"target_t+{horizon}"

        # Prepare data
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        models = create_models(horizon)
        horizon_results = {}

        for name, model in models.items():
            print(f"\n  [{name}] Training ...")
            model.fit(X_train, y_train)

            # Validation predictions
            val_preds = model.predict(X_val)
            # Test predictions (only for the best model later, but we
            # predict for all to allow comparison)
            test_preds = model.predict(X_test)

            horizon_results[name] = {
                "model": model,
                "val_predictions": val_preds,
                "val_actuals": y_val.values,
                "test_predictions": test_preds,
                "test_actuals": y_test.values,
                "feature_columns": feature_cols,
            }
            print(f"  [{name}] Val predictions: {len(val_preds)} samples")

        all_results[horizon] = horizon_results

    if save:
        _save_models(all_results, feature_cols)

    return all_results


def _save_models(results: dict, feature_cols: list[str]) -> None:
    """Save all models and metadata to disk."""
    # Save all models in a single dict
    all_models = {}
    for horizon, horizon_results in results.items():
        for name, res in horizon_results.items():
            all_models[f"h{horizon}_{name}"] = res["model"]

    joblib.dump(all_models, MODELS_PATH)
    print(f"\n[train] All models saved to {MODELS_PATH}")

    # Save feature columns for later use in prediction
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")
    print(f"[train] Feature columns saved to {MODELS_DIR / 'feature_columns.joblib'}")


def select_best_model(results: dict) -> dict:
    """
    Select the best model per horizon based on validation MAE.

    Returns a dict: best[horizon] = {
        "name": str,
        "model": fitted_model,
        "val_mae": float,
    }
    """
    best = {}
    for horizon, horizon_results in results.items():
        best_name = None
        best_mae = float("inf")
        for name, res in horizon_results.items():
            mae = np.mean(np.abs(
                res["val_predictions"] - res["val_actuals"]
            ))
            print(f"  [select] h+{horizon} {name}: val_MAE={mae:.4f}")
            if mae < best_mae:
                best_mae = mae
                best_name = name

        best[horizon] = {
            "name": best_name,
            "model": horizon_results[best_name]["model"],
            "val_mae": best_mae,
        }
        print(f"  [select] Best for h+{horizon}: {best_name} (val_MAE={best_mae:.4f})")

    # Save best models
    best_models = {h: v["model"] for h, v in best.items()}
    best_names = {h: v["name"] for h, v in best.items()}
    joblib.dump(best_models, MODEL_PATH)
    joblib.dump(best_names, MODELS_DIR / "best_model_names.joblib")
    print(f"\n[train] Best models saved to {MODEL_PATH}")
    print(f"[train] Best model names: {best_names}")

    return best


if __name__ == "__main__":
    results = train_all_models()
    best = select_best_model(results)
    print("\n" + "=" * 60)
    print("Training complete.")
    for h, info in best.items():
        print(f"  Horizon t+{h}h: {info['name']} (val MAE={info['val_mae']:.4f})")
