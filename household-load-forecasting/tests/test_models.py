"""
Tests for model training, chronological split, prediction shapes, and metrics.

Run with::

    pytest tests/test_models.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

from src.config import HORIZONS, RANDOM_SEED, TARGET, TRAIN_RATIO, VAL_RATIO
from src.feature_engineering import build_features, get_feature_columns, split_features_targets
from src.train import (
    PersistenceModel,
    SeasonalNaiveModel,
    chronological_split,
    create_models,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_features():
    """Create 500 hours of synthetic data and build features."""
    idx = pd.date_range("2023-01-01 00:00", periods=500, freq="h")
    np.random.seed(42)
    # Create a realistic pattern: daily cycle + noise
    hours = idx.hour
    daily_pattern = 2.0 + 1.0 * np.sin(2 * np.pi * hours / 24)
    noise = np.random.normal(0, 0.3, 500)
    target = daily_pattern + noise

    df = pd.DataFrame({
        "Global_active_power": target,
        "Global_reactive_power": np.random.uniform(0.0, 0.5, 500),
        "Voltage": np.random.uniform(230, 245, 500),
        "Global_intensity": target * 5,
        "Sub_metering_1_Wh": np.random.uniform(0, 10, 500),
        "Sub_metering_2_Wh": np.random.uniform(0, 15, 500),
        "Sub_metering_3_Wh": np.random.uniform(0, 30, 500),
    }, index=idx)
    df.index.name = "datetime"
    return build_features(df)


# ── Chronological split tests ─────────────────────────────────────────────────
class TestChronologicalSplit:
    def test_split_sizes(self, sample_features):
        """Train/val/test sizes should match the configured ratios."""
        train, val, test = chronological_split(sample_features)
        total = len(sample_features)
        expected_train = int(total * TRAIN_RATIO)
        expected_val = int(total * VAL_RATIO)
        expected_test = total - expected_train - expected_val

        assert len(train) == expected_train
        assert len(val) == expected_val
        assert len(test) == expected_test
        assert len(train) + len(val) + len(test) == total

    def test_split_is_chronological(self, sample_features):
        """Train data must come before val, which comes before test."""
        train, val, test = chronological_split(sample_features)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_no_overlap_between_splits(self, sample_features):
        """No index should appear in more than one split."""
        train, val, test = chronological_split(sample_features)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_no_shuffling(self, sample_features):
        """The split should preserve temporal order (no shuffling)."""
        train, val, test = chronological_split(sample_features)
        # Train should start at the same point as the full dataset
        assert train.index[0] == sample_features.index[0]
        # Test should end at the same point as the full dataset
        assert test.index[-1] == sample_features.index[-1]


# ── Model factory tests ───────────────────────────────────────────────────────
class TestModelFactory:
    def test_create_models_returns_5(self):
        models = create_models(horizon=1)
        assert len(models) == 5
        assert "persistence" in models
        assert "seasonal_naive" in models
        assert "ridge" in models
        assert "decision_tree" in models
        assert "random_forest" in models

    def test_model_types(self):
        models = create_models(horizon=1)
        assert isinstance(models["ridge"], Ridge)
        assert isinstance(models["decision_tree"], DecisionTreeRegressor)
        assert isinstance(models["random_forest"], RandomForestRegressor)
        assert isinstance(models["persistence"], PersistenceModel)
        assert isinstance(models["seasonal_naive"], SeasonalNaiveModel)

    def test_random_seed_set(self):
        """Models should use the configured random seed for reproducibility."""
        models = create_models(horizon=1)
        assert models["ridge"].random_state == RANDOM_SEED
        assert models["decision_tree"].random_state == RANDOM_SEED
        assert models["random_forest"].random_state == RANDOM_SEED


# ── Model training and prediction shape tests ────────────────────────────────
class TestModelShapes:
    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_ridge_fit_predict(self, sample_features, horizon):
        """Ridge should train and predict with correct shapes."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, horizon)
        X_val, y_val = split_features_targets(val, horizon)

        model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        assert preds.shape == y_val.shape
        assert len(preds) == len(X_val)

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_decision_tree_fit_predict(self, sample_features, horizon):
        """Decision Tree should train and predict with correct shapes."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, horizon)
        X_val, y_val = split_features_targets(val, horizon)

        model = DecisionTreeRegressor(max_depth=10, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        assert preds.shape == y_val.shape

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_random_forest_fit_predict(self, sample_features, horizon):
        """Random Forest should train and predict with correct shapes."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, horizon)
        X_val, y_val = split_features_targets(val, horizon)

        model = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        assert preds.shape == y_val.shape

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_persistence_fit_predict(self, sample_features, horizon):
        """Persistence baseline should produce predictions matching input size."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, horizon)
        X_val, y_val = split_features_targets(val, horizon)

        model = PersistenceModel(horizon=horizon)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        assert len(preds) == len(X_val)

    @pytest.mark.parametrize("horizon", HORIZONS)
    def test_seasonal_naive_fit_predict(self, sample_features, horizon):
        """Seasonal naive baseline should produce predictions matching input size."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, horizon)
        X_val, y_val = split_features_targets(val, horizon)

        model = SeasonalNaiveModel(horizon=horizon)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        assert len(preds) == len(X_val)

    def test_predictions_are_finite(self, sample_features):
        """Model predictions should not contain NaN or Inf."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, 1)
        X_val, y_val = split_features_targets(val, 1)

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        assert np.all(np.isfinite(preds))


# ── Baseline model behavior tests ─────────────────────────────────────────────
class TestBaselineModels:
    def test_persistence_uses_lag1(self, sample_features):
        """Persistence model should predict lag_1 (last observed value)."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, 1)
        X_val, y_val = split_features_targets(val, 1)

        model = PersistenceModel(horizon=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        # Persistence should return lag_1 values
        np.testing.assert_array_almost_equal(preds, X_val["lag_1"].values)

    def test_seasonal_naive_uses_lag168(self, sample_features):
        """Seasonal naive should predict lag_168 (same hour last week)."""
        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, 1)
        X_val, y_val = split_features_targets(val, 1)

        model = SeasonalNaiveModel(horizon=1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        np.testing.assert_array_almost_equal(preds, X_val["lag_168"].values)


# ── Metric calculation tests ──────────────────────────────────────────────────
class TestMetrics:
    def test_mae_perfect_prediction(self):
        """MAE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 0.0

    def test_mae_known_value(self):
        """MAE should match a manual calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        # errors: 0.5, 0.5, 0.5 → MAE = 0.5
        mae = mean_absolute_error(y_true, y_pred)
        assert abs(mae - 0.5) < 1e-10

    def test_rmse_perfect_prediction(self):
        """RMSE should be 0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        assert rmse == 0.0

    def test_rmse_known_value(self):
        """RMSE should match a manual calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        # errors: 1, 1, 1 → MSE = 1 → RMSE = 1
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        assert abs(rmse - 1.0) < 1e-10

    def test_rmse_penalizes_large_errors(self):
        """RMSE should be larger than MAE when there are large errors."""
        y_true = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
        y_pred = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        # One error of 9, rest are 0
        # MAE = 9/5 = 1.8
        # MSE = 81/5 = 16.2 → RMSE = 4.02...
        assert rmse > mae  # RMSE penalizes the large error more

    def test_r2_perfect_prediction(self):
        """R2 should be 1.0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        r2 = r2_score(y_true, y_pred)
        assert abs(r2 - 1.0) < 1e-10

    def test_r2_mean_prediction(self):
        """R2 should be ~0.0 when predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full(5, y_true.mean())
        r2 = r2_score(y_true, y_pred)
        assert abs(r2 - 0.0) < 1e-10

    def test_r2_negative_for_bad_model(self):
        """R2 should be negative when the model is worse than predicting the mean."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        r2 = r2_score(y_true, y_pred)
        assert r2 < 0.0


# ── Model save/load test ────────────────────────────────────────────────────────
class TestModelSaveLoad:
    def test_save_and_load_model(self, sample_features, tmp_path):
        """Save a trained model with joblib and verify it can be loaded."""
        import joblib

        train, val, test = chronological_split(sample_features)
        X_train, y_train = split_features_targets(train, 1)

        model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        # Save
        model_path = tmp_path / "test_model.joblib"
        joblib.dump(model, model_path)
        assert model_path.exists()

        # Load
        loaded_model = joblib.load(model_path)
        X_val, y_val = split_features_targets(val, 1)
        preds_original = model.predict(X_val)
        preds_loaded = loaded_model.predict(X_val)

        np.testing.assert_array_almost_equal(preds_original, preds_loaded)
