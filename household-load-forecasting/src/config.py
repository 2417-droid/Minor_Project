"""
Central configuration for the household load forecasting project.

All paths, constants, and hyperparameters live here so that every module
shares the same definitions.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
# Project root = parent of this src/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset
ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
ZIP_PATH = RAW_DIR / "household_power_consumption.zip"
RAW_CSV = RAW_DIR / "household_power_consumption.txt"
PROCESSED_CSV = PROCESSED_DIR / "hourly_consumption.csv"
FEATURE_CSV = PROCESSED_DIR / "features.csv"

# Model artifacts
MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODELS_PATH = MODELS_DIR / "all_models.joblib"  # dict of all models
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.csv"

# ── Column names ─────────────────────────────────────────────────────────────
# Raw dataset columns (UCI schema)
RAW_COLUMNS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

TARGET = "Global_active_power"  # kW, the primary forecasting target

# ── Forecasting horizons (hours) ─────────────────────────────────────────────
HORIZONS = [1, 6, 24]

# ── Resampling ───────────────────────────────────────────────────────────────
RESAMPLE_FREQ = "h"  # hourly

# ── Feature engineering ──────────────────────────────────────────────────────
LAG_HOURS = [1, 2, 3, 6, 12, 24, 168]  # 168 = 1 week
ROLLING_WINDOWS = [3, 6, 12, 24]

# ── Chronological split ratios ───────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Model hyperparameters ───────────────────────────────────────────────────
MODEL_PARAMS = {
    "ridge": {"alpha": 1.0, "random_state": RANDOM_SEED},
    "decision_tree": {
        "max_depth": 15,
        "min_samples_leaf": 10,
        "random_state": RANDOM_SEED,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
}


def ensure_dirs() -> None:
    """Create all project directories if they do not exist."""
    for d in [
        DATA_DIR,
        RAW_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
