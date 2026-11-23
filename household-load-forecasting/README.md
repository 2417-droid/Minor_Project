# Household Short-Term Electricity Load Forecasting Using Classical Machine Learning

A complete, reproducible machine learning project that forecasts an individual
household's electricity consumption using historical smart-meter data and
classical ML algorithms.

---

## Problem Statement and Motivation

Grid operators and utility providers rely on aggregate regional demand
forecasts that cannot capture individual household consumption patterns.
As rooftop solar, electric vehicles, and distributed energy resources grow,
accurate household-level forecasts become increasingly valuable.

This project builds a machine learning system that forecasts a single
household's electricity consumption at three horizons:

- **Next-hour** (t+1h)
- **Next-6-hour** (t+6h)
- **Next-24-hour** (t+24h)

using only historical power measurements—no additional hardware required.

---

## Dataset

**Source:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

| Attribute | Value |
|-----------|-------|
| Location | Sceaux, France (7 km south of Paris) |
| Period | December 2006 – November 2010 (47 months) |
| Original sampling | 1-minute intervals |
| Original rows | 2,075,259 |
| Features | 7 numeric (power, voltage, current, sub-metering) |
| Missing values | ~1.25% (encoded as `?`) |
| License | CC BY 4.0 |

### Data Dictionary

| Column | Description | Unit | Aggregation to Hourly |
|--------|-------------|------|----------------------|
| Global_active_power | Household active power (forecasting target) | kW | Mean |
| Global_reactive_power | Reactive power | kW | Mean |
| Voltage | Voltage | V | Mean |
| Global_intensity | Current intensity | A | Mean |
| Sub_metering_1 | Kitchen (dishwasher, oven, microwave) | Wh/min | Sum |
| Sub_metering_2 | Laundry (washer, dryer, fridge, light) | Wh/min | Sum |
| Sub_metering_3 | Water heater & air conditioner | Wh/min | Sum |

**Why mean for power?** Global_active_power is minute-averaged power in kW.
The hourly mean of minute-averaged kW equals hourly energy in kWh, because
each minute contributes `power_kW * (1/60) h` and there are 60 minutes per hour.

See `reports/dataset_report.md` for the full dataset analysis.

---

## System Architecture and Workflow

```
data/raw/              data/processed/           models/              reports/
household_power_... --> hourly_consumption --> features --> train --> evaluate
     (2M rows)           (34.5K rows)        (34.4K x 35)  (15 models) (metrics + plots)
       |                     |                   |              |           |
    download              preprocess          feature_eng      train      evaluate
```

### Pipeline Steps

1. **Download** (`src/data_download.py`): Fetches the ZIP from UCI, extracts the
   raw text file.
2. **Preprocess** (`src/data_preprocessing.py`): Parses timestamps, handles
   missing values, removes duplicates, resamples to hourly.
3. **Feature Engineering** (`src/feature_engineering.py`): Creates lag, rolling,
   time, and exogenous features with leakage prevention.
4. **Train** (`src/train.py`): Trains 5 models x 3 horizons with chronological
   split.
5. **Evaluate** (`src/evaluate.py`): Computes metrics, generates plots, selects
   best model, saves report.
6. **EDA** (`src/eda.py`): Generates exploratory analysis visualizations.
7. **Predict** (`src/predict.py`): Loads saved models for inference.
8. **App** (`app.py`): Streamlit web interface for interactive forecasting.

---

## Feature Engineering

### Features Used (35 total)

**Lag features** (7): `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`, `lag_24`,
`lag_168` (1 week). These are past values of Global_active_power.

**Rolling statistics** (8): `roll_mean_3`, `roll_std_3`, `roll_mean_6`,
`roll_std_6`, `roll_mean_12`, `roll_std_12`, `roll_mean_24`, `roll_std_24`.
Computed with `shift(1).rolling(window)` to prevent leakage.

**Time features** (8): `hour`, `dayofweek`, `is_weekend`, `month`, `sin_hour`,
`cos_hour`, `sin_dow`, `cos_dow`. Cyclical encodings help linear models
understand that hour 23 is close to hour 0.

**Exogenous lagged features** (6): Lagged reactive power, voltage, current,
and sub-metering values (lag 1 hour).

**Original exogenous** (6): Global_reactive_power, Voltage, Global_intensity,
Sub_metering_1/2/3_Wh.

### Leakage Prevention

This is the most critical design principle:

1. **Lag features** use `shift(h)` — only past values.
2. **Rolling features** use `shift(1).rolling(window)` — the current
   observation is excluded from its own rolling statistic. The window spans
   `[t - window, t - 1]`, never including `t`.
3. **Targets** use `shift(-H)` — the model predicts `H` hours ahead.
4. **Chronological split** — no random shuffling, train comes before
   validation which comes before test.
5. **Tests verify this** — `test_no_future_data_in_features` modifies a future
   value and checks that no feature at the current row changes.

### Forecasting Strategy

**Direct multi-horizon prediction**: A separate model is trained for each
horizon. This avoids error accumulation from recursive prediction and is
simpler to implement correctly without leakage.

---

## Models

| # | Model | Description |
|---|-------|-------------|
| 1 | Persistence | Forecast = last observed value (lag_1) |
| 2 | Seasonal Naive | Forecast = value from same hour last week (lag_168) |
| 3 | Ridge Regression | Linear model with L2 regularization |
| 4 | Decision Tree | Single tree, max_depth=15, min_samples_leaf=10 |
| 5 | Random Forest | 100 trees, max_depth=20, min_samples_leaf=5 |

### Model Selection

Models are compared on the **validation set** using MAE. The best model per
horizon is selected, then evaluated **once** on the held-out test set. This
prevents overfitting to the test set.

---

## Evaluation Results

### Test Set Metrics (Best Model: Random Forest)

| Horizon | Model | MAE (kW) | RMSE (kW) | R² |
|---------|-------|----------|-----------|-----|
| t+1h | Random Forest | 0.305 | 0.446 | 0.598 |
| t+6h | Random Forest | 0.413 | 0.569 | 0.345 |
| t+24h | Random Forest | 0.426 | 0.581 | 0.317 |

### Key Findings

- **Random Forest is the best model** for all three horizons.
- The **persistence and seasonal naive baselines have negative R²** on the
  test set, meaning they are worse than predicting the mean. This highlights
  the value of ML models.
- **MAE increases with horizon**: 0.305 kW (1h) to 0.426 kW (24h). Longer
  horizons are harder to forecast, as expected.
- **Ridge Regression** is a strong runner-up, especially for t+24h, showing
  that even a simple linear model with good features outperforms baselines.

### Metric Definitions (Beginner-Friendly)

- **MAE (Mean Absolute Error)**: On average, how far off is the prediction?
  If MAE = 0.3 kW, the forecast is typically within 0.3 kW of the actual value.

- **RMSE (Root Mean Square Error)**: Similar to MAE but penalizes large errors
  more heavily. Always >= MAE. Useful when large errors are particularly costly.

- **R² (R-squared)**: What fraction of the variation in consumption does the
  model explain? 1.0 = perfect, 0.0 = no better than always predicting the
  average, negative = worse than predicting the average.

---

## Installation and Execution

### Prerequisites

- Python 3.10 or higher
- ~500 MB free disk space (for data and models)
- Windows 10/11 (or any OS with Python)

### Step 1: Create Virtual Environment

```powershell
cd household-load-forecasting
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Download Dataset

```powershell
python -m src.data_download
```

This downloads the ~20 MB ZIP from UCI and extracts a ~127 MB text file to
`data/raw/`.

### Step 4: Run Preprocessing

```powershell
python -m src.data_preprocessing
```

Parses 2 million minute-level readings, handles missing values, and resamples
to 34,589 hourly rows. Output: `data/processed/hourly_consumption.csv`.

### Step 5: Run EDA (Optional)

```powershell
python -m src.eda
```

Generates exploratory analysis plots in `reports/figures/`.

### Step 6: Train Models

```powershell
python -m src.train
```

Trains 5 models x 3 horizons = 15 models. Saves to `models/`. Takes ~2-5
minutes on a typical laptop.

### Step 7: Evaluate Models

```powershell
python -m src.evaluate
```

Computes metrics, generates 50+ plots, saves `reports/metrics.csv` and
`reports/model_evaluation.md`.

### Step 8: Run Tests

```powershell
python -m pytest tests/ -v
```

Runs 82 tests covering all pipeline components.

### Step 9: Launch Streamlit App

```powershell
streamlit run app.py
```

Opens an interactive web app at `http://localhost:8501`.

### Step 10: Launch Jupyter Notebook

```powershell
pip install jupyter
jupyter notebook notebooks/
```

### Full Pipeline (One Command)

```powershell
python -m src.data_download; python -m src.data_preprocessing; python -m src.train; python -m src.evaluate; python -m src.eda
```

---

## Project Structure

```
household-load-forecasting/
  data/
    raw/                      # Downloaded dataset
    processed/                # Hourly CSV, features CSV
  notebooks/
    01_data_exploration.ipynb # EDA notebook
    02_model_training.ipynb   # Training notebook
  src/
    config.py                 # Central configuration
    data_download.py          # Dataset download
    data_preprocessing.py      # Parsing, cleaning, resampling
    feature_engineering.py    # Lag, rolling, time features
    eda.py                    # Exploratory analysis plots
    train.py                  # Model training (5 models)
    evaluate.py               # Metrics, plots, reports
    predict.py                # Inference
  models/                     # Saved model artifacts (.joblib)
  reports/
    figures/                  # 59 generated plots
    metrics.csv               # All metrics
    model_evaluation.md       # Evaluation report
    dataset_report.md         # Dataset description
  tests/
    test_data.py              # Preprocessing tests
    test_features.py          # Feature + leakage tests
    test_models.py            # Model + split + metrics tests
    test_download.py          # Download tests
  app.py                      # Streamlit web app
  requirements.txt
  .gitignore
  README.md
```

---

## Limitations and Future Improvements

### Limitations

1. **Single household**: Trained on one home in France (2006-2010). Results may
   not generalize to other households, climates, or time periods.
2. **No external factors**: Weather, occupancy changes, new appliances, and
   tariff changes are not modeled.
3. **Classical ML only**: Deep learning (LSTMs, Transformers) is not used to
   keep the project lightweight and interpretable.
4. **Historical patterns**: Assumes future consumption follows historical
   patterns. Sudden behavioral changes cause forecast errors.
5. **No real-time data**: Uses pre-trained models. Production systems would
   retrain periodically.
6. **Resampling loss**: 1-minute data is aggregated to hourly, losing
   intra-hour variability.
7. **Direct multi-horizon**: Each horizon has a separate model. Predictions
   for different horizons may not be temporally consistent.

### Future Improvements

- Add weather data as an exogenous feature
- Implement recursive or hybrid forecasting strategies
- Add prediction intervals / quantile forecasts
- Explore gradient-boosted trees (XGBoost, LightGBM)
- Implement online learning for model updates
- Add anomaly detection for unusual consumption
- Deploy as a real-time service with automated retraining

### Ethical Considerations

- Household consumption data is privacy-sensitive. In deployment, data should
  be anonymized and handled per applicable privacy laws.
- Forecasts should not be used for discriminatory pricing or targeting.

---

## Citation

```
Hebrail, G. & Berard, A. (2006). Individual Household Electric Power Consumption
[Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C58K54
```

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
