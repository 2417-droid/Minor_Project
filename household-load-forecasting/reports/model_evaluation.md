# Model Evaluation Report

## Overview

All models are evaluated on both validation and test sets using chronological (temporal) splits. Model selection is based on validation MAE. The test set is used only once for final evaluation.

## Validation Metrics

| Horizon | Model | MAE (kW) | RMSE (kW) | R² |
|---------|-------|----------|-----------|-----|
| t+1h | persistence | 0.6633 | 0.9380 | -0.1258 |
| t+1h | seasonal_naive | 0.6764 | 0.9551 | -0.1671 |
| t+1h | ridge | 0.4207 | 0.5823 | 0.5662 |
| t+1h | decision_tree | 0.4213 | 0.6282 | 0.4950 |
| t+1h | random_forest | 0.3480 | 0.5142 | 0.6617 |
| t+6h | persistence | 1.0722 | 1.3374 | -1.2884 |
| t+6h | seasonal_naive | 1.0808 | 1.3478 | -1.3242 |
| t+6h | ridge | 0.5724 | 0.7474 | 0.2854 |
| t+6h | decision_tree | 0.5675 | 0.7946 | 0.1922 |
| t+6h | random_forest | 0.4899 | 0.6748 | 0.4173 |
| t+24h | persistence | 0.7122 | 0.9997 | -0.2776 |
| t+24h | seasonal_naive | 0.6974 | 0.9911 | -0.2557 |
| t+24h | ridge | 0.5414 | 0.7390 | 0.3019 |
| t+24h | decision_tree | 0.5984 | 0.8379 | 0.1025 |
| t+24h | random_forest | 0.5088 | 0.6889 | 0.3933 |

## Test Metrics

| Horizon | Model | MAE (kW) | RMSE (kW) | R² |
|---------|-------|----------|-----------|-----|
| t+1h | persistence | 0.5239 | 0.7676 | -0.1899 |
| t+1h | seasonal_naive | 0.6044 | 0.8419 | -0.4314 |
| t+1h | ridge | 0.3554 | 0.4972 | 0.5009 |
| t+1h | decision_tree | 0.3570 | 0.5406 | 0.4099 |
| t+1h | random_forest | 0.3051 | 0.4461 | 0.5982 |
| t+6h | persistence | 0.7457 | 1.0036 | -1.0374 |
| t+6h | seasonal_naive | 0.7713 | 1.0163 | -1.0891 |
| t+6h | ridge | 0.4939 | 0.6244 | 0.2113 |
| t+6h | decision_tree | 0.4789 | 0.6821 | 0.0589 |
| t+6h | random_forest | 0.4130 | 0.5689 | 0.3452 |
| t+24h | persistence | 0.5506 | 0.7991 | -0.2914 |
| t+24h | seasonal_naive | 0.5763 | 0.8141 | -0.3401 |
| t+24h | ridge | 0.4532 | 0.6026 | 0.2658 |
| t+24h | decision_tree | 0.4967 | 0.7092 | -0.0171 |
| t+24h | random_forest | 0.4264 | 0.5811 | 0.3172 |

## Best Model Selection

The best model per horizon is selected based on validation MAE (lowest = best):


- **Horizon t+1h**: `random_forest` (validation MAE = 0.3480 kW)
- **Horizon t+6h**: `random_forest` (validation MAE = 0.4899 kW)
- **Horizon t+24h**: `random_forest` (validation MAE = 0.5088 kW)

## Metric Definitions

- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual values. Lower is better.
- **RMSE** (Root Mean Square Error): Square root of the average squared errors. Penalises large errors more than MAE. Lower is better.
- **R²** (Coefficient of Determination): Proportion of variance in the target explained by the model. 1.0 = perfect, 0.0 = same as predicting the mean. Can be negative for very poor models.

## Figures

All plots are saved in `reports/figures/`. Key plots include:
- Actual vs predicted consumption over time
- Error distribution histograms
- Scatter plots of actual vs predicted
- Model comparison bar charts
- Horizon-wise performance comparisons