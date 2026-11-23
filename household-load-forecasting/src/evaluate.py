"""
Model evaluation and visualization for household load forecasting.

Computes MAE, RMSE, and R^2 for every model and every horizon on both the
validation and test sets.  Generates comparison plots and saves them to
``reports/figures/``.

Usage::

    python -m src.evaluate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    FIGURES_DIR,
    HORIZONS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_PATH,
    REPORTS_DIR,
    TARGET,
    ensure_dirs,
)
from .train import (
    chronological_split,
    create_models,
    select_best_model,
    train_all_models,
)
from .data_preprocessing import load_hourly_data
from .feature_engineering import build_features, get_feature_columns


# ── Metric calculation ────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and R^2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def evaluate_all_models(results: dict) -> pd.DataFrame:
    """
    Evaluate all models on validation and test sets.

    Returns a DataFrame with columns:
        horizon, model, split, MAE, RMSE, R2
    """
    rows = []
    for horizon, horizon_results in results.items():
        for name, res in horizon_results.items():
            # Validation metrics
            val_metrics = compute_metrics(res["val_actuals"], res["val_predictions"])
            rows.append({
                "horizon": f"t+{horizon}h",
                "model": name,
                "split": "validation",
                **val_metrics,
            })
            # Test metrics
            test_metrics = compute_metrics(res["test_actuals"], res["test_predictions"])
            rows.append({
                "horizon": f"t+{horizon}h",
                "model": name,
                "split": "test",
                **test_metrics,
            })

    metrics_df = pd.DataFrame(rows)
    return metrics_df


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_actual_vs_predicted(
    results: dict,
    horizon: int,
    model_name: str,
    split: str = "test",
    n_points: int = 168,
    save: bool = True,
) -> Path | None:
    """
    Plot actual vs predicted consumption — dual-panel layout for clarity.

    Top panel  : Actual (solid blue) vs Predicted (dashed red) on same axis,
                 with a shaded error band.
    Bottom panel: Prediction error (residual) over time, centred at zero.
    """
    res = results[horizon][model_name]
    if split == "test":
        actuals = res["test_actuals"]
        preds   = res["test_predictions"]
    else:
        actuals = res["val_actuals"]
        preds   = res["val_predictions"]

    n      = min(n_points, len(actuals))
    x      = np.arange(n)
    act    = actuals[:n]
    pred   = preds[:n]
    errors = pred - act

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # ── Top: Actual vs Predicted ──────────────────────────────────────────
    ax1.plot(x, act,  label="Actual",
             color="#1f77b4", linewidth=2.0, zorder=3)
    ax1.plot(x, pred, label=f"Predicted (t+{horizon}h)",
             color="#d62728", linewidth=1.8, linestyle="--", zorder=4)
    # Shade the gap between actual and predicted
    ax1.fill_between(x, act, pred,
                     alpha=0.15, color="orange", label="Error band")

    ax1.set_ylabel("Global Active Power (kW)", fontsize=11)
    ax1.set_title(
        f"{model_name.replace('_', ' ').title()}  —  "
        f"Actual vs Predicted  |  Horizon t+{horizon}h  |  {split.title()} set",
        fontsize=12, fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n - 1)

    # ── Bottom: Residuals ─────────────────────────────────────────────────
    ax2.bar(x, errors, color=np.where(errors >= 0, "#d62728", "#1f77b4"),
            alpha=0.6, width=1.0)
    ax2.axhline(0, color="black", linewidth=1.2, linestyle="-")
    mae_str = f"MAE = {np.abs(errors).mean():.3f} kW"
    ax2.set_ylabel("Error (kW)", fontsize=10)
    ax2.set_xlabel(f"Hour index (showing first {n} hours)", fontsize=10)
    ax2.set_title(f"Residuals  ({mae_str})", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(h_pad=0.4)

    path = FIGURES_DIR / f"actual_vs_pred_{model_name}_h{horizon}_{split}.png"
    if save:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [plot] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_error_distribution(
    results: dict,
    horizon: int,
    model_name: str,
    split: str = "test",
    save: bool = True,
) -> Path | None:
    """Plot histogram of prediction errors (residuals)."""
    res = results[horizon][model_name]
    if split == "test":
        actuals = res["test_actuals"]
        preds = res["test_predictions"]
    else:
        actuals = res["val_actuals"]
        preds = res["val_predictions"]

    errors = preds - actuals

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, edgecolor="black", color="steelblue", alpha=0.7)
    ax.axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
    ax.set_xlabel("Prediction Error (kW)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution - {model_name} - Horizon t+{horizon}h ({split})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / f"error_dist_{model_name}_h{horizon}_{split}.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [plot] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    split: str = "validation",
    metric: str = "MAE",
    save: bool = True,
) -> Path | None:
    """Bar chart comparing models by a given metric for all horizons."""
    subset = metrics_df[metrics_df["split"] == split].copy()
    horizons = sorted(subset["horizon"].unique())
    models = sorted(subset["model"].unique())

    x = np.arange(len(horizons))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        vals = []
        for h in horizons:
            row = subset[(subset["horizon"] == h) & (subset["model"] == model)]
            vals.append(row[metric].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison - {metric} ({split})")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = FIGURES_DIR / f"model_comparison_{metric}_{split}.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [plot] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_horizon_comparison(
    metrics_df: pd.DataFrame,
    model_name: str,
    split: str = "test",
    save: bool = True,
) -> Path | None:
    """Line chart showing how a model's metrics change across horizons."""
    subset = metrics_df[
        (metrics_df["model"] == model_name) & (metrics_df["split"] == split)
    ].copy()
    horizons = sorted(subset["horizon"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ["MAE", "RMSE"]:
        vals = []
        for h in horizons:
            row = subset[subset["horizon"] == h]
            vals.append(row[metric].values[0] if len(row) > 0 else 0)
        ax.plot(horizons, vals, marker="o", linewidth=2, label=metric)

    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Error (kW)")
    ax.set_title(f"Horizon-wise Performance - {model_name} ({split})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / f"horizon_comparison_{model_name}_{split}.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [plot] Saved {path}")
    plt.close(fig)
    return path if save else None


def plot_scatter_actual_vs_pred(
    results: dict,
    horizon: int,
    model_name: str,
    split: str = "test",
    save: bool = True,
) -> Path | None:
    """Scatter plot of actual vs predicted values."""
    res = results[horizon][model_name]
    if split == "test":
        actuals = res["test_actuals"]
        preds = res["test_predictions"]
    else:
        actuals = res["val_actuals"]
        preds = res["val_predictions"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(actuals, preds, alpha=0.3, s=10, color="steelblue")
    # Perfect prediction line
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2,
            label="Perfect prediction")
    ax.set_xlabel("Actual (kW)")
    ax.set_ylabel("Predicted (kW)")
    ax.set_title(f"Scatter: {model_name} - Horizon t+{horizon}h ({split})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / f"scatter_{model_name}_h{horizon}_{split}.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [plot] Saved {path}")
    plt.close(fig)
    return path if save else None


# ── Main evaluation routine ──────────────────────────────────────────────────
def run_evaluation(results: dict | None = None, save: bool = True) -> pd.DataFrame:
    """
    Run full evaluation: compute metrics, generate all plots, save results.

    Returns the metrics DataFrame.
    """
    ensure_dirs()

    if results is None:
        print("[eval] No results provided — training models first ...")
        results = train_all_models(save=save)

    # Compute metrics
    print("\n[eval] Computing metrics ...")
    metrics_df = evaluate_all_models(results)

    # Select best model per horizon (based on validation MAE)
    best = select_best_model(results)

    # Save metrics
    if save:
        metrics_df.to_csv(METRICS_PATH, index=False)
        print(f"[eval] Metrics saved to {METRICS_PATH}")

    # Print metrics table
    print("\n" + "=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)
    val_df = metrics_df[metrics_df["split"] == "validation"]
    print(val_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TEST METRICS")
    print("=" * 70)
    test_df = metrics_df[metrics_df["split"] == "test"]
    print(test_df.to_string(index=False))

    # Generate plots
    print("\n[eval] Generating plots ...")

    for horizon in HORIZONS:
        for name in results[horizon].keys():
            plot_actual_vs_predicted(results, horizon, name, split="test", save=save)
            plot_error_distribution(results, horizon, name, split="test", save=save)
            plot_scatter_actual_vs_pred(results, horizon, name, split="test", save=save)

        # Best model horizon comparison
        best_name = best[horizon]["name"]
        plot_horizon_comparison(metrics_df, best_name, split="test", save=save)

    # Model comparison charts
    for metric in ["MAE", "RMSE", "R2"]:
        plot_model_comparison(metrics_df, split="validation", metric=metric, save=save)
        plot_model_comparison(metrics_df, split="test", metric=metric, save=save)

    # Best model test plots
    print("\n[eval] Best models:")
    for h, info in best.items():
        print(f"  Horizon t+{h}h: {info['name']} (val MAE={info['val_mae']:.4f})")

    return metrics_df


def generate_model_evaluation_report(
    metrics_df: pd.DataFrame,
    best: dict,
) -> Path:
    """Generate the model_evaluation.md report from actual metrics."""
    report_path = REPORTS_DIR / "model_evaluation.md"

    lines = ["# Model Evaluation Report\n"]
    lines.append("## Overview\n")
    lines.append(
        "All models are evaluated on both validation and test sets using "
        "chronological (temporal) splits. Model selection is based on "
        "validation MAE. The test set is used only once for final evaluation.\n"
    )

    # Validation metrics table
    lines.append("## Validation Metrics\n")
    val_df = metrics_df[metrics_df["split"] == "validation"].copy()
    lines.append("| Horizon | Model | MAE (kW) | RMSE (kW) | R² |")
    lines.append("|---------|-------|----------|-----------|-----|")
    for _, row in val_df.iterrows():
        lines.append(
            f"| {row['horizon']} | {row['model']} | "
            f"{row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} |"
        )

    # Test metrics table
    lines.append("\n## Test Metrics\n")
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    lines.append("| Horizon | Model | MAE (kW) | RMSE (kW) | R² |")
    lines.append("|---------|-------|----------|-----------|-----|")
    for _, row in test_df.iterrows():
        lines.append(
            f"| {row['horizon']} | {row['model']} | "
            f"{row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} |"
        )

    # Best models
    lines.append("\n## Best Model Selection\n")
    lines.append(
        "The best model per horizon is selected based on validation MAE "
        "(lowest = best):\n\n"
    )
    for h, info in best.items():
        lines.append(f"- **Horizon t+{h}h**: `{info['name']}` (validation MAE = {info['val_mae']:.4f} kW)")

    lines.append("\n## Metric Definitions\n")
    lines.append("- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual values. Lower is better.")
    lines.append("- **RMSE** (Root Mean Square Error): Square root of the average squared errors. Penalises large errors more than MAE. Lower is better.")
    lines.append("- **R²** (Coefficient of Determination): Proportion of variance in the target explained by the model. 1.0 = perfect, 0.0 = same as predicting the mean. Can be negative for very poor models.")

    lines.append("\n## Figures\n")
    lines.append("All plots are saved in `reports/figures/`. Key plots include:")
    lines.append("- Actual vs predicted consumption over time")
    lines.append("- Error distribution histograms")
    lines.append("- Scatter plots of actual vs predicted")
    lines.append("- Model comparison bar charts")
    lines.append("- Horizon-wise performance comparisons")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] Report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    results = train_all_models()
    metrics_df = run_evaluation(results)
    best = select_best_model(results)
    generate_model_evaluation_report(metrics_df, best)
