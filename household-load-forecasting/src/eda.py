"""
Exploratory Data Analysis (EDA) module.

Generates visualizations of consumption patterns and saves them to
reports/figures/.

Usage::

    python -m src.eda
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURES_DIR, TARGET, ensure_dirs
from .data_preprocessing import load_hourly_data


def plot_consumption_over_time(df: pd.DataFrame, save: bool = True):
    """Plot the full time-series of hourly consumption."""
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df.index, df[TARGET], linewidth=0.3, color="steelblue", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Hourly Electricity Consumption Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / "eda_consumption_over_time.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [eda] Saved {path}")
    plt.close(fig)


def plot_daily_pattern(df: pd.DataFrame, save: bool = True):
    """Plot average consumption by hour of day."""
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    daily_avg = df_copy.groupby("hour")[TARGET].mean()
    daily_std = df_copy.groupby("hour")[TARGET].std()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(daily_avg.index, daily_avg.values, marker="o", color="darkorange",
            linewidth=2, label="Mean")
    ax.fill_between(daily_avg.index, daily_avg - daily_std, daily_avg + daily_std,
                    alpha=0.2, color="darkorange", label="+/- 1 std")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Average Daily Consumption Pattern")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / "eda_daily_pattern.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [eda] Saved {path}")
    plt.close(fig)


def plot_weekly_pattern(df: pd.DataFrame, save: bool = True):
    """Plot average consumption by day of week."""
    df_copy = df.copy()
    df_copy["dayofweek"] = df_copy.index.dayofweek
    weekly_avg = df_copy.groupby("dayofweek")[TARGET].mean()
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(day_names, weekly_avg.values, color="seagreen")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Average Weekly Consumption Pattern")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = FIGURES_DIR / "eda_weekly_pattern.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [eda] Saved {path}")
    plt.close(fig)


def plot_monthly_pattern(df: pd.DataFrame, save: bool = True):
    """Plot average consumption by month (seasonal pattern)."""
    df_copy = df.copy()
    df_copy["month"] = df_copy.index.month
    monthly_avg = df_copy.groupby("month")[TARGET].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(month_names, monthly_avg.values, color="coral")
    ax.set_xlabel("Month")
    ax.set_ylabel("Global Active Power (kW)")
    ax.set_title("Average Monthly Consumption (Seasonal Pattern)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = FIGURES_DIR / "eda_monthly_pattern.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [eda] Saved {path}")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    """Plot correlation heatmap of all variables."""
    cols = [TARGET, "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1_Wh", "Sub_metering_2_Wh", "Sub_metering_3_Wh"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()

    path = FIGURES_DIR / "eda_correlation_heatmap.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [eda] Saved {path}")
    plt.close(fig)


def plot_consumption_distribution(df: pd.DataFrame, save: bool = True):
    """Plot histogram and boxplot of consumption."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(df[TARGET], bins=100, edgecolor="black", color="steelblue", alpha=0.7)
    ax1.set_xlabel("Global Active Power (kW)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Consumption Distribution")
    ax1.grid(True, alpha=0.3)

    # Boxplot by hour
    df_copy = df.copy()
    df_copy["hour"] = df_copy.index.hour
    df_copy.boxplot(column=TARGET, by="hour", ax=ax2, grid=False)
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Global Active Power (kW)")
    ax2.set_title("Consumption by Hour")
    plt.suptitle("")

    fig.tight_layout()

    path = FIGURES_DIR / "eda_consumption_distribution.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [eda] Saved {path}")
    plt.close(fig)


def plot_missing_data_intervals(df: pd.DataFrame, save: bool = True):
    """Visualize missing data intervals over time."""
    # Check for gaps in the hourly index
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing_hours = full_range.difference(df.index)

    fig, ax = plt.subplots(figsize=(16, 3))
    if len(missing_hours) > 0:
        for mh in missing_hours[:1000]:  # limit for performance
            ax.axvline(mh, color="red", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_title(f"Missing Hourly Intervals ({len(missing_hours)} gaps)")
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = FIGURES_DIR / "eda_missing_intervals.png"
    if save:
        fig.savefig(path, dpi=150)
        print(f"  [eda] Saved {path}")
    plt.close(fig)


def run_eda(save: bool = True):
    """Run the full EDA pipeline."""
    ensure_dirs()

    print("[eda] Loading hourly data ...")
    df = load_hourly_data()
    print(f"[eda] Data shape: {df.shape}")
    print(f"[eda] Date range: {df.index.min()} to {df.index.max()}")

    print("\n[eda] Generating plots ...")
    plot_consumption_over_time(df, save=save)
    plot_daily_pattern(df, save=save)
    plot_weekly_pattern(df, save=save)
    plot_monthly_pattern(df, save=save)
    plot_correlation_heatmap(df, save=save)
    plot_consumption_distribution(df, save=save)
    plot_missing_data_intervals(df, save=save)

    # Summary statistics
    print(f"\n[eda] Summary Statistics:")
    print(df.describe().round(3))

    print(f"\n[eda] All EDA plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
