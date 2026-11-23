"""
Data preprocessing for the UCI Household Electric Power Consumption dataset.

Pipeline steps
----------
1. Load the raw minute-level text file.
2. Parse Date + Time into a single ``datetime`` index.
3. Convert numeric columns (missing values are encoded as ``'?'``).
4. Detect and report duplicate timestamps.
5. Chronologically sort.
6. Report missing values and impute with forward-fill then backward-fill.
7. Resample to hourly frequency:
      * Global_active_power   -- mean  (kW, also = kWh per hour)
      * Global_reactive_power -- mean  (kW)
      * Voltage                -- mean  (V)
      * Global_intensity       -- mean  (A)
      * Sub_metering_1/2/3     -- sum   (Wh per hour)
8. Save the cleaned hourly DataFrame to ``data/processed/hourly_consumption.csv``.

Usage::

    python -m src.data_preprocessing
"""

from __future__ import annotations

import pandas as pd

from .config import (
    NUMERIC_COLUMNS,
    PROCESSED_CSV,
    RAW_COLUMNS,
    RAW_CSV,
    RESAMPLE_FREQ,
    TARGET,
    ensure_dirs,
)


# --- 1. Load -------------------------------------------------------------------------------------------
def load_raw_data(filepath=None, nrows=None) -> pd.DataFrame:
    """
    Load the raw UCI text file into a DataFrame.

    Parameters
    -------
    filepath : Path or None
        Path to ``household_power_consumption.txt``.  Defaults to config.
    nrows : int or None
        Read only the first *n* rows (useful for testing / quick inspection).

    Returns
    -----
    pd.DataFrame
        Raw data with original column names, datetime not yet parsed.
    """
    path = filepath or RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}.  Run `python -m src.data_download` first."
        )

    df = pd.read_csv(
        path,
        sep=";",
        names=RAW_COLUMNS,
        skiprows=1,               # skip header row (we supply our own names)
        nrows=nrows,
        na_values="?",
        low_memory=False,
    )
    return df


# --- 2. Parse datetime ----------------------------------------------------------------------------
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine ``Date`` and ``Time`` columns into a ``datetime`` index.

    The raw format is ``dd/mm/yyyy`` and ``hh:mm:ss``.
    """
    # Combine into a single string column
    dt_str = df["Date"].astype(str) + " " + df["Time"].astype(str)
    datetime_index = pd.to_datetime(dt_str, format="%d/%m/%Y %H:%M:%S",
                                     errors="coerce")

    df = df.drop(columns=["Date", "Time"]).copy()
    df.index = datetime_index
    df.index.name = "datetime"

    # Drop rows where datetime parsing failed
    n_bad = df.index.isna().sum()
    if n_bad:
        print(f"[preprocess] WARNING: {n_bad} rows with unparseable datetime dropped")
        df = df[~df.index.isna()]

    # Explicitly sort by datetime index (step 5: chronological sort)
    df = df.sort_index()

    return df


# --- 3. Convert dtypes ----------------------------------------------------------------------------
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns from object/string to float."""
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# --- 4. Duplicates ----------------------------------------------------------------------------------
def check_duplicates(df: pd.DataFrame) -> int:
    """Report and return the number of duplicate timestamps."""
    n_dup = df.index.duplicated().sum()
    if n_dup:
        print(f"[preprocess] {n_dup} duplicate timestamps found -- averaging them")
    return n_dup


def handle_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Average rows that share the same timestamp."""
    return df.groupby(level=0).mean()


# --- 5. Missing values ----------------------------------------------------------------------------
def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with missing-value counts and percentages."""
    total = len(df)
    missing = df.isna().sum()
    pct = (missing / total * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using forward-fill then backward-fill.

    For time-series data, forward-fill (carrying the last known value forward)
    is a conservative approach that does not use future information for the
    current row.  Backward-fill mops up any leading NaNs.
    """
    n_before = df.isna().sum().sum()
    df = df.ffill().bfill()
    n_after = df.isna().sum().sum()
    if n_before > 0:
        print(f"[preprocess] Imputed {n_before} missing values (ffill -- bfill).  "
              f"Remaining: {n_after}")
    return df


# --- 6. Resample to hourly -----------------------------------------------------------------------
def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute-level data to hourly frequency.

    Aggregation rules
    ~~~~~~~~~~~~~~~~~
    * ``Global_active_power`` (kW, minute-averaged) -- **mean**
      The hourly mean of minute-averaged power in kW equals the hourly energy
      consumption in kWh, because each minute contributes
      ``power_kW -- (1/60) h`` of energy and there are 60 minutes in an hour.
    * ``Global_reactive_power`` (kW) -- mean
    * ``Voltage`` (V) -- mean
    * ``Global_intensity`` (A) -- mean
    * ``Sub_metering_1/2/3`` (Wh per minute) -- **sum** (total Wh in the hour)
    """
    agg_funcs = {}
    for col in df.columns:
        if col.startswith("Sub_metering"):
            agg_funcs[col] = "sum"
        else:
            agg_funcs[col] = "mean"

    hourly = df.resample(RESAMPLE_FREQ).agg(agg_funcs)

    # Rename sub-metering columns to clarify they are now hourly Wh
    rename_map = {
        "Sub_metering_1": "Sub_metering_1_Wh",
        "Sub_metering_2": "Sub_metering_2_Wh",
        "Sub_metering_3": "Sub_metering_3_Wh",
    }
    hourly = hourly.rename(columns=rename_map)

    return hourly


# --- Full pipeline ----------------------------------------------------------------------------------
def run_preprocessing(nrows=None, save: bool = True) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline.

    Parameters
    -------
    nrows : int or None
        If set, only process the first *n* raw rows (for quick testing).
    save : bool
        Save the result to ``PROCESSED_CSV``.

    Returns
    -----
    pd.DataFrame
        Hourly resampled DataFrame.
    """
    ensure_dirs()

    print("[preprocess] Loading raw data -...")
    df = load_raw_data(nrows=nrows)
    print(f"[preprocess] Loaded {len(df):,} rows")

    print("[preprocess] Parsing datetime -...")
    df = parse_datetime(df)

    print("[preprocess] Converting data types -...")
    df = convert_dtypes(df)

    print("[preprocess] Checking duplicates -...")
    check_duplicates(df)
    df = handle_duplicates(df)

    print("[preprocess] Reporting missing values -...")
    missing_report = report_missing(df)
    if not missing_report.empty:
        print(missing_report)
    else:
        print("[preprocess] No missing values in raw data.")

    print("[preprocess] Imputing missing values -...")
    df = impute_missing(df)

    print("[preprocess] Resampling to hourly -...")
    hourly = resample_hourly(df)

    # After resampling there may be a few hours with no underlying minutes
    # (e.g. the very first / last partial hour).  Impute again.
    n_missing_hourly = hourly.isna().sum().sum()
    if n_missing_hourly:
        print(f"[preprocess] {n_missing_hourly} NaNs after resampling -- imputing")
        hourly = hourly.ffill().bfill()

    print(f"[preprocess] Hourly data: {len(hourly):,} rows -- {hourly.shape[1]} cols")
    print(f"[preprocess] Date range: {hourly.index.min()} -- {hourly.index.max()}")

    if save:
        hourly.to_csv(PROCESSED_CSV)
        print(f"[preprocess] Saved to {PROCESSED_CSV}")

    return hourly


def load_hourly_data(filepath=None) -> pd.DataFrame:
    """Load the pre-processed hourly CSV (or run preprocessing if missing)."""
    path = filepath or PROCESSED_CSV
    if not path.exists():
        print("[preprocess] Processed file not found -- running preprocessing -...")
        return run_preprocessing()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    return df


if __name__ == "__main__":
    hourly = run_preprocessing()
    print(f"\n-- Preprocessing complete. Shape: {hourly.shape}")
    print(hourly.head())
    print(hourly.describe())
