"""
Download the UCI Individual Household Electric Power Consumption dataset.

Usage (run from project root)::

    python -m src.data_download

The module downloads the ZIP archive, extracts the inner .txt file to
``data/raw/``, and verifies its integrity.
"""

from __future__ import annotations

import io
import sys
import zipfile

import requests

from .config import RAW_CSV, RAW_DIR, ZIP_PATH, ZIP_URL, ensure_dirs

# Expected size of the extracted txt (≈127 MB).  Used only for a sanity check.
MIN_TXT_BYTES = 50_000_000  # 50 MB lower bound


def _download_zip() -> Path:
    """Download the dataset ZIP to ``ZIP_PATH`` and return the path."""
    ensure_dirs()
    print(f"[download] Fetching dataset from {ZIP_URL} …")
    try:
        resp = requests.get(ZIP_URL, timeout=300, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[download] ERROR – could not download: {exc}", file=sys.stderr)
        raise

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(ZIP_PATH, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r[download] {downloaded:,}/{total:,} bytes ({pct:.1f}%)", end="")
    print()
    print(f"[download] Saved to {ZIP_PATH}")
    return ZIP_PATH


def _extract_zip(zip_path: Path = ZIP_PATH) -> Path:
    """Extract the .txt file from the downloaded ZIP into ``RAW_DIR``."""
    print(f"[extract] Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # The archive contains a single file: household_power_consumption.txt
        members = zf.namelist()
        txt_members = [m for m in members if m.endswith(".txt")]
        if not txt_members:
            raise RuntimeError("No .txt file found inside the ZIP archive.")
        target = txt_members[0]
        zf.extract(target, RAW_DIR)

        # Rename / move to the canonical location
        extracted = RAW_DIR / target
        if extracted != RAW_CSV:
            # Remove old file if present and rename
            if RAW_CSV.exists():
                RAW_CSV.unlink()
            extracted.rename(RAW_CSV)

    size = RAW_CSV.stat().st_size
    print(f"[extract] Extracted {RAW_CSV.name} ({size:,} bytes)")
    if size < MIN_TXT_BYTES:
        raise RuntimeError(
            f"Extracted file unexpectedly small ({size:,} bytes < {MIN_TXT_BYTES:,})"
        )
    return RAW_CSV


def download_dataset(force: bool = False) -> Path:
    """
    Download and extract the dataset if not already present.

    Parameters
    ----------
    force : bool
        Re-download even if the raw file already exists.

    Returns
    -------
    Path
        Path to the extracted ``household_power_consumption.txt``.
    """
    if RAW_CSV.exists() and not force:
        print(f"[download] Raw data already present at {RAW_CSV}")
        return RAW_CSV

    if not ZIP_PATH.exists() or force:
        _download_zip()

    return _extract_zip()


def verify_dataset() -> bool:
    """Return True if the raw dataset file exists and looks valid."""
    if not RAW_CSV.exists():
        return False
    if RAW_CSV.stat().st_size < MIN_TXT_BYTES:
        return False
    return True


if __name__ == "__main__":
    try:
        path = download_dataset()
        print(f"\n[OK] Dataset ready at {path}")
    except Exception as exc:
        print(f"\n[FAIL] Download failed: {exc}", file=sys.stderr)
        print(
            "\nManual fallback:\n"
            "  1. Visit https://archive.ics.uci.edu/dataset/235/"
            "individual+household+electric+power+consumption\n"
            "  2. Click 'Download'.\n"
            "  3. Unzip and copy household_power_consumption.txt into "
            f"{RAW_DIR}",
            file=sys.stderr,
        )
        sys.exit(1)
