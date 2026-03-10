"""
data_preprocessing.py
─────────────────────
Loads the raw retail sales CSV, cleans it, and aggregates it into
daily demand per SKU — the standard input format expected by Prophet.

Dataset note
────────────
The raw file (sales_with_sku.csv) uses:
  • Date_of_Sale  → date of the transaction  (YYYY-MM-DD)
  • SKU           → product SKU code
  • Sales_Amount  → revenue per transaction

Because the dataset has no Quantity_Sold column, 'demand' is defined
as the NUMBER OF TRANSACTIONS per SKU per day.  Each row represents
one sale, so row-count is a direct proxy for units demanded.

Output
──────
  data/processed/daily_demand.csv  →  Date | SKU | demand
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH      = Path("data/raw/sales_with_sku.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH   = PROCESSED_DIR / "daily_demand.csv"

# ── Column name mapping (adapts to actual CSV headers) ────────────────────────
DATE_COL     = "Date_of_Sale"   # date column in the raw file
SKU_COL      = "SKU"
SALES_ID_COL = "Sales_ID"       # used as the row counter for demand


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV and coerce the date column to datetime."""
    df = pd.read_csv(path)

    if DATE_COL not in df.columns:
        raise KeyError(
            f"Expected column '{DATE_COL}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Parse dates — the file uses YYYY-MM-DD format
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d", errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unusable rows and fill numeric gaps.

    Rules
    ─────
    • Drop any row where the date or SKU is null  (cannot aggregate without them).
    • Fill missing Sales_Amount with the per-SKU median  (preserves scale).
    • Fill missing Discount with 0  (no discount is a safe default).
    """
    before = len(df)
    df = df.dropna(subset=[DATE_COL, SKU_COL])
    dropped = before - len(df)
    if dropped:
        print(f"  [clean] Dropped {dropped} rows with null date/SKU.")

    # Fill numeric columns per-SKU to avoid cross-SKU contamination
    if "Sales_Amount" in df.columns:
        df["Sales_Amount"] = df.groupby(SKU_COL)["Sales_Amount"].transform(
            lambda x: x.fillna(x.median())
        )
    if "Discount" in df.columns:
        df["Discount"] = df["Discount"].fillna(0)

    return df.reset_index(drop=True)


def aggregate_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a tidy daily demand series:  Date | SKU | demand

    'demand' = count of sales transactions per SKU per day.

    After aggregation the series is reindexed to a full daily grid
    (all SKUs × all dates) with 0 filling any missing days, giving
    Prophet a gap-free time series.
    """
    daily = (
        df.groupby([DATE_COL, SKU_COL])
        .agg(demand=(SALES_ID_COL, "count"))
        .reset_index()
        .rename(columns={DATE_COL: "Date"})
    )

    # Reindex to a complete Date × SKU grid — fills implicit zeros
    daily = _fill_missing_dates(daily)
    return daily.sort_values(["SKU", "Date"]).reset_index(drop=True)


def _fill_missing_dates(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete Date × SKU Cartesian product and fill missing
    cells with 0.  This ensures Prophet models are trained on a
    continuous, gap-free series.
    """
    date_range = pd.date_range(daily["Date"].min(), daily["Date"].max(), freq="D")
    skus       = daily["SKU"].unique()

    full_index = pd.MultiIndex.from_product(
        [date_range, skus], names=["Date", "SKU"]
    )
    daily = (
        daily.set_index(["Date", "SKU"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    return daily


def preprocess(raw_path: Path = RAW_PATH, save: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline.  Call this from pipeline.py.

    Returns the aggregated daily demand DataFrame and optionally
    writes it to data/processed/daily_demand.csv.
    """
    print("[Preprocessing] Loading raw data …")
    df = load_raw_data(raw_path)
    print(f"  Raw rows   : {len(df):,}  |  columns: {df.columns.tolist()}")

    df = clean_data(df)
    print(f"  Clean rows : {len(df):,}")

    daily = aggregate_daily_demand(df)
    n_skus  = daily["SKU"].nunique()
    n_dates = daily["Date"].nunique()
    print(f"  Daily grid : {len(daily):,} rows  ({n_skus} SKUs × {n_dates} dates)")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(OUTPUT_PATH, index=False)
        print(f"  Saved → {OUTPUT_PATH}")

    return daily


if __name__ == "__main__":
    preprocess()
