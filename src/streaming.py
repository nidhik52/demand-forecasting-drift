"""
streaming.py
────────────
Simulates real-time retail data streaming.

In a production system, new sales transactions would arrive via Kafka,
Kinesis, or a database CDC feed.  Here we replay historical 2025 data
day-by-day to create a realistic simulation without extra infrastructure.

Each iteration yields one day's worth of transactions (all SKUs) as a
small DataFrame — exactly what a drift detector or monitoring dashboard
would receive in production.

Drift Simulation
────────────────
The historical dataset does not contain natural concept drift.  To make
drift detection and automatic retraining demonstrable, synthetic demand
drift is injected for three SKUs after DRIFT_START_DATE:

  ELEC-001 → demand × 3.0  (sharp spike — simulates electronics viral trend)
  GROC-002 → demand × 2.5  (moderate spike — simulates supply disruption)
  CLTH-003 → demand × 2.0  (mild spike — simulates seasonal fashion shift)

This causes large forecast errors that push the rolling MAE above the
detection threshold, triggering automatic model retraining.
"""

import time
from pathlib import Path
from typing import Generator

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DAILY_PATH    = Path("data/processed/daily_demand.csv")
FORECAST_PATH = Path("data/processed/forecast_2026.csv")

# ── Default streaming parameters ──────────────────────────────────────────────
STREAM_YEAR  = 2025     # historical year to replay as the "live" stream
STREAM_DELAY = 0.0      # seconds between batches  (0 = full speed for pipeline)
STREAM_DAYS  = 90       # number of days to stream (enough for drift to trigger)

# ── Drift injection configuration ─────────────────────────────────────────────
# After DRIFT_START_DATE, multiply demand for these SKUs to simulate concept drift.
# This creates large forecast errors → rolling MAE exceeds threshold → retrain.
DRIFT_START_DATE = pd.Timestamp("2025-01-15")

DRIFT_MULTIPLIERS: dict[str, float] = {
    "ELEC-001": 3.0,   # sharp spike — electronics viral trend
    "GROC-002": 2.5,   # moderate spike — supply disruption
    "CLTH-003": 2.0,   # mild spike — seasonal fashion shift
}


def _apply_drift(sku: str, demand: float, date: pd.Timestamp) -> tuple[float, bool]:
    """
    Apply synthetic drift multiplier if the SKU and date qualify.

    Returns
    ───────
    (adjusted_demand, drift_injected)
    """
    if date >= DRIFT_START_DATE and sku in DRIFT_MULTIPLIERS:
        multiplier = DRIFT_MULTIPLIERS[sku]
        return demand * multiplier, True
    return demand, False


def stream_daily_batches(
    path: Path = DAILY_PATH,
    year: int = STREAM_YEAR,
    delay: float = STREAM_DELAY,
    max_days: int = STREAM_DAYS,
) -> Generator[tuple[pd.Timestamp, pd.DataFrame], None, None]:
    """
    Yield (date, batch_df) for every day in `year`, in chronological order.

    Each yielded batch_df contains columns:  Date | SKU | demand
    and represents one incoming day of retail transactions.

    Synthetic drift is injected for configured SKUs after DRIFT_START_DATE
    (see module-level DRIFT_MULTIPLIERS).

    Parameters
    ──────────
    path     : path to the aggregated daily demand CSV
    year     : calendar year whose data is replayed (default 2025)
    delay    : seconds to sleep between yields — useful for live demos
    max_days : cap on number of days to stream (default 90)
    """
    daily = pd.read_csv(path, parse_dates=["Date"])

    stream_data = daily[daily["Date"].dt.year == year].copy()

    if stream_data.empty:
        available = sorted(daily["Date"].dt.year.unique())
        print(
            f"[Streaming] No data found for year {year}.  "
            f"Available years: {available}"
        )
        return

    sorted_dates = sorted(stream_data["Date"].unique())[:max_days]
    total        = len(sorted_dates)
    print(f"[Streaming] Replaying {total} days from {year} "
          f"(drift injection starts {DRIFT_START_DATE.date()}) …")

    for date in sorted_dates:
        ts    = pd.Timestamp(date)
        batch = stream_data[stream_data["Date"] == date].copy()

        # ── Apply synthetic drift ────────────────────────────────────────────
        drift_rows: list[str] = []
        for idx, row in batch.iterrows():
            adjusted, injected = _apply_drift(row["SKU"], float(row["demand"]), ts)
            if injected:
                batch.at[idx, "demand"] = adjusted
                drift_rows.append(row["SKU"])

        if drift_rows:
            for sku in drift_rows:
                print(f"  [Drift Injection] {sku} on {ts.date()} "
                      f"→ demand × {DRIFT_MULTIPLIERS[sku]}")

        yield ts, batch

        if delay > 0:
            time.sleep(delay)


def get_forecast_for_date(
    date: pd.Timestamp,
    forecast_path: Path = FORECAST_PATH,
) -> pd.DataFrame:
    """
    Return the forecast rows for a specific date across all SKUs.

    Used by the drift detector to retrieve the predicted value that
    corresponds to each incoming actual observation.

    Returns DataFrame with columns:  SKU | forecast_demand
    """
    fc = pd.read_csv(forecast_path, parse_dates=["Date"])
    return fc[fc["Date"] == date][["SKU", "forecast_demand"]].copy()
