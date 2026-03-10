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
"""

import time
from pathlib import Path
from typing import Generator

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DAILY_PATH    = Path("data/processed/daily_demand.csv")
FORECAST_PATH = Path("data/processed/forecast_2026.csv")

# ── Default parameters ────────────────────────────────────────────────────────
STREAM_YEAR  = 2025     # historical year to replay as the "live" stream
STREAM_DELAY = 0.0      # seconds between batches  (0 = full speed for pipeline)


def stream_daily_batches(
    path: Path = DAILY_PATH,
    year: int = STREAM_YEAR,
    delay: float = STREAM_DELAY,
) -> Generator[tuple[pd.Timestamp, pd.DataFrame], None, None]:
    """
    Yield (date, batch_df) for every day in `year`, in chronological order.

    Each yielded batch_df contains columns:  Date | SKU | demand
    and represents one incoming day of retail transactions.

    Parameters
    ──────────
    path  : path to the aggregated daily demand CSV
    year  : calendar year whose data is replayed (default 2025)
    delay : seconds to sleep between yields — useful for live demos
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

    sorted_dates = sorted(stream_data["Date"].unique())
    total        = len(sorted_dates)
    print(f"[Streaming] Replaying {total} days from {year} …")

    for i, date in enumerate(sorted_dates, 1):
        batch = stream_data[stream_data["Date"] == date].copy()
        print(
            f"  Batch {i:>3}/{total}  {pd.Timestamp(date).date()}"
            f"  — {len(batch)} SKU rows"
        )
        yield pd.Timestamp(date), batch

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
