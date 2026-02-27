"""
Kafka Producer — Real-Time Sales Event Stream
==============================================
Project : Drift-Aware Continuous Learning Framework
File    : kafka/producer.py

Streams historical retail transactions from sales_100k_generated.csv
to the Kafka topic 'sales_events', row by row, simulating a live POS
(point-of-sale) feed.

Each message is a JSON-encoded transaction:
  {
    "date"            : "2025-11-01",
    "product_category": "Electronics",
    "category_group"  : "Electronics & Tech",
    "sales_amount"    : 1234.56,
    "region"          : "North",
    "row_index"       : 42
  }

Usage
-----
# Start Kafka first:
    docker compose -f kafka/docker-compose.kafka.yml up -d

# Stream all 100k transactions at 200 rows/sec (default):
    python kafka/producer.py

# Stream only from the drift window at faster speed:
    python kafka/producer.py --start-date 2025-11-01 --speed 500

# Stream slowly to watch each day tick by:
    python kafka/producer.py --start-date 2025-11-01 --speed 10

Options
-------
--bootstrap   Kafka broker address          (default: localhost:9092)
--topic       Kafka topic name              (default: sales_events)
--speed       Rows per second to publish    (default: 200)
--start-date  Only stream rows on/after this date  (default: all data)
--data        Path to the raw CSV           (default: auto-detected)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Allow running from any working directory ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError:
    sys.exit("❌  pandas not installed.  Run: pip install pandas")

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
except ImportError:
    sys.exit("❌  kafka-python not installed.  Run: pip install kafka-python")

# ── Category mapping (mirrors master_data_pipeline.py) ───────────────────────
CATEGORY_MAP: dict[str, str] = {
    'Electronics'    : 'Electronics & Tech',
    'Software'       : 'Electronics & Tech',
    'Appliances'     : 'Electronics & Tech',
    'Health'         : 'Health & Personal Care',
    'Beauty'         : 'Health & Personal Care',
    'Personal Care'  : 'Health & Personal Care',
    'Baby Products'  : 'Health & Personal Care',
    'Home & Kitchen' : 'Home & Lifestyle',
    'Furniture'      : 'Home & Lifestyle',
    'Garden'         : 'Home & Lifestyle',
    'DIY'            : 'Home & Lifestyle',
    'Tools'          : 'Home & Lifestyle',
    'Groceries'      : 'Home & Lifestyle',
    'Sports'         : 'Sports & Outdoors',
    'Outdoor'        : 'Sports & Outdoors',
    'Automotive'     : 'Sports & Outdoors',
    'Clothing'       : 'Fashion & Accessories',
    'Jewelry'        : 'Fashion & Accessories',
    'Movies'         : 'Entertainment & Office',
    'Music'          : 'Entertainment & Office',
    'Books'          : 'Entertainment & Office',
    'Toys'           : 'Entertainment & Office',
    'Office Supplies': 'Entertainment & Office',
    'Pet Supplies'   : 'Entertainment & Office',
}

# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream retail transactions to Kafka topic 'sales_events'",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--bootstrap', default='localhost:9092',
                   help='Kafka broker address')
    p.add_argument('--topic',     default='sales_events',
                   help='Kafka topic to publish to')
    p.add_argument('--speed',     type=float, default=200,
                   help='Rows per second to publish (increase for faster simulation)')
    p.add_argument('--start-date', default=None,
                   help='Only stream transactions on or after this date (YYYY-MM-DD). '
                        'Default: stream from the beginning of the dataset.')
    p.add_argument('--data',      default=None,
                   help='Path to sales_100k_generated.csv. Auto-detected if omitted.')
    p.add_argument('--dry-run',   action='store_true',
                   help='Print messages to stdout without connecting to Kafka')
    return p.parse_args()


def find_data(override: str | None) -> Path:
    if override:
        return Path(override)
    candidates = [
        ROOT / 'data' / 'raw' / 'sales_100k_generated.csv',
        ROOT / 'data' / 'raw' / 'sales_100k.csv',
    ]
    for c in candidates:
        if c.exists():
            return c
    sys.exit(f"❌  Raw CSV not found. Run:  python src/data/master_data_pipeline.py")


def connect(bootstrap: str, retries: int = 5) -> KafkaProducer:
    """Connect to Kafka, retrying if the broker is still starting up."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3,
            )
            print(f"✅  Connected to Kafka at {bootstrap}")
            return producer
        except NoBrokersAvailable:
            wait = attempt * 3
            print(f"⏳  Kafka not ready (attempt {attempt}/{retries}) — retrying in {wait}s…")
            time.sleep(wait)
    sys.exit("❌  Could not connect to Kafka after multiple retries.\n"
             "    Make sure the stack is up:  "
             "docker compose -f kafka/docker-compose.kafka.yml up -d")


def main() -> None:
    args = parse_args()
    data_path = find_data(args.data)

    print(f"📂  Loading: {data_path.relative_to(ROOT)}")
    df = pd.read_csv(data_path)
    df['Date_of_Sale'] = pd.to_datetime(df['Date_of_Sale'])
    df = df.sort_values('Date_of_Sale').reset_index(drop=True)

    # Apply start-date filter if requested
    if args.start_date:
        cutoff = pd.Timestamp(args.start_date)
        df = df[df['Date_of_Sale'] >= cutoff].reset_index(drop=True)
        print(f"📅  Streaming from {args.start_date} — {len(df):,} transactions")
    else:
        print(f"📅  Streaming full dataset — {len(df):,} transactions")

    # Fill missing sales amounts with column median (mirrors pipeline)
    df['Sales_Amount'] = df['Sales_Amount'].fillna(df['Sales_Amount'].median())

    # Pre-compute category groups
    df['category_group'] = df['Product_Category'].map(CATEGORY_MAP).fillna('Other')

    date_range = f"{df['Date_of_Sale'].min().date()} → {df['Date_of_Sale'].max().date()}"
    print(f"📆  Date range:   {date_range}")
    print(f"⚡  Speed:        {args.speed:,.0f} rows/sec")
    print(f"⏱️   Est. duration: ~{len(df)/args.speed:.0f}s")
    print(f"📡  Topic:        {args.topic}")
    print()

    if args.dry_run:
        print("🔍  DRY RUN — first 5 messages:")
        for _, row in df.head(5).iterrows():
            msg = _build_message(row)
            print(f"    {json.dumps(msg)}")
        print("  (no Kafka connection made)")
        return

    producer = connect(args.bootstrap)

    sleep_s    = 1.0 / args.speed
    total      = len(df)
    sent       = 0
    current_day: str | None = None
    day_count  = 0

    print("🚀  Streaming started. Press Ctrl+C to stop.\n")
    try:
        for idx, row in df.iterrows():
            day = str(row['Date_of_Sale'].date())
            if day != current_day:
                current_day = day
                day_count += 1
                # Print a summary line at each new day
                pct = sent / total * 100
                print(f"\r    Day {day_count:>4}  {day}  |  {sent:>7,}/{total:,} sent  ({pct:5.1f}%)",
                      end='', flush=True)

            msg = _build_message(row)
            producer.send(args.topic, value=msg)
            sent += 1
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n⛔  Interrupted by user.")

    producer.flush()
    producer.close()
    print(f"\n✅  Done. Sent {sent:,} messages across {day_count} days to '{args.topic}'.")


def _build_message(row: 'pd.Series') -> dict:  # type: ignore[type-arg]
    return {
        'date'            : str(row['Date_of_Sale'].date()),
        'product_category': str(row['Product_Category']),
        'category_group'  : str(row['category_group']),
        'sales_amount'    : float(row['Sales_Amount']),
        'region'          : str(row.get('Sales_Region', '')),
        'row_index'       : int(row.name),  # type: ignore[arg-type]
    }


if __name__ == '__main__':
    main()
