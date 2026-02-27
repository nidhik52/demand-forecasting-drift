"""
Kafka Consumer — Real-Time Drift Detection & Auto-Retraining
=============================================================
Project : Drift-Aware Continuous Learning Framework
File    : kafka/consumer.py

Subscribes to the 'sales_events' Kafka topic, aggregates daily demand
per category in real time, and feeds each completed day to the
DriftDetectorRegistry.  When drift is confirmed over 3 consecutive
days, RetrainPipeline is triggered automatically — identical to the
batch CI/CD pipeline but driven by a live event stream.

Architecture
------------
  [Kafka topic 'sales_events']
       ↓  (1 msg = 1 retail transaction)
  [consumer.py]
       → aggregate by date + category_group
       → on day-complete: predict with cached Prophet model
       → DriftDetectorRegistry.update(actual, predicted)
       → if retrain_triggered → RetrainPipeline.retrain()
       → reload models/*.pkl after accepted retrain
       → print live dashboard to terminal

Startup sequence
----------------
  1. Load data/processed/final_demand_series.csv  (up to train_end)
  2. Load models/*.pkl   (pre-trained Prophet models)
  3. Compute baseline MAE on val window (last 30 days before stream)
  4. Initialise DriftDetectorRegistry + RetrainPipeline
  5. Poll Kafka — process each completed day in real time

Usage
-----
# Requires Kafka running:
    docker compose -f kafka/docker-compose.kafka.yml up -d

# Then start producer in another terminal:
    python kafka/producer.py --start-date 2025-11-01 --speed 200

# Then run consumer:
    python kafka/consumer.py

# Optional flags:
    python kafka/consumer.py --threshold 1.5 --bootstrap localhost:9092

Options
-------
--bootstrap    Kafka broker address        (default: localhost:9092)
--topic        Kafka topic name            (default: sales_events)
--threshold    Drift MAE ratio threshold   (default: 1.5)
--model-dir    Directory of .pkl models    (default: models/)
--val-days     Days for baseline MAE calc  (default: 30)
--idle-flush   Seconds idle before flushing last day  (default: 5)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ── Allow running from any working directory ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
except ImportError:
    sys.exit("❌  numpy/pandas not installed.  Run: pip install numpy pandas")

try:
    import joblib
except ImportError:
    sys.exit("❌  joblib not installed.  Run: pip install joblib")

try:
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
except ImportError:
    sys.exit("❌  kafka-python not installed.  Run: pip install kafka-python")

try:
    from src.drift_detection.drift_detector import DriftDetectorRegistry
    from src.retraining.retrain_pipeline import RetrainPipeline
except ImportError as e:
    sys.exit(f"❌  Import error: {e}\n    Run from repo root, e.g.:\n    python kafka/consumer.py")

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

CATEGORIES = [
    'Electronics & Tech',
    'Entertainment & Office',
    'Fashion & Accessories',
    'Health & Personal Care',
    'Home & Lifestyle',
    'Sports & Outdoors',
]

def _slug(cat: str) -> str:
    return cat.lower().replace(' & ', '_').replace(' ', '_')


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time drift detection consumer for 'sales_events' Kafka topic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--bootstrap', default='localhost:9092')
    p.add_argument('--topic',     default='sales_events')
    p.add_argument('--group-id',  default='drift-consumer-group')
    p.add_argument('--threshold', type=float, default=1.5,
                   help='Drift threshold (MAE ratio to baseline)')
    p.add_argument('--model-dir', default='models',
                   help='Directory containing .pkl Prophet models')
    p.add_argument('--val-days',  type=int, default=30,
                   help='Days of historical data used for baseline MAE')
    p.add_argument('--idle-flush', type=float, default=5.0,
                   help='Seconds without messages before flushing the current day buffer')
    return p.parse_args()


# ─── Setup ────────────────────────────────────────────────────────────────────

def load_history(val_days: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed demand series and split into train / val / remainder."""
    csv = ROOT / 'data' / 'processed' / 'final_demand_series.csv'
    if not csv.exists():
        sys.exit(f"❌  {csv} not found.  Run: python src/data/master_data_pipeline.py")

    df = pd.read_csv(csv, parse_dates=['ds'])
    latest      = df['ds'].max()
    val_end     = latest
    val_start   = val_end   - timedelta(days=val_days - 1)
    train_end   = val_start - timedelta(days=1)

    train_df = df[df['ds'] <= train_end].copy()
    val_df   = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)].copy()

    print(f"  Historical data : {csv.relative_to(ROOT)}")
    print(f"  Train ends      : {train_end.date()}")
    print(f"  Val window      : {val_start.date()} → {val_end.date()}  ({val_days}d)")
    print(f"  Stream from     : {(latest + timedelta(days=1)).date()} onwards (real-time)")
    return df, train_df, val_df


def load_models(model_dir: str | Path) -> dict[str, Any]:
    """Load all .pkl Prophet models from model_dir."""
    model_dir = ROOT / model_dir
    models: dict[str, Any] = {}
    for cat in CATEGORIES:
        slug = _slug(cat)
        path = model_dir / f'{slug}.pkl'
        if path.exists():
            models[cat] = joblib.load(path)
        else:
            print(f"  {YELLOW}⚠  No cached model for {cat} — skipping{RESET}")
    return models


def compute_baselines(
    models:  dict[str, Any],
    val_df:  pd.DataFrame,
) -> dict[str, float]:
    """Compute baseline MAE per category on the validation window."""
    baselines: dict[str, float] = {}
    for cat, model in models.items():
        cat_val = val_df[val_df['category'] == cat].sort_values('ds')
        if cat_val.empty:
            continue
        forecast    = model.predict(pd.DataFrame({'ds': cat_val['ds']}))
        errors      = np.abs(cat_val['y'].values - forecast['yhat'].values)
        baselines[cat] = float(np.mean(errors))
        print(f"  {cat[:30]:30}  baseline MAE = {baselines[cat]:,.0f}")
    return baselines


def connect_consumer(bootstrap: str, topic: str, group_id: str,
                     retries: int = 5) -> KafkaConsumer:
    for attempt in range(1, retries + 1):
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap,
                group_id=group_id,
                auto_offset_reset='earliest',      # replay from the start of the topic
                enable_auto_commit=True,
                value_deserializer=lambda b: json.loads(b.decode('utf-8')),
                consumer_timeout_ms=1000,          # 1s poll timeout → allows idle detection
            )
            print(f"  {GREEN}✅  Connected to Kafka at {bootstrap}, topic '{topic}'{RESET}")
            return consumer
        except NoBrokersAvailable:
            wait = attempt * 3
            print(f"  {YELLOW}⏳  Kafka not ready (attempt {attempt}/{retries}) — retry in {wait}s{RESET}")
            time.sleep(wait)
    sys.exit(f"{RED}❌  Could not connect to Kafka.  "
             f"Start the stack:  docker compose -f kafka/docker-compose.kafka.yml up -d{RESET}")


# ─── Day processing ───────────────────────────────────────────────────────────

def process_day(
    date:         str,
    day_demand:   dict[str, float],   # category → total sales for the day
    models:       dict[str, Any],
    registry:     DriftDetectorRegistry,
    baselines:    dict[str, float],   # category → baseline MAE (for retrain pre_mae)
    full_history: pd.DataFrame,
    pipeline:     RetrainPipeline,
    model_dir:    Path,
) -> dict[str, Any]:
    """
    Feed one completed day to the drift detector.

    For each category: compare the day's actual demand with the model's
    prediction, then call registry.update().  If a retrain is triggered,
    run RetrainPipeline and reload the updated model.

    Returns a summary dict with per-category drift status.
    """
    results: dict[str, Any] = {}

    for cat in CATEGORIES:
        if cat not in models:
            continue

        model   = models[cat]
        actual  = day_demand.get(cat, 0.0)
        ds_ts   = pd.Timestamp(date)

        try:
            forecast  = model.predict(pd.DataFrame({'ds': [ds_ts]}))
            predicted = float(forecast['yhat'].values[0])
        except Exception:
            predicted = actual   # fallback — no error penalty

        status = registry.update(cat, actual, predicted, date)

        drift_symbol = f"{RED}🔴{RESET}" if status.is_drifting else f"{GREEN}✅{RESET}"
        results[cat] = {
            'actual'    : actual,
            'predicted' : predicted,
            'short_ratio': status.short_ratio,
            'long_ratio' : status.long_ratio,
            'is_drifting': status.is_drifting,
            'retrain_triggered': status.retrain_triggered,
        }

        # ── Auto-retrain on trigger ─────────────────────────────────────────
        if status.retrain_triggered:
            print(f"\n  {BOLD}{YELLOW}⚡ RETRAIN TRIGGERED  {cat}  ({date}){RESET}")
            try:
                cat_history = full_history[full_history['category'] == cat].copy()
                result, new_model = pipeline.retrain(
                    category        = cat,
                    retrain_date    = date,
                    current_model   = model,
                    pre_retrain_mae = baselines.get(cat, 0.0),
                    trigger_reason  = 'kafka_drift_detected',
                )
                if result.model_accepted and new_model is not None:
                    # Persist and reload so the next prediction uses the new model
                    slug      = _slug(cat)
                    save_path = model_dir / f'{slug}.pkl'
                    joblib.dump(new_model, save_path)
                    models[cat] = new_model
                    decision = f"{GREEN}ACCEPTED{RESET}  new MAE {result.post_retrain_mae:,.0f}"
                    # Update detector baseline to new model's performance
                    registry.reset(cat, new_baseline_mae=result.post_retrain_mae)
                else:
                    decision = f"{RED}REJECTED{RESET}  old model retained"
                print(f"     Decision : {decision}")
            except Exception as exc:
                print(f"     {RED}Retrain failed: {exc}{RESET}")

    return results


def print_day_summary(date: str, results: dict[str, Any]) -> None:
    """Print a compact real-time dashboard line for a completed day."""
    parts = []
    for cat in CATEGORIES:
        if cat not in results:
            continue
        r    = results[cat]
        sym  = "🔴" if r['is_drifting'] else "✅"
        abbr = cat.split()[0][:4]               # e.g. "Elec", "Heal"
        parts.append(f"{sym}{abbr} {r['short_ratio']:.2f}x")

    bar = "  |  ".join(parts)
    print(f"  {CYAN}{date}{RESET}  {bar}")


# ─── Main loop ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print(f"\n{BOLD}{'='*64}")
    print(" Real-Time Demand Drift Consumer")
    print(f"{'='*64}{RESET}\n")

    # ── 1. Load history + models + baselines ──────────────────────────────────
    print(f"{BOLD}[1/4] Loading historical context{RESET}")
    full_history, train_df, val_df = load_history(val_days=args.val_days)

    print(f"\n{BOLD}[2/4] Loading pre-trained models from '{args.model_dir}'{RESET}")
    models = load_models(args.model_dir)
    if not models:
        sys.exit("❌  No .pkl models found.  "
                 "Run: python src/drift_detection/run_drift_check.py --days 61 --threshold 1.5 "
                 "--model-dir models/ --retrain")

    print(f"\n{BOLD}[3/4] Computing baseline MAE on validation window{RESET}")
    baselines = compute_baselines(models, val_df)

    registry = DriftDetectorRegistry(
        baseline_maes = baselines,
        threshold     = args.threshold,
        short_window  = 7,
        long_window   = 30,
        min_days      = 3,
    )

    pipeline = RetrainPipeline(
        demand_data     = full_history,
        window_days     = 45,
        holdout_days    = 14,
        mlflow_tracking = True,
        experiment_name = 'demand_forecasting_drift_kafka',
        model_dir       = args.model_dir,
    )

    # ── 2. Connect to Kafka ───────────────────────────────────────────────────
    print(f"\n{BOLD}[4/4] Connecting to Kafka{RESET}")
    consumer = connect_consumer(args.bootstrap, args.topic, args.group_id)

    print(f"\n{BOLD}{'─'*64}")
    print(" Streaming  —  waiting for messages …")
    print(f" Open http://localhost:8080 to browse the topic in Kafka UI")
    print(f"{'─'*64}{RESET}\n")

    model_dir        = ROOT / args.model_dir
    current_date: str | None = None
    day_buffer: dict[str, float] = defaultdict(float)
    last_msg_time    = time.time()
    days_processed   = 0

    try:
        while True:
            # ── Poll with 1s timeout (raises StopIteration when idle) ─────────
            msgs_this_poll = 0
            try:
                for msg in consumer:
                    data = msg.value
                    date = data.get('date', '')
                    cat  = data.get('category_group', '')
                    amt  = float(data.get('sales_amount', 0.0))

                    if not date or cat == 'Other' or cat not in CATEGORIES:
                        continue

                    if date != current_date:
                        # Day transition — process the completed day
                        if current_date is not None and day_buffer:
                            results = process_day(
                                current_date, dict(day_buffer),
                                models, registry, baselines,
                                full_history, pipeline, model_dir,
                            )
                            print_day_summary(current_date, results)
                            days_processed += 1
                        current_date = date
                        day_buffer   = defaultdict(float)

                    day_buffer[cat] += amt
                    last_msg_time    = time.time()
                    msgs_this_poll  += 1

            except StopIteration:
                # consumer_timeout_ms expired — no messages in this poll
                pass

            # ── Idle flush: no messages for idle_flush seconds ────────────────
            idle = time.time() - last_msg_time
            if idle >= args.idle_flush and current_date is not None and day_buffer:
                print(f"\n  {YELLOW}⏱  Idle {idle:.1f}s — flushing last day {current_date}{RESET}")
                results = process_day(
                    current_date, dict(day_buffer),
                    models, registry, baselines,
                    full_history, pipeline, model_dir,
                )
                print_day_summary(current_date, results)
                days_processed += 1
                current_date = None
                day_buffer   = defaultdict(float)
                last_msg_time = time.time()  # reset so we don't flush again immediately

            # ── Exit if idle for a long time with no new data ─────────────────
            if idle > 30 and days_processed > 0 and msgs_this_poll == 0:
                print(f"\n  {YELLOW}Stream idle >30s with {days_processed} days processed — exiting.{RESET}")
                break

            time.sleep(0.1)  # prevent busy-wait

    except KeyboardInterrupt:
        # Flush whatever is still buffered
        if current_date and day_buffer:
            print(f"\n⛔  Interrupted — flushing {current_date}…")
            results = process_day(
                current_date, dict(day_buffer),
                models, registry, baselines,
                full_history, pipeline, model_dir,
            )
            print_day_summary(current_date, results)
        print("\n⛔  Consumer stopped.\n")

    finally:
        consumer.close()
        # ── Final summary ──────────────────────────────────────────────────────
        print(f"\n{BOLD}{'─'*64}")
        print(f" Summary — {days_processed} days processed")
        print(f"{'─'*64}{RESET}")
        all_summaries = registry.get_all_summaries()
        for cat in CATEGORIES:
            if cat not in all_summaries:
                continue
            s = all_summaries[cat]
            drift_flag = f"{RED}DRIFT{RESET}" if s.get('drift_days', 0) > 0 else f"{GREEN}CLEAN{RESET}"
            print(f"  {cat[:30]:30}  short={s['short_ratio']:.2f}x  "
                  f"long={s['long_ratio']:.2f}x  {drift_flag}")

        print(f"\n  MLflow experiment : 'demand_forecasting_drift_kafka'")
        print(f"  View UI           : mlflow ui --backend-store-uri mlruns/\n")


if __name__ == '__main__':
    main()
