"""
pipeline.py
───────────────────────────────────────────────────────────────────────────────
End-to-end orchestrator for the Drift-Aware Continuous Learning Framework
for Demand Forecasting and Inventory Replenishment.

Pipeline stages
───────────────
  1. Preprocess  — clean raw data → data/processed/daily_demand.csv
  2. Forecast    — train Prophet per SKU → models/ + forecast_2026.csv
  3. Inventory   — safety stock / reorder point → inventory_recommendations.csv
  4. Stream      — replay 2025 data day-by-day
                 → drift detection → auto-retrain → MLflow logging

Usage
─────
  # Run the full pipeline
  python pipeline.py

  # Run a single stage
  python pipeline.py --step preprocess
  python pipeline.py --step forecast
  python pipeline.py --step inventory
  python pipeline.py --step stream
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on the Python path (safe to run from any directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_preprocessing import preprocess
from src.forecasting        import run_forecasting
from src.inventory          import compute_inventory
from src.streaming          import stream_daily_batches, get_forecast_for_date
from src.drift_detection    import DriftDetector
from src.retraining         import retrain

# ── Helpers ───────────────────────────────────────────────────────────────────

def print_stage(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ── Stage 1: Data Preprocessing ───────────────────────────────────────────────

def step_preprocess() -> None:
    print_stage("STAGE 1 — DATA PREPROCESSING")
    daily = preprocess(save=True)
    print(f"  Result: {len(daily):,} rows  |  {daily['SKU'].nunique()} SKUs")


# ── Stage 2: Demand Forecasting ───────────────────────────────────────────────

def step_forecast() -> tuple[dict, object]:
    print_stage("STAGE 2 — PROPHET FORECASTING (2026)")
    models, forecast_df = run_forecasting(save=True)
    print(f"  Result: {len(forecast_df):,} forecast rows across {len(models)} SKUs")
    return models, forecast_df


# ── Stage 3: Inventory Decision Engine ───────────────────────────────────────

def step_inventory(forecast_df=None) -> None:
    print_stage("STAGE 3 — INVENTORY DECISION ENGINE")
    recs = compute_inventory(forecast_df=forecast_df, save=True)
    flagged = (recs["Recommended_Order_Qty"] > 0).sum()
    print(f"  Result: {flagged} / {len(recs)} SKUs need replenishment.")


# ── Stage 4: Streaming → Drift Detection → Auto-Retrain ──────────────────────

def step_stream() -> None:
    print_stage("STAGE 4 — STREAMING · DRIFT DETECTION · AUTO-RETRAIN")

    detector       = DriftDetector()   # stateful rolling-MAE monitor
    retrained_skus: set[str] = set()   # prevent retrain storms (once per run)

    batches = list(stream_daily_batches())
    total   = len(batches)
    for i, (date, batch) in enumerate(batches, 1):
        print(f"  Streaming day {i} / {total} — {date.date()}")

        # Retrieve the pre-generated forecast for this date
        forecast_day = get_forecast_for_date(date)
        forecast_map = dict(
            zip(forecast_day["SKU"], forecast_day["forecast_demand"])
        )

        # Process each SKU row in today's batch
        for _, row in batch.iterrows():
            sku      = row["SKU"]
            actual   = float(row["demand"])
            if sku not in forecast_map:
                print(f"  Warning: no forecast for SKU {sku} on {date.date()} — using 0.0")
            forecast = float(forecast_map.get(sku, 0.0))

            result = detector.update(
                sku=sku, date=date, actual=actual, forecast=forecast
            )

            # ── Drift → Retrain ───────────────────────────────────────────────
            if result["drift_detected"] and sku not in retrained_skus:
                print(
                    f"\n  ⚠  DRIFT detected on {sku}  "
                    f"rolling_MAE={result['rolling_mae']:.1f}  "
                    f"(threshold={detector.threshold})  →  retraining …"
                )
                retrain(
                    sku=sku,
                    data_up_to=date,
                    rolling_mae=result["rolling_mae"],
                )
                # Reset the error buffer so we start fresh after retraining
                detector.reset(sku)
                retrained_skus.add(sku)
                print(f"  ✓  {sku} retrained — forecast updated — run logged to MLflow.")

    print(f"\n  Streaming complete.  Days processed: {total}")
    if retrained_skus:
        print(f"  Retraining summary: {len(retrained_skus)} SKU(s) retrained:")
        for sku in sorted(retrained_skus):
            print(f"    \u2022 {sku}")
    else:
        print("  No retraining events triggered.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(step: str = "all") -> None:
    start_time = time.time()
    print("\n" + "#" * 64)
    print("  Drift-Aware Demand Forecasting Pipeline")
    print("  M.Tech Project — Continuous Learning Framework")
    print("#" * 64)

    forecast_df = None

    if step in ("all", "preprocess"):
        step_preprocess()

    if step in ("all", "forecast"):
        _, forecast_df = step_forecast()

    if step in ("all", "inventory"):
        # Pass the in-memory forecast to avoid re-reading from disk
        step_inventory(forecast_df=forecast_df)

    if step in ("all", "stream"):
        step_stream()

    elapsed = time.time() - start_time
    print(f"\n  Pipeline completed in {elapsed:.1f}s")
    print("\n✅  Pipeline complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drift-Aware Demand Forecasting Pipeline"
    )
    parser.add_argument(
        "--step",
        choices=["all", "preprocess", "forecast", "inventory", "stream"],
        default="all",
        help="Pipeline stage to run  (default: all)",
    )
    args = parser.parse_args()
    main(step=args.step)
