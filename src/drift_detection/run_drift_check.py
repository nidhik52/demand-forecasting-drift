"""
run_drift_check.py — Standalone CI/CD Drift Detection Script
=============================================================
Project : Drift-Aware Continuous Learning Framework
File    : src/drift_detection/run_drift_check.py

PURPOSE
-------
This script is called by GitHub Actions (.github/workflows/drift_pipeline.yml).
It runs the full drift detection pipeline headlessly (no Jupyter needed):

  1. Load latest demand data
  2. Load or train Prophet models
  3. Run drift detector on most recent 30 days
  4. If drift detected: trigger retraining, log to MLflow
  5. Output drift report as JSON (uploaded as GitHub Actions artifact)
  6. Exit code 0 = no drift, Exit code 1 = drift detected (triggers email)

This is the PRODUCTION version of what the notebooks demonstrate.

USAGE
-----
  python src/drift_detection/run_drift_check.py
  python src/drift_detection/run_drift_check.py --threshold 2.0
  python src/drift_detection/run_drift_check.py --days 30 --threshold 2.0
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s  %(levelname)s  %(message)s',
    datefmt= '%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# ── Add project root to path so src.* imports work ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='Drift detection check for CI/CD pipeline')
    parser.add_argument('--data',      default='data/processed/final_demand_series.csv',
                        help='Path to processed demand CSV')
    parser.add_argument('--days',      type=int, default=30,
                        help='Number of recent days to evaluate (default: 30)')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='MAE ratio threshold for drift flag (default: 2.0)')
    parser.add_argument('--output',    default='reports/drift_report.json',
                        help='Path to save drift report JSON')
    parser.add_argument('--retrain',   action='store_true',
                        help='Trigger retraining if drift detected')
    return parser.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    """Load and validate the demand time series."""
    path = PROJECT_ROOT / data_path
    if not path.exists():
        log.error(f"Data file not found: {path}")
        sys.exit(2)
    df = pd.read_csv(path)
    df['ds'] = pd.to_datetime(df['ds'])
    log.info(f"Loaded {len(df):,} rows from {path.name}")
    log.info(f"Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    return df


def train_prophet(train_df: pd.DataFrame, category: str):
    """Train a Prophet model for one category."""
    try:
        from prophet import Prophet
    except ImportError:
        log.error("Prophet not installed. Run: pip install prophet")
        sys.exit(2)

    import logging as _log
    _log.getLogger('cmdstanpy').setLevel(_log.WARNING)

    model = Prophet(
        yearly_seasonality      = 'auto',
        weekly_seasonality      = 'auto',
        daily_seasonality       = 'auto',
        seasonality_mode        = 'multiplicative',
        changepoint_prior_scale = 0.05,
        interval_width          = 0.95,
    )
    model.fit(train_df[['ds', 'y']])
    return model


def compute_rolling_mae(errors: list, window: int) -> float:
    """Rolling MAE over last `window` errors."""
    recent = errors[-window:] if len(errors) >= window else errors
    return float(np.mean(recent)) if recent else 0.0


def run_drift_check(df: pd.DataFrame, args) -> dict:
    """
    Main drift detection logic.
    Returns a drift report dict.
    """
    categories   = sorted(df['category'].unique())
    report       = {
        'run_date'        : datetime.now().isoformat(),
        'threshold'       : args.threshold,
        'eval_days'       : args.days,
        'drift_detected'  : False,
        'categories'      : {},
        'summary'         : '',
    }

    # ── Determine evaluation window ──────────────────────────────────────────
    latest_date  = df['ds'].max()
    eval_start   = latest_date - timedelta(days=args.days)
    train_end    = eval_start - timedelta(days=1)

    log.info(f"Train window: up to {train_end.date()}")
    log.info(f"Eval window : {eval_start.date()} to {latest_date.date()} ({args.days} days)")
    log.info(f"Threshold   : {args.threshold}x baseline MAE")
    log.info(f"Categories  : {len(categories)}")
    log.info("-" * 55)

    drifted_cats = []

    for cat in categories:
        cat_df = df[df['category'] == cat].sort_values('ds')

        train_data = cat_df[cat_df['ds'] <= train_end][['ds', 'y']].reset_index(drop=True)
        eval_data  = cat_df[cat_df['ds'] >  train_end].reset_index(drop=True)

        if len(train_data) < 60:
            log.warning(f"  {cat[:20]:20} — insufficient training data ({len(train_data)} days), skipping")
            continue
        if len(eval_data) == 0:
            log.warning(f"  {cat[:20]:20} — no evaluation data, skipping")
            continue

        # Train model
        log.info(f"  Training {cat[:25]}...")
        model = train_prophet(train_df=train_data, category=cat)

        # Forecast eval window
        future   = pd.DataFrame({'ds': eval_data['ds']})
        forecast = model.predict(future)

        # Compute errors
        actuals    = eval_data['y'].values
        predicted  = forecast['yhat'].values
        errors     = np.abs(actuals - predicted).tolist()

        # Baseline MAE (first 7 days of eval as warmup, rest as signal)
        if len(errors) >= 14:
            baseline_errors = errors[:7]
            signal_errors   = errors[7:]
        else:
            baseline_errors = errors[:max(1, len(errors)//2)]
            signal_errors   = errors[len(baseline_errors):]

        baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else 0.0
        recent_mae   = compute_rolling_mae(signal_errors, window=7)
        ratio        = (recent_mae / baseline_mae) if baseline_mae > 0 else 0.0
        is_drifting  = ratio > args.threshold

        log.info(
            f"  {'✅' if not is_drifting else '🔴'} {cat[:25]:25} "
            f"baseline={baseline_mae:>7,.0f}  recent={recent_mae:>7,.0f}  "
            f"ratio={ratio:>5.2f}x  {'DRIFT' if is_drifting else 'OK'}"
        )

        report['categories'][cat] = {
            'baseline_mae': round(baseline_mae, 1),
            'recent_mae'  : round(recent_mae, 1),
            'ratio'       : round(ratio, 3),
            'is_drifting' : is_drifting,
            'eval_days'   : len(eval_data),
        }

        if is_drifting:
            drifted_cats.append(cat)

    # ── Summary ──────────────────────────────────────────────────────────────
    report['drift_detected']  = len(drifted_cats) > 0
    report['drifted_categories'] = drifted_cats
    report['summary'] = (
        f"Drift detected in {len(drifted_cats)}/{len(categories)} categories: "
        f"{', '.join(drifted_cats)}"
        if drifted_cats
        else f"No drift detected across {len(categories)} categories (threshold={args.threshold}x)"
    )

    log.info("-" * 55)
    log.info(report['summary'])
    return report


def save_report(report: dict, output_path: str):
    """Save drift report as JSON artifact."""
    path = PROJECT_ROOT / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {path}")


def trigger_retraining(drifted_cats: list, df: pd.DataFrame):
    """
    Trigger retraining for drifted categories.
    Imports and uses RetrainPipeline.
    """
    try:
        from src.retraining.retrain_pipeline import RetrainPipeline
    except ImportError:
        log.error("RetrainPipeline not found — skipping retraining")
        return

    pipeline = RetrainPipeline(
        demand_data     = df,
        window_days     = 45,
        holdout_days    = 14,
        mlflow_tracking = True,
    )

    for cat in drifted_cats:
        log.info(f"Retraining {cat}...")
        cat_df    = df[df['category'] == cat].sort_values('ds')
        today_str = str(cat_df['ds'].max().date())

        # Dummy model for pipeline (will retrain from scratch)
        try:
            from prophet import Prophet
            from src.forecasting.prophet_model import ProphetForecaster
            forecaster = ProphetForecaster()
            forecaster.fit(cat_df[cat_df['ds'] <= cat_df['ds'].max()-timedelta(days=45)])
            current_model = forecaster.model
        except Exception as e:
            log.warning(f"  Could not load current model: {e}")
            current_model = None

        result, _ = pipeline.retrain(
            category        = cat,
            retrain_date    = today_str,
            current_model   = current_model,
            pre_retrain_mae = 9999,
            trigger_reason  = 'ci_cd_drift_check',
        )
        log.info(f"  Retrain result: {result}")


def main():
    args   = parse_args()
    log.info("=" * 55)
    log.info("DRIFT DETECTION CHECK — CI/CD Pipeline")
    log.info("=" * 55)

    # Load data
    df = load_data(args.data)

    # Run drift check
    report = run_drift_check(df, args)

    # Save report
    save_report(report, args.output)

    # Trigger retraining if requested and drift found
    if args.retrain and report['drift_detected']:
        log.info("\nTriggering retraining for drifted categories...")
        trigger_retraining(report['drifted_categories'], df)

    # Exit code: 0 = clean, 1 = drift detected
    # GitHub Actions uses exit code to decide next steps
    exit_code = 1 if report['drift_detected'] else 0
    log.info(f"\nExit code: {exit_code} ({'DRIFT DETECTED' if exit_code else 'CLEAN'})")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
