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
import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _category_slug(category: str) -> str:
    """Convert 'Electronics & Tech' -> 'electronics_tech' for use as filename."""
    return re.sub(r'[^a-z0-9]+', '_', category.lower()).strip('_')

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
    parser.add_argument('--threshold', type=float, default=1.5,
                        help='MAE ratio threshold for drift flag (default: 1.5, matches DriftDetector)')
    parser.add_argument('--output',    default='reports/drift_report.json',
                        help='Path to save drift report JSON')
    parser.add_argument('--retrain',   action='store_true',
                        help='Trigger retraining if drift detected')
    parser.add_argument('--model-dir', default='models',
                        help='Directory to load/save trained Prophet models (default: models/)')
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


def run_drift_check(df: pd.DataFrame, args) -> dict:
    """
    Walk-forward drift detection using DriftDetectorRegistry
    (dual-window: 7-day short + 30-day long, min_days=3 consecutive).
    Identical logic to notebooks/03_drift_detection.ipynb and the Airflow DAG.
    Returns a drift report dict.
    """
    from src.drift_detection.drift_detector import DriftDetectorRegistry

    categories = sorted(df['category'].unique())
    report = {
        'run_date'           : datetime.now().isoformat(),
        'threshold'          : args.threshold,
        'eval_days'          : args.days,
        'drift_detected'     : False,
        'drifted_categories' : [],
        'categories'         : {},
        'summary'            : '',
    }

    # ── Determine windows ─────────────────────────────────────────────────────
    # Eval: last args.days of data
    # Val:  30 days immediately before eval → used to compute baseline MAE
    # Train: everything before val
    latest_date = df['ds'].max()
    eval_start  = latest_date - timedelta(days=args.days - 1)
    val_days    = 30
    val_end     = eval_start - timedelta(days=1)
    val_start   = val_end   - timedelta(days=val_days - 1)
    train_end   = val_start - timedelta(days=1)

    log.info(f"Train window : up to {train_end.date()}")
    log.info(f"Val window   : {val_start.date()} to {val_end.date()} ({val_days}d — baseline MAE)")
    log.info(f"Eval window  : {eval_start.date()} to {latest_date.date()} ({args.days}d)")
    log.info(f"Detector     : dual-window 7d/30d  |  threshold={args.threshold}x  |  min_days=3")
    log.info(f"Categories   : {len(categories)}")
    log.info("-" * 60)

    baseline_maes  = {}
    trained_models = {}

    # ── Phase 1: Train Prophet + compute baseline MAE per category ────────────
    for cat in categories:
        cat_df     = df[df['category'] == cat].sort_values('ds')
        train_data = cat_df[cat_df['ds'] <= train_end][['ds', 'y']].reset_index(drop=True)
        val_data   = cat_df[(cat_df['ds'] >= val_start) & (cat_df['ds'] <= val_end)].reset_index(drop=True)

        if len(train_data) < 60:
            log.warning(f"  {cat[:25]:25} — insufficient training data ({len(train_data)} days), skipping")
            continue
        if len(val_data) == 0:
            log.warning(f"  {cat[:25]:25} — no validation data, skipping")
            continue

        slug       = _category_slug(cat)
        model_path = PROJECT_ROOT / args.model_dir / f'{slug}.pkl'
        if model_path.exists():
            log.info(f"  Loading    {cat[:30]} from cache ({model_path.name})")
            model = joblib.load(model_path)
        else:
            log.info(f"  Training   {cat[:30]} (no cache)...")
            model = train_prophet(train_df=train_data, category=cat)
            # Save freshly trained model so next run can skip training
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            log.info(f"             saved → {model_path.name}")

        val_forecast = model.predict(pd.DataFrame({'ds': val_data['ds']}))
        val_errors   = np.abs(val_data['y'].values - val_forecast['yhat'].values)
        baseline_mae = float(np.mean(val_errors))

        baseline_maes[cat]  = baseline_mae
        trained_models[cat] = model
        log.info(f"             baseline MAE = {baseline_mae:,.0f}")

    if not baseline_maes:
        log.error("No categories had sufficient data. Cannot run drift check.")
        sys.exit(2)

    # ── Phase 2: Walk-forward eval with DriftDetectorRegistry ────────────────
    registry = DriftDetectorRegistry(
        baseline_maes = baseline_maes,
        threshold     = args.threshold,   # 1.5x — matches notebooks and DAG
        short_window  = 7,
        long_window   = 30,
        min_days      = 3,
    )

    drifted_cats = []

    for cat in baseline_maes:
        cat_df    = df[df['category'] == cat].sort_values('ds')
        eval_data = cat_df[cat_df['ds'] >= eval_start].reset_index(drop=True)

        if len(eval_data) == 0:
            continue

        eval_forecast = trained_models[cat].predict(pd.DataFrame({'ds': eval_data['ds']}))
        # Build a fast lookup: ds → yhat
        yhat_map = dict(zip(eval_forecast['ds'], eval_forecast['yhat']))

        drift_days   = 0
        retrain_days = []

        for _, row in eval_data.iterrows():
            actual    = float(row['y'])
            predicted = float(yhat_map.get(row['ds'], actual))
            date_str  = str(row['ds'].date())

            status = registry.update(cat, actual, predicted, date_str)
            if status.is_drifting:
                drift_days += 1
            if status.retrain_triggered:
                retrain_days.append(date_str)

        n_eval      = len(eval_data)
        cat_summary = registry.get_all_summaries()[cat]
        is_drifting = drift_days > 0

        log.info(
            f"  {'✅' if not is_drifting else '🔴'} {cat[:25]:25} "
            f"baseline={baseline_maes[cat]:>7,.0f}  "
            f"short={cat_summary['short_ratio']:>4.2f}x  "
            f"long={cat_summary['long_ratio']:>4.2f}x  "
            f"drift={drift_days}/{n_eval}d  "
            f"retrains={len(retrain_days)}"
        )

        report['categories'][cat] = {
            'baseline_mae' : round(baseline_maes[cat], 1),
            'short_ratio'  : round(cat_summary['short_ratio'], 3),
            'long_ratio'   : round(cat_summary['long_ratio'], 3),
            'drift_days'   : drift_days,
            'eval_days'    : n_eval,
            'retrain_days' : retrain_days,
            'is_drifting'  : is_drifting,
        }

        if is_drifting:
            drifted_cats.append(cat)

    # ── Summary ───────────────────────────────────────────────────────────────
    report['drift_detected']     = len(drifted_cats) > 0
    report['drifted_categories'] = drifted_cats
    report['summary'] = (
        f"Drift detected in {len(drifted_cats)}/{len(categories)} categories: "
        f"{', '.join(drifted_cats)}"
        if drifted_cats
        else f"No drift detected across {len(categories)} categories (threshold={args.threshold}x)"
    )

    log.info("-" * 60)
    log.info(report['summary'])
    return report


def save_report(report: dict, output_path: str):
    """Save drift report as JSON artifact."""
    path = PROJECT_ROOT / output_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved: {path}")


def trigger_retraining(drifted_cats: list, df: pd.DataFrame, model_dir: str = 'models'):
    """
    Trigger retraining for drifted categories.
    Saves accepted models to model_dir so next CI run can skip training.
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
        model_dir       = model_dir,
    )

    for cat in drifted_cats:
        log.info(f"Retraining {cat}...")
        cat_df    = df[df['category'] == cat].sort_values('ds')
        today_str = str(cat_df['ds'].max().date())

        # Load current model from cache if available
        slug       = _category_slug(cat)
        model_path = PROJECT_ROOT / model_dir / f'{slug}.pkl'
        try:
            current_model = joblib.load(model_path) if model_path.exists() else None
        except Exception:
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
        trigger_retraining(report['drifted_categories'], df, model_dir=args.model_dir)

    # Exit code: 0 = clean, 1 = drift detected
    # GitHub Actions uses exit code to decide next steps
    exit_code = 1 if report['drift_detected'] else 0
    log.info(f"\nExit code: {exit_code} ({'DRIFT DETECTED' if exit_code else 'CLEAN'})")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
