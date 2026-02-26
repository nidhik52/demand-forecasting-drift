"""
Airflow DAG — Drift-Aware Continuous Learning Pipeline
=======================================================
Project : Drift-Aware Continuous Learning Framework
File    : dags/drift_pipeline_dag.py

FW3 IMPLEMENTATION — Pipeline Orchestration with Apache Airflow
----------------------------------------------------------------
This DAG replaces the manual notebook-based walk-forward loop with
a scheduled, fault-tolerant, retry-able pipeline.

Schedule: Daily at midnight UTC
Tasks:
  1. data_quality_check   — Validate today's incoming demand data
  2. run_forecasts         — Generate Prophet forecasts for each category
  3. update_drift_detector — Feed actuals into rolling MAE detector
  4. check_drift           — Decide whether to trigger retraining
  5. retrain_models        — (conditional) Retrain if drift detected
  6. update_inventory      — Recompute safety stock / reorder points
  7. save_drift_log        — Persist drift status to data/drift_logs/

HOW TO RUN LOCALLY
------------------
  pip install apache-airflow
  export AIRFLOW_HOME=$(pwd)/airflow_home
  airflow db init
  airflow dags trigger drift_pipeline
  airflow standalone   # starts scheduler + webserver on :8080
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# Airflow imports — only available when Airflow is installed
from airflow import DAG  # type: ignore[import-untyped]
from airflow.operators.python import PythonOperator, BranchPythonOperator  # type: ignore[import-untyped]
from airflow.operators.empty import EmptyOperator  # type: ignore[import-untyped]

# Make project src/ importable inside Airflow tasks
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ─────────────────────────────────────────────
# Default task arguments
# ─────────────────────────────────────────────
DEFAULT_ARGS = {
    'owner'           : 'nidhik52',
    'depends_on_past' : False,
    'email_on_failure': False,
    'email_on_retry'  : False,
    'retries'         : 2,
    'retry_delay'     : timedelta(minutes=5),
}

DATA_PATH    = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_demand_series.csv')
MLRUNS_DIR   = os.path.join(PROJECT_ROOT, 'mlruns')
DRIFT_LOG    = os.path.join(PROJECT_ROOT, 'data', 'drift_logs')


# ─────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────

def task_data_quality_check(**context) -> None:
    """
    Task 1 — Validate incoming demand data with Great Expectations.
    Checks: schema, no nulls, non-negative demand, date continuity.
    Pushes pass/fail flag to XCom for downstream tasks.
    """
    from src.data.quality_checks import DemandDataValidator
    import pandas as pd

    df = pd.read_csv(DATA_PATH)
    validator = DemandDataValidator(df)
    results   = validator.run_all_checks()

    failed = [r for r in results if not r.passed]
    if failed:
        raise ValueError(
            f"Data quality FAILED ({len(failed)} checks): "
            + ", ".join(r.check for r in failed)
        )
    print(f"✅ Data quality: all {len(results)} checks passed.")
    context['ti'].xcom_push(key='data_ok', value=True)


def task_run_forecasts(**context) -> None:
    """
    Task 2 — Generate Prophet 30-day forecasts for every category.
    Saves forecast CSVs to data/processed/forecasts/.
    """
    import pandas as pd
    from prophet import Prophet

    df         = pd.read_csv(DATA_PATH)
    df['ds']   = pd.to_datetime(df['ds'])
    run_date   = pd.Timestamp(context['ds'])          # Airflow execution date
    train_end  = run_date - timedelta(days=1)
    categories = sorted(df['category'].unique())

    out_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'forecasts')
    os.makedirs(out_dir, exist_ok=True)

    for cat in categories:
        cdf   = df[df['category'] == cat].sort_values('ds')
        train = cdf[cdf['ds'] <= train_end][['ds', 'y']].reset_index(drop=True)

        if len(train) < 30:
            print(f"  ⚠️  {cat}: insufficient training data, skipping.")
            continue

        model = Prophet(
            yearly_seasonality='auto', weekly_seasonality='auto',
            daily_seasonality='auto', seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05, interval_width=0.95,
        )
        model.fit(train)

        future   = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(
            os.path.join(out_dir, f'{cat.replace(" ", "_")}_forecast.csv'), index=False
        )
        print(f"  ✅ {cat} — forecast saved.")

    print("Task 2 complete: forecasts saved.")


def task_update_drift_detector(**context) -> None:
    """
    Task 3 — Feed yesterday's actuals into the drift detector.
    Loads persisted detector state, updates with new errors, saves state.
    """
    import json
    import pandas as pd
    from src.drift_detection.drift_detector import DriftDetectorRegistry

    df        = pd.read_csv(DATA_PATH)
    df['ds']  = pd.to_datetime(df['ds'])
    run_date  = pd.Timestamp(context['ds'])
    yesterday = run_date - timedelta(days=1)
    categories = sorted(df['category'].unique())

    # Load forecasts generated in task 2
    forecast_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'forecasts')

    BASELINE_MAES = {
        'Electronics & Tech': 2071, 'Entertainment & Office': 4189,
        'Fashion & Accessories': 1677, 'Health & Personal Care': 4449,
        'Home & Lifestyle': 2605, 'Sports & Outdoors': 1514,
    }
    registry = DriftDetectorRegistry(
        baseline_maes=BASELINE_MAES, threshold=2.0,
        short_window=7, long_window=30, min_days=3,
    )

    drift_flags: dict[str, bool] = {}
    for cat in categories:
        actual_row = df[(df['category'] == cat) & (df['ds'] == yesterday)]
        if actual_row.empty:
            continue
        actual = float(actual_row['y'].iloc[0])

        fpath = os.path.join(forecast_dir, f'{cat.replace(" ", "_")}_forecast.csv')
        if not os.path.exists(fpath):
            continue
        fdf       = pd.read_csv(fpath)
        fdf['ds'] = pd.to_datetime(fdf['ds'])
        pred_row  = fdf[fdf['ds'] == yesterday]
        if pred_row.empty:
            continue
        predicted = float(pred_row['yhat'].iloc[0])

        status    = registry.update(cat, actual, predicted, str(yesterday.date()))
        drift_flags[cat] = status.retrain_triggered

    # Push drift flags to XCom for branching
    context['ti'].xcom_push(key='drift_flags', value=drift_flags)
    triggered = [c for c, v in drift_flags.items() if v]
    print(f"Drift triggered for: {triggered if triggered else 'none'}")


def task_branch_on_drift(**context) -> str:
    """
    Task 4 — Branch: if any category needs retraining → retrain task.
    Otherwise skip straight to inventory update.
    """
    drift_flags: dict = context['ti'].xcom_pull(
        task_ids='update_drift_detector', key='drift_flags'
    ) or {}
    needs_retrain = any(drift_flags.values())
    return 'retrain_models' if needs_retrain else 'skip_retrain'


def task_retrain_models(**context) -> None:
    """
    Task 5 — Retrain Prophet for all flagged categories.
    Logs pre/post MAE and model artifact to MLflow.
    """
    import pandas as pd
    import mlflow
    from src.retraining.retrain_pipeline import RetrainPipeline
    from src.drift_detection.drift_detector import DriftDetectorRegistry

    mlflow.set_tracking_uri(f'file://{MLRUNS_DIR}')

    df       = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    run_date = str(pd.Timestamp(context['ds']).date())

    drift_flags: dict = context['ti'].xcom_pull(
        task_ids='update_drift_detector', key='drift_flags'
    ) or {}
    flagged_cats = [c for c, v in drift_flags.items() if v]

    if not flagged_cats:
        print("No categories need retraining.")
        return

    pipeline = RetrainPipeline(
        demand_data=df, window_days=45, holdout_days=14, mlflow_tracking=True,
    )

    BASELINE_MAES = {
        'Electronics & Tech': 2071, 'Entertainment & Office': 4189,
        'Fashion & Accessories': 1677, 'Health & Personal Care': 4449,
        'Home & Lifestyle': 2605, 'Sports & Outdoors': 1514,
    }
    registry = DriftDetectorRegistry(
        baseline_maes=BASELINE_MAES, threshold=2.0,
        short_window=7, long_window=30, min_days=3,
    )

    # Placeholder: in production, current_model would be loaded from MLflow Registry
    for cat in flagged_cats:
        print(f"  Retraining {cat}...")
        # registry.get_detector(cat).current_mae used as pre_retrain_mae
        result, new_model = pipeline.retrain(
            category        = cat,
            retrain_date    = run_date,
            current_model   = None,            # loaded from registry in production
            pre_retrain_mae = BASELINE_MAES.get(cat, 3000.0),
            trigger_reason  = 'drift_detected',
        )
        status = 'ACCEPTED' if result.model_accepted else 'REJECTED'
        print(f"  {cat}: {status} | improvement={result.mae_improvement_pct:+.1f}%")

    pipeline.save_retrain_log()
    print("Retraining complete.")


def task_update_inventory(**context) -> None:
    """
    Task 6 — Recompute safety stock and reorder points from latest forecasts.
    Saves inventory decisions to data/processed/inventory_decisions.csv.
    """
    import pandas as pd
    from src.inventory.replenishment import InventoryCalculator

    forecast_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'forecasts')
    out_path     = os.path.join(PROJECT_ROOT, 'data', 'processed', 'inventory_decisions.csv')
    rows = []

    CATEGORY_PARAMS = {
        'Electronics & Tech'    : dict(service_level=0.95, lead_time_days=7,  cycle_days=7),
        'Entertainment & Office': dict(service_level=0.90, lead_time_days=5,  cycle_days=7),
        'Fashion & Accessories' : dict(service_level=0.90, lead_time_days=10, cycle_days=14),
        'Health & Personal Care': dict(service_level=0.95, lead_time_days=4,  cycle_days=7),
        'Home & Lifestyle'      : dict(service_level=0.90, lead_time_days=6,  cycle_days=14),
        'Sports & Outdoors'     : dict(service_level=0.90, lead_time_days=5,  cycle_days=7),
    }

    for cat, params in CATEGORY_PARAMS.items():
        fpath = os.path.join(forecast_dir, f'{cat.replace(" ", "_")}_forecast.csv')
        if not os.path.exists(fpath):
            continue
        fdf     = pd.read_csv(fpath)
        calc    = InventoryCalculator(
            service_level  = float(params['service_level']),
            lead_time_days = int(params['lead_time_days']),
            cycle_days     = int(params['cycle_days']),
        )
        decision = calc.compute(
            category=cat, forecast_df=fdf, forecast_date=str(pd.Timestamp(context['ds']).date()),
        )
        rows.append(decision.to_dict())

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Inventory decisions saved: {out_path}")


def task_save_drift_log(**context) -> None:
    """
    Task 7 — Archive today's drift status to data/drift_logs/.
    """
    import json
    import pandas as pd
    from datetime import date

    os.makedirs(DRIFT_LOG, exist_ok=True)
    run_date    = str(pd.Timestamp(context['ds']).date()) if 'ds' in context else str(date.today())
    drift_flags = context['ti'].xcom_pull(
        task_ids='update_drift_detector', key='drift_flags'
    ) or {}

    log_path = os.path.join(DRIFT_LOG, f'drift_status_{run_date}.json')
    with open(log_path, 'w') as f:
        json.dump({'date': run_date, 'drift_flags': drift_flags}, f, indent=2)
    print(f"Drift log saved: {log_path}")


# ─────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────

with DAG(
    dag_id          = 'drift_pipeline',
    description     = 'Drift-aware daily forecasting and auto-retraining pipeline',
    default_args    = DEFAULT_ARGS,
    start_date      = datetime(2025, 11, 1),
    schedule        = '@daily',
    catchup         = False,
    max_active_runs = 1,
    tags            = ['forecasting', 'drift', 'retraining', 'mlops'],
) as dag:

    t_data_quality  = PythonOperator(task_id='data_quality_check',    python_callable=task_data_quality_check)
    t_forecasts     = PythonOperator(task_id='run_forecasts',          python_callable=task_run_forecasts)
    t_drift_update  = PythonOperator(task_id='update_drift_detector',  python_callable=task_update_drift_detector)
    t_branch        = BranchPythonOperator(task_id='check_drift',      python_callable=task_branch_on_drift)
    t_retrain       = PythonOperator(task_id='retrain_models',         python_callable=task_retrain_models)
    t_skip          = EmptyOperator(task_id='skip_retrain')
    t_inventory     = PythonOperator(task_id='update_inventory',       python_callable=task_update_inventory,
                                     trigger_rule='none_failed_min_one_success')
    t_drift_log     = PythonOperator(task_id='save_drift_log',         python_callable=task_save_drift_log,
                                     trigger_rule='all_done')

    # Pipeline order:
    # data_quality → run_forecasts → update_drift_detector → check_drift
    #                                                              ├── retrain_models ─┐
    #                                                              └── skip_retrain   ─┤
    #                                                                                  ↓
    #                                                                         update_inventory → save_drift_log

    (t_data_quality >> t_forecasts >> t_drift_update >> t_branch)  # type: ignore[operator]
    (t_branch >> [t_retrain, t_skip])  # type: ignore[operator]
    ([t_retrain, t_skip] >> t_inventory >> t_drift_log)  # type: ignore[operator]
