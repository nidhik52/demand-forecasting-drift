"""
retraining.py
─────────────
Handles automatic model retraining when concept drift is detected.

Steps
─────
1. Gather all available daily demand data up to the current stream date.
2. Retrain a fresh Prophet model for the affected SKU.
3. Overwrite the saved model pickle in models/<SKU>.pkl.
4. Log the retraining event and MAE metric to MLflow.
5. Regenerate the 2026 forecast for that SKU and update forecast_2026.csv.
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DAILY_PATH    = Path("data/processed/daily_demand.csv")
FORECAST_PATH = Path("data/processed/forecast_2026.csv")
MODELS_DIR    = Path("models")

# ── MLflow experiment ─────────────────────────────────────────────────────────
EXPERIMENT_NAME  = "demand-forecasting-drift"
FORECAST_YEAR    = 2026
FORECAST_PERIODS = 365


def retrain(
    sku: str,
    data_up_to: pd.Timestamp | str,
    rolling_mae: float,
) -> Prophet:
    """
    Retrain the Prophet model for `sku` using all data up to `data_up_to`.

    Parameters
    ──────────
    sku          : SKU identifier (e.g. "ELEC-001")
    data_up_to   : cutoff date — rows after this date are excluded
    rolling_mae  : rolling MAE value that triggered the retrain (logged)

    Returns
    ───────
    Newly trained Prophet model.
    """
    daily  = pd.read_csv(DAILY_PATH, parse_dates=["Date"])
    cutoff = pd.Timestamp(data_up_to)

    # Use all available data for this SKU up to the cutoff
    sku_data = daily[(daily["SKU"] == sku) & (daily["Date"] <= cutoff)].copy()

    if len(sku_data) < 2:
        raise ValueError(
            f"Not enough data to retrain '{sku}': only {len(sku_data)} rows."
        )

    train_start = sku_data["Date"].min()
    train_end   = sku_data["Date"].max()
    print(
        f"  [Retrain] {sku}: {train_start.date()} → {train_end.date()}"
        f"  ({len(sku_data)} rows)"
    )

    # ── Train new model ────────────────────────────────────────────────────────
    ts = sku_data.rename(columns={"Date": "ds", "demand": "y"})[["ds", "y"]]
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    model.fit(ts)

    # ── Persist model (overwrites old pickle) ─────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{sku}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    _log_to_mlflow(
        sku=sku,
        model_path=model_path,
        train_start=train_start,
        train_end=train_end,
        rolling_mae=rolling_mae,
    )

    # ── Update 2026 forecast for this SKU ─────────────────────────────────────
    _update_forecast(sku, model)

    return model


def _log_to_mlflow(
    sku: str,
    model_path: Path,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    rolling_mae: float,
) -> None:
    """
    Log one retraining run to MLflow.

    Logged items
    ────────────
    • Parameters : sku, train_start, train_end, timestamp
    • Metrics    : rolling_mae_at_trigger
    • Tag        : trigger = "drift_detected"
    • Artifact   : the model pickle
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = f"retrain-{sku}-{datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("trigger", "drift_detected")
        mlflow.set_tag("sku", sku)

        mlflow.log_params({
            "sku":         sku,
            "train_start": str(train_start.date()),
            "train_end":   str(train_end.date()),
            "timestamp":   datetime.now().isoformat(),
            "seasonality": "yearly+weekly",
        })

        mlflow.log_metric("rolling_mae_at_trigger", rolling_mae)

        # Save the model pickle as a tracked MLflow artifact
        mlflow.log_artifact(str(model_path), artifact_path="model")


def _update_forecast(sku: str, model: Prophet) -> None:
    """
    Replace the 2026 forecast rows for `sku` in forecast_2026.csv
    with predictions from the newly retrained model.
    """
    # Generate fresh 2026 predictions
    future  = model.make_future_dataframe(periods=FORECAST_PERIODS, freq="D")
    fc      = model.predict(future)
    fc_2026 = fc[fc["ds"].dt.year == FORECAST_YEAR].copy()

    fc_2026["SKU"]             = sku
    fc_2026["forecast_demand"] = fc_2026["yhat"].clip(lower=0).round(2)
    new_rows = fc_2026[["ds", "SKU", "forecast_demand"]].rename(
        columns={"ds": "Date"}
    )

    # Merge: drop old rows for this SKU, append fresh ones
    if FORECAST_PATH.exists():
        existing = pd.read_csv(FORECAST_PATH, parse_dates=["Date"])
        existing = existing[existing["SKU"] != sku]
        updated  = pd.concat([existing, new_rows], ignore_index=True)
    else:
        updated = new_rows

    updated.sort_values(["SKU", "Date"], inplace=True)
    FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(FORECAST_PATH, index=False)
    print(f"  [Retrain] Forecast updated for {sku}.")
