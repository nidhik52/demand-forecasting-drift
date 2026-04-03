"""
forecasting.py
──────────────
Trains one Facebook Prophet model per SKU using a clear time-based split,
then generates a 365-day demand forecast for the full year 2026.

Data Split Strategy
───────────────────
  Training data    : 2024-01-01 → 2024-12-31  (in-sample fit)
  Validation data  : 2025-01-01 → 2025-06-30  (out-of-sample MAE evaluation)
  Final training   : 2024-01-01 → 2025-12-31  (all history, before 2026 forecast)

Steps
─────
  1. Validation phase  — train on 2024, predict 2025-H1, compute MAE per SKU
  2. Final phase       — retrain on full 2024–2025 data, forecast all of 2026
  3. Save final models and forecast CSV

Why Prophet?
────────────
• Designed for business time series with strong seasonality.
• Handles missing values and outliers gracefully.
• Produces human-interpretable trend + seasonality components.

Inputs  →  data/processed/daily_demand.csv
Outputs →  data/processed/forecast_2026.csv
           models/<SKU>.pkl   (one trained model pickle per SKU)
"""

import pickle
import warnings
from pathlib import Path

import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")   # suppress Stan / Prophet internal noise

# ── Paths ─────────────────────────────────────────────────────────────────────
DAILY_PATH    = Path("data/processed/daily_demand.csv")
FORECAST_PATH = Path("data/processed/forecast_2026.csv")
MODELS_DIR    = Path("models")

# ── Forecast parameters ───────────────────────────────────────────────────────
FORECAST_YEAR    = 2026
FORECAST_PERIODS = 365   # one full calendar year

# ── Data split boundaries ──────────────────────────────────────────────────────
TRAIN_END   = pd.Timestamp("2024-12-31")   # end of training window
VAL_START   = pd.Timestamp("2025-01-01")   # start of validation window
VAL_END     = pd.Timestamp("2025-06-30")   # end   of validation window


def load_daily_demand(path: Path = DAILY_PATH) -> pd.DataFrame:
    """Load the preprocessed daily demand CSV."""
    return pd.read_csv(path, parse_dates=["Date"])


def train_prophet(sku_df: pd.DataFrame) -> Prophet:
    """
    Fit a Prophet model for a single SKU's daily demand series.

    Prophet requires exactly two columns:
      ds  →  date
      y   →  numeric target (demand)

    Seasonality settings
    ────────────────────
    • yearly  : captures annual purchase cycles (e.g. peak Q4)
    • weekly  : captures weekday vs. weekend patterns
    • daily   : off — not meaningful at daily granularity
    """
    ts = sku_df.rename(columns={"Date": "ds", "demand": "y"})[["ds", "y"]]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,        # 95 % prediction interval
        changepoint_prior_scale=0.05,  # controls trend flexibility
    )
    model.fit(ts)
    return model


def forecast_sku(model: Prophet, periods: int = FORECAST_PERIODS) -> pd.DataFrame:
    """
    Extend the model's horizon by `periods` days and return predictions.

    Returned columns: ds | yhat | yhat_lower | yhat_upper
    """
    future   = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def save_model(model: Prophet, sku: str) -> Path:
    """Pickle a trained Prophet model to models/<SKU>.pkl."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{sku}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(sku: str) -> Prophet:
    """Load a previously saved Prophet model for the given SKU."""
    path = MODELS_DIR / f"{sku}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model for SKU '{sku}' at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def run_forecasting(save: bool = True) -> tuple[dict, pd.DataFrame]:
    """
    Main entry point — three-phase forecasting pipeline.

    Phase 1 — Validation
    ────────────────────
    For each SKU, train on 2024 data only, then predict 2025-H1 and
    compute the Mean Absolute Error (MAE) against held-out actuals.
    This validates that the model generalises before we commit it to
    a 2026 production forecast.

    Phase 2 — Final training
    ────────────────────────
    Retrain on the full 2024–2025 dataset (all available history) and
    generate the 2026 forecast that the inventory engine will use.

    Returns
    ───────
    models      : dict  {sku → Prophet}   (final models)
    forecast_df : DataFrame  Date | SKU | forecast_demand
    """
    daily = load_daily_demand()
    skus  = sorted(daily["SKU"].unique())

    # ── Phase 1 — Validation ────────────────────────────────────────────────
    print(f"[Forecasting] Phase 1 — Validation  "
          f"(train: 2024 | val: 2025-H1)  [{len(skus)} SKUs]")

    val_maes: list[float] = []

    for i, sku in enumerate(skus, 1):
        sku_df = daily[daily["SKU"] == sku].copy()

        train_df = sku_df[sku_df["Date"] <= TRAIN_END]
        val_df   = sku_df[(sku_df["Date"] >= VAL_START) & (sku_df["Date"] <= VAL_END)]

        if len(train_df) < 2 or val_df.empty:
            continue

        # Train on 2024
        val_model = train_prophet(train_df)

        # Predict far enough to cover the validation window
        val_days_needed = int((VAL_END - TRAIN_END).days) + 1
        fc = forecast_sku(val_model, periods=val_days_needed)

        # Align predictions with actuals
        fc_val = fc[(fc["ds"] >= VAL_START) & (fc["ds"] <= VAL_END)].copy()
        fc_val = fc_val.rename(columns={"ds": "Date", "yhat": "forecast_demand"})

        merged  = val_df.merge(fc_val[["Date", "forecast_demand"]], on="Date", how="inner")
        if merged.empty:
            continue

        mae = (merged["demand"] - merged["forecast_demand"]).abs().mean()
        val_maes.append(mae)

    avg_val_mae = sum(val_maes) / len(val_maes) if val_maes else float("nan")
    print(f"  Validation complete — avg MAE across {len(val_maes)} SKUs: "
          f"{avg_val_mae:.2f} units/day")

    # ── Phase 2 — Final training & 2026 forecast ────────────────────────────
    print(f"\n[Forecasting] Phase 2 — Final training  "
          f"(train: 2024–2025 | forecast: {FORECAST_YEAR})")

    all_forecasts: list[pd.DataFrame] = []
    models:        dict               = {}

    for i, sku in enumerate(skus, 1):
        sku_df = daily[daily["SKU"] == sku].copy()

        # Train on full 2024–2025 history
        model = train_prophet(sku_df)

        # Generate FORECAST_PERIODS days beyond the training end
        fc = forecast_sku(model, periods=FORECAST_PERIODS)

        # Keep only 2026 rows
        fc_2026 = fc[fc["ds"].dt.year == FORECAST_YEAR].copy()
        fc_2026["SKU"] = sku
        fc_2026 = fc_2026.rename(columns={"ds": "Date", "yhat": "forecast_demand"})

        # Clip negative forecasts to zero (demand cannot be negative)
        fc_2026["forecast_demand"] = fc_2026["forecast_demand"].clip(lower=0).round(2)

        all_forecasts.append(fc_2026[["Date", "SKU", "forecast_demand"]])

        models[sku] = model
        save_model(model, sku)
        print(f"  [{i:>3}/{len(skus)}] {sku} — retrained on 2024–2025 & 2026 forecast ✓")

    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    if save:
        FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(FORECAST_PATH, index=False)
        print(f"  Saved → {FORECAST_PATH}")

    print("[Forecasting] Done.")
    return models, forecast_df


if __name__ == "__main__":
    run_forecasting()
