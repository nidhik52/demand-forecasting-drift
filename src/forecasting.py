"""
forecasting.py
──────────────
Trains one Facebook Prophet model per SKU on historical daily demand
and generates a 365-day forecast covering the full year 2026.

Why Prophet?
────────────
• Designed for business time series with strong seasonality.
• Handles missing values and outliers gracefully.
• Produces human-interpretable trend + seasonality components.
• No manual feature engineering needed.

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
    Main entry point — trains one Prophet model per SKU and
    collects the 2026 forecasts into a single DataFrame.

    Returns
    ───────
    models      : dict  {sku → Prophet}
    forecast_df : DataFrame  Date | SKU | forecast_demand
    """
    daily = load_daily_demand()
    skus  = sorted(daily["SKU"].unique())
    print(f"[Forecasting] Training {len(skus)} Prophet models …")

    all_forecasts = []
    models        = {}

    for i, sku in enumerate(skus, 1):
        sku_df = daily[daily["SKU"] == sku].copy()

        # Train
        model = train_prophet(sku_df)

        # Generate FORECAST_PERIODS days beyond training data
        fc = forecast_sku(model, periods=FORECAST_PERIODS)

        # Keep only 2026 rows (the training data ends in 2025)
        fc_2026 = fc[fc["ds"].dt.year == FORECAST_YEAR].copy()
        fc_2026["SKU"] = sku
        fc_2026 = fc_2026.rename(columns={"ds": "Date", "yhat": "forecast_demand"})

        # Clip negative forecasts to zero (demand can't be negative)
        fc_2026["forecast_demand"] = fc_2026["forecast_demand"].clip(lower=0).round(2)

        all_forecasts.append(fc_2026[["Date", "SKU", "forecast_demand"]])

        # Persist model
        models[sku] = model
        save_model(model, sku)
        print(f"  [{i:>3}/{len(skus)}] {sku} — trained & forecast generated ✓")

    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    if save:
        FORECAST_PATH.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_csv(FORECAST_PATH, index=False)
        print(f"  Saved → {FORECAST_PATH}")

    print("[Forecasting] Done.")
    return models, forecast_df


if __name__ == "__main__":
    run_forecasting()
