import pandas as pd
import numpy as np
from prophet import Prophet
from tqdm import tqdm
from pathlib import Path
import sys
import pickle
import glob

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DAILY_DEMAND_FILE, FORECAST_FILE, MODELS_DIR


class SimpleAvgModel:
    """Fallback lightweight model: predicts historical value when available else mean."""
    def __init__(self, hist_df):
        self.hist = dict((pd.to_datetime(k).to_pydatetime(), float(v)) for k, v in hist_df.set_index('ds')['y'].to_dict().items())
        self.avg = float(hist_df['y'].mean()) if len(hist_df) > 0 else 0.0

    def predict(self, df):
        # df expected to have column 'ds'
        dates = pd.to_datetime(df['ds'])
        yhat = [self.hist.get(d.to_pydatetime(), self.avg) for d in dates]
        return pd.DataFrame({'ds': dates.values, 'yhat': yhat})


def _save_model_file(model, sku, timestamp):
    safe_sku = str(sku).replace("/", "-")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = f"{safe_sku}_{timestamp}.pkl"
    model_path = MODELS_DIR / model_filename
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        # update latest pointer
        latest_ptr = MODELS_DIR / f"{safe_sku}_latest.pkl"
        with open(latest_ptr, 'wb') as f:
            pickle.dump(model, f)
        return model_path
    except Exception:
        return None

def run_forecasting(df, silent=False):

    if not silent:
        print("\n🔮 Starting forecasting...\n")

    forecasts = []

    for sku in tqdm(df["SKU"].unique(), desc="Forecasting SKUs", disable=silent):

        sku_df = df[df["SKU"] == sku].copy()

        if len(sku_df) < 10:
            if not silent:
                print(f"⚠ Skipping {sku} (not enough data)")
            continue

        prophet_df = sku_df.rename(columns={"Date": "ds", "Demand": "y"})[["ds", "y"]]

        # Try to load latest model for SKU if exists; otherwise attempt to train a fresh Prophet model
        model = load_latest_model(sku)

        if model is None:
            try:
                model = Prophet(
                    daily_seasonality=True,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                )
                model.fit(prophet_df)
                # Save an initial model only if no model exists for this SKU yet
                safe_sku = str(sku).replace("/", "-")
                if get_latest_model_path(safe_sku) is None:
                    ts = prophet_df["ds"].max().strftime("%Y-%m-%d_%H-%M-%S")
                    _save_model_file(model, sku, f"initial_{ts}")
            except Exception:
                # if Prophet cannot be instantiated or trained, leave model as None and use fallback
                model = None

        # If a usable model exists, try to predict; otherwise use fallback average-based forecast
        if model is not None:
            try:
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                forecast["yhat"] = forecast["yhat"].clip(lower=0)
            except Exception:
                model = None

        if model is None:
            # Fallback: simple average-based forecast when Prophet is unavailable
            hist_min = prophet_df["ds"].min()
            hist_max = prophet_df["ds"].max()
            full_dates = pd.date_range(start=hist_min, end=hist_max + pd.Timedelta(days=30), freq="D")
            avg = prophet_df["y"].mean()
            yhat = []
            hist_map = prophet_df.set_index("ds")["y"].to_dict()
            for d in full_dates:
                if d in hist_map:
                    yhat.append(float(hist_map[d]))
                else:
                    yhat.append(float(avg))
            forecast = pd.DataFrame({"ds": full_dates, "yhat": yhat})
            # create a lightweight fallback model and save only if no existing model
            model = SimpleAvgModel(prophet_df)
            safe_sku = str(sku).replace("/", "-")
            if get_latest_model_path(safe_sku) is None:
                ts = prophet_df["ds"].max().strftime("%Y-%m-%d_%H-%M-%S")
                _save_model_file(model, sku, f"initial_{ts}")

        forecast["SKU"] = sku

        forecast = forecast[["ds", "yhat", "SKU"]]
        forecast.columns = ["Date", "Forecast_Demand", "SKU"]

        forecasts.append(forecast)

    final_forecast = pd.concat(forecasts)

    FORECAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    final_forecast.to_csv(FORECAST_FILE, index=False)

    if not silent:
        print(f"\n✅ Forecast saved to {FORECAST_FILE}")

    return final_forecast


def train_and_forecast(df, sku=None, silent=False):
    if sku is not None:
        df = df[df["SKU"] == sku].copy()
    return run_forecasting(df, silent=silent)


def save_retrained_model_artifact(sku_data, sku, drift_time):
    prophet_df = sku_data.rename(columns={"Date": "ds", "Demand": "y"})[["ds", "y"]]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Train candidate model
    try:
        new_model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
        )
        new_model.fit(prophet_df)
    except Exception:
        # fallback to lightweight model when Prophet is unavailable
        new_model = SimpleAvgModel(prophet_df)

    # Evaluate against existing latest model if present
    safe_sku = str(sku).replace("/", "-")
    latest = get_latest_model_path(safe_sku)

    def _mape_for(model, df):
        if model is None or len(df) == 0:
            return float('inf')
        test_df = df.rename(columns={"Date": "ds", "Demand": "y"})[["ds", "y"]]
        try:
            pred = model.predict(test_df[["ds"]])
        except Exception:
            return float('inf')

        y_true = test_df["y"].values
        y_pred = pred["yhat"].values

        # avoid divide by zero by using at least 1 in the denominator
        denom = np.maximum(np.abs(y_true), 1.0)
        mape = np.mean(np.abs(y_true - y_pred) / denom)
        return float(mape)

    new_mape = _mape_for(new_model, sku_data)

    if latest is not None:
        try:
            with open(latest, 'rb') as f:
                old_model = pickle.load(f)
        except Exception:
            old_model = None
        old_mape = _mape_for(old_model, sku_data)
    else:
        old_model = None
        old_mape = float('inf')

    timestamp = pd.to_datetime(drift_time).strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"{safe_sku}_{timestamp}.pkl"
    model_path = MODELS_DIR / model_filename

    # Save new model only if it improves MAPE (lower is better) or no previous model
    if new_mape < old_mape or latest is None:
        with open(model_path, 'wb') as f:
            pickle.dump(new_model, f)

        # update latest pointer
        latest_ptr = MODELS_DIR / f"{safe_sku}_latest.pkl"
        with open(latest_ptr, 'wb') as f:
            pickle.dump(new_model, f)

        return model_path
    else:
        # keep existing model
        return Path(latest) if latest is not None else None


def get_latest_model_path(safe_sku):
    # find model files for SKU and return the most recent by name ordering
    pattern = str(MODELS_DIR / f"{safe_sku}_*.pkl")
    files = glob.glob(pattern)
    if not files:
        # also support latest pointer
        latest_ptr = MODELS_DIR / f"{safe_sku}_latest.pkl"
        if latest_ptr.exists():
            return str(latest_ptr)
        return None
    files.sort()
    return files[-1]


def load_latest_model(sku):
    safe_sku = str(sku).replace("/", "-")
    latest = get_latest_model_path(safe_sku)
    if latest is None:
        return None
    try:
        with open(latest, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception:
        return None


if __name__ == "__main__":
    df = pd.read_csv(DAILY_DEMAND_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    run_forecasting(df)