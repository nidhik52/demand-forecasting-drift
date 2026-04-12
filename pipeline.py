import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
import random
import os
import warnings
from collections import deque
from tqdm import tqdm
from datetime import timedelta, datetime
from typing import Dict, Optional, Any, List, cast

warnings.filterwarnings("ignore")

import mlflow

MODEL_TYPE = os.getenv("MODEL_TYPE", "prophet")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

from sqlalchemy.orm import Session
from src.db import SessionLocal, Inventory, Order

# ----------------------------
# CmdStan setup
# ----------------------------
_CMDSTAN_SETUP_DONE = False

def setup_cmdstan_once():
    global _CMDSTAN_SETUP_DONE
    if _CMDSTAN_SETUP_DONE:
        return
    try:
        import cmdstanpy
        os.environ.setdefault("CXXFLAGS", "-O0 -g0 -std=c++14")
        os.environ.setdefault("MAKEFLAGS", "-j2")
        cmdstan_dir = Path.home() / ".cmdstan"
        cmdstan_path = cmdstan_dir / "cmdstan-2.33.1"
        if not cmdstan_path.exists():
            cmdstanpy.install_cmdstan(version="2.33.1", dir=str(cmdstan_dir), overwrite=False)
        cmdstanpy.set_cmdstan_path(str(cmdstan_path))
        _CMDSTAN_SETUP_DONE = True
    except Exception as e:
        raise RuntimeError(f"CmdStan setup failed: {e}")

if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
    setup_cmdstan_once()

# ----------------------------
# Model functions (unchanged)
# ----------------------------
def train_baseline(df: pd.DataFrame) -> float:
    return float(df["Demand"].mean())

def predict_baseline(model: float, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), model)

def train_prophet(df: pd.DataFrame):
    if Prophet is None:
        raise RuntimeError("Prophet not available.")
    prophet_df = df.rename(columns={"Date": "ds", "Demand": "y"})[["ds", "y"]]
    model = Prophet(
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        stan_backend="CMDSTANPY"
    )
    model.fit(prophet_df)
    return model

def predict_prophet(model, df: pd.DataFrame) -> np.ndarray:
    future = pd.DataFrame({"ds": pd.to_datetime(df["Date"])})
    forecast = model.predict(future)
    pred = forecast["yhat"].astype(float).to_numpy()
    return np.clip(pred, 0.0, None)

def calculate_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))

def calculate_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    nonzero = actual != 0
    if not np.any(nonzero):
        return 0.0
    return float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100)

def calculate_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))

def data_quality_check(df: pd.DataFrame, sku: str, date: pd.Timestamp) -> Dict:
    issues = []
    if df["Demand"].isnull().any():
        issues.append("missing_demand")
    if (df["Demand"] < 0).any():
        issues.append("negative_demand")
    mean = df["Demand"].mean()
    std = df["Demand"].std()
    if std > 0 and (df["Demand"] > mean + 3 * std).any():
        issues.append("outlier_demand")
    return {"sku": sku, "date": date, "issues": issues, "pass": len(issues) == 0}

# ----------------------------
# Data loading
# ----------------------------
def load_data() -> pd.DataFrame:
    from src.config import DAILY_DEMAND_FILE
    df = pd.read_csv(DAILY_DEMAND_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def ensure_data_range(start: str, end: str) -> None:
    from src.config import DAILY_DEMAND_FILE
    from src.preprocessing import load_data as load_raw_data, preprocess_data, save_processed_data
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if not DAILY_DEMAND_FILE.exists():
        raw_df = load_raw_data()
        demand = preprocess_data(raw_df)
        save_processed_data(demand)
    else:
        existing = pd.read_csv(DAILY_DEMAND_FILE)
        existing["Date"] = pd.to_datetime(existing["Date"])
        min_date, max_date = existing["Date"].min(), existing["Date"].max()
        if start_dt < min_date or end_dt > max_date:
            raw_df = load_raw_data()
            demand = preprocess_data(raw_df)
            save_processed_data(demand)

# ----------------------------
# Event logger
# ----------------------------
try:
    from src.event_logger import log_event as _log_event
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

def log_event(event_type: str, message: str, sim_date: pd.Timestamp):
    """
    Log event using the SIMULATION date (not datetime.now()).
    Adds a realistic hour offset so timestamps aren't 00:00:00.
    """
    hour   = random.randint(8, 18)
    minute = random.randint(0, 59)
    event_time = sim_date + timedelta(hours=hour, minutes=minute)

    if LOGGER_AVAILABLE:
        _log_event(event_type, message, event_time)
    else:
        print(f"[{event_time}] {event_type} → {message}")

# ----------------------------
# Rolling MAE drift detector
# FIX: replaces random.random() with real MAE-ratio logic
# ----------------------------
class RollingDriftDetector:
    """
    Per-SKU rolling MAE drift detector.
    Flags drift when recent MAE > threshold * baseline MAE.
    Cooldown prevents re-triggering immediately after retrain.
    """
    def __init__(self, threshold: float = 2.0, window: int = 7, min_days: int = 3, cooldown_days: int = 7):
        self.threshold    = threshold
        self.window       = window
        self.min_days     = min_days
        self.cooldown_days = cooldown_days

        self._errors:       Dict[str, deque] = {}
        self._baselines:    Dict[str, float] = {}
        self._consec:       Dict[str, int]   = {}
        self._last_retrain: Dict[str, Optional[pd.Timestamp]] = {}

    def update(self, sku: str, actual: float, predicted: float,
               current_date: pd.Timestamp) -> Dict:
        """
        Feed one day's error. Returns drift status dict.
        """
        if sku not in self._errors:
            self._errors[sku]       = deque(maxlen=self.window)
            self._baselines[sku]    = None
            self._consec[sku]       = 0
            self._last_retrain[sku] = None

        err = abs(actual - predicted)
        self._errors[sku].append(err)

        # Need at least window days before we have a baseline
        if len(self._errors[sku]) < self.window:
            return {"drift": False, "retrain": False, "ratio": 0.0, "rolling_mae": err}

        rolling_mae = float(np.mean(self._errors[sku]))

        # Set baseline once from first full window, then freeze it
        # (reset when retraining happens)
        if self._baselines[sku] is None:
            self._baselines[sku] = rolling_mae
            return {"drift": False, "retrain": False, "ratio": 1.0, "rolling_mae": rolling_mae}

        baseline = self._baselines[sku]
        ratio    = rolling_mae / baseline if baseline > 0 else 1.0

        flagged = ratio > self.threshold
        if flagged:
            self._consec[sku] += 1
        else:
            self._consec[sku] = 0

        drift_detected = self._consec[sku] >= self.min_days

        # Cooldown check
        in_cooldown = False
        if self._last_retrain[sku] is not None:
            days_since = (current_date - self._last_retrain[sku]).days
            in_cooldown = days_since < self.cooldown_days

        retrain_trigger = drift_detected and not in_cooldown

        return {
            "drift":       drift_detected,
            "retrain":     retrain_trigger,
            "ratio":       round(ratio, 3),
            "rolling_mae": round(rolling_mae, 3),
            "in_cooldown": in_cooldown,
        }

    def record_retrain(self, sku: str, current_date: pd.Timestamp, new_baseline: Optional[float] = None):
        """Call after successful retrain. Resets cooldown and baseline."""
        self._last_retrain[sku] = current_date
        self._consec[sku]       = 0
        self._errors[sku].clear()
        if new_baseline is not None:
            self._baselines[sku] = new_baseline
        else:
            self._baselines[sku] = None  # will recompute after window fills

# ----------------------------
# Inventory helpers (unchanged)
# ----------------------------
def process_in_transit_orders(current_date: pd.Timestamp, session: Session):
    orders = session.query(Order).filter(Order.restock_date <= current_date, Order.received == 0).all()
    for order in orders:
        inv = session.query(Inventory).filter_by(sku=order.sku).first()
        if inv:
            inv.current_stock += order.order_qty   # type: ignore
            inv.in_transit    -= order.order_qty   # type: ignore
            order.received     = 1                 # type: ignore
    session.commit()

def update_inventory_after_demand(sku: str, demand_quantity: int, session: Session) -> int:
    inv = session.query(Inventory).filter_by(sku=sku).first()
    if not inv:
        inv = Inventory(sku=sku, current_stock=100, lead_time_days=7, safety_stock=10)
        session.add(inv)
        session.commit()
        session.refresh(inv)
    current_stock = int(cast(Optional[int], inv.current_stock) or 0)
    new_stock     = max(0, current_stock - demand_quantity)
    inv.current_stock = new_stock          # type: ignore
    inv.last_updated  = datetime.utcnow() # type: ignore
    session.commit()
    return new_stock

def get_inventory_data(sku: str, session: Session) -> Dict:
    inv = session.query(Inventory).filter_by(sku=sku).first()
    if not inv:
        return {"current_stock": 100, "lead_time_days": 7, "safety_stock": 10, "in_transit": 0}
    return {
        "current_stock": int(inv.current_stock),    # type: ignore
        "lead_time_days": int(inv.lead_time_days),  # type: ignore
        "safety_stock": int(inv.safety_stock),      # type: ignore
        "in_transit": int(inv.in_transit)           # type: ignore
    }

# ----------------------------
# Main pipeline
# FIX: drift uses RollingDriftDetector not random
# FIX: event timestamps use sim date not datetime.now()
# FIX: MLflow uses log_model correctly for Prophet
# ----------------------------
def run_pipeline(start: str, end: str, run_id: str = "manual",
                 drift_threshold: float = 2.0, cooldown_days: int = 7) -> None:
    print(f"\n🚀 Pipeline | {start} → {end} | Threshold={drift_threshold}x | Cooldown={cooldown_days}d")
    ensure_data_range(start, end)
    df = load_data()
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]

    detector = RollingDriftDetector(
        threshold=drift_threshold,
        window=7,
        min_days=3,
        cooldown_days=cooldown_days,
    )

    sku_model   = {}
    results     = []
    inv_results = []
    quality_log = []

    # Always clear event log at start so timestamps match this run's sim dates
    from src.config import EVENT_LOG_FILE
    # Also clear the hardcoded path in event_logger (same file, different reference)
    hardcoded_log = Path("data/processed/system_events.csv")
    for log_path in set([EVENT_LOG_FILE, hardcoded_log]):
        if log_path.exists():
            log_path.unlink()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "demand_forecasting_drift"
    artifact_location = str(Path("./mlruns").absolute())
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    else:
        # If artifact location is not as desired, warn but do not delete (DB mode does not allow delete by name)
        if not exp.artifact_location.startswith("file://" + str(Path("./mlruns").absolute())) and not exp.artifact_location.endswith("/mlruns"):
            print(f"WARNING: MLflow experiment '{experiment_name}' exists with artifact location {exp.artifact_location}.\nArtifacts may not be stored in the desired location. To change, delete the experiment manually from the MLflow UI or DB.")
    mlflow.set_experiment(experiment_name)

    session = SessionLocal()

    for current_date in tqdm(sorted(df["Date"].unique()), desc="Processing days"):
        current_date = pd.Timestamp(current_date)
        process_in_transit_orders(current_date, session)

        day_df = df[df["Date"] == current_date]
        for sku in day_df["SKU"].unique():
            hist = df[(df["SKU"] == sku) & (df["Date"] <= current_date)].sort_values("Date")
            if len(hist) < 3:
                continue

            train_df = hist.iloc[:-1]
            test_df  = hist.iloc[-1:]

            quality = data_quality_check(train_df, sku, current_date)
            quality_log.append(quality)

            # Train initial model
            if sku not in sku_model:
                if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                    model = train_prophet(train_df)
                else:
                    model = train_baseline(train_df)
                sku_model[sku] = model

            model = sku_model[sku]
            if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                preds = predict_prophet(model, test_df)
            else:
                preds = predict_baseline(model, test_df)

            actual = test_df["Demand"].values
            mae    = calculate_mae(actual, preds)
            mape   = calculate_mape(actual, preds)
            rmse   = calculate_rmse(actual, preds)

            # FIX: use rolling MAE ratio, not random
            drift_status     = detector.update(sku, float(actual[0]), float(preds[0]), current_date)
            drift_detected   = drift_status["drift"]
            retrain_happened = False

            if drift_detected:
                # FIX: event timestamp = simulation date (not now())
                log_event("DRIFT",
                          f"{sku} drift detected (ratio={drift_status['ratio']:.2f}x, rolling_MAE={drift_status['rolling_mae']:.2f})",
                          current_date)

                if drift_status["retrain"]:
                    print(f"  🔄 {sku}: RETRAIN on {current_date.date()} (ratio={drift_status['ratio']:.2f}x)")
                    if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                        new_model = train_prophet(train_df)
                    else:
                        new_model = train_baseline(train_df)

                    sku_model[sku] = new_model
                    retrain_happened = True

                    # FIX: save model with sim date timestamp

                    # Use a relative, writable path for model artifacts (CI/CD safe)
                    model_dir = Path("./models") / ("prophet" if MODEL_TYPE == "prophet" else "baseline")
                    model_dir.mkdir(parents=True, exist_ok=True)
                    ts_str = current_date.strftime("%Y%m%d_%H%M%S")
                    model_path = model_dir / f"{sku}_{ts_str}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(new_model, f)

                    # Log artifact using a relative path
                    with mlflow.start_run(run_name=f"{sku}_{current_date.strftime('%Y%m%d')}"):
                        mlflow.log_param("sku", sku)
                        mlflow.log_param("model_type", MODEL_TYPE)
                        mlflow.log_param("retrain_date", str(current_date.date()))
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("mape", mape)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("drift_ratio", drift_status["ratio"])
                        mlflow.log_artifact(str(model_path))

                    # New baseline = post-retrain MAE on same day
                    new_preds   = predict_prophet(new_model, test_df) if MODEL_TYPE == "prophet" else predict_baseline(new_model, test_df)
                    new_mae     = calculate_mae(actual, new_preds)
                    detector.record_retrain(sku, current_date, new_baseline=new_mae)

                    log_event("RETRAIN",
                              f"{sku} retrained (pre_MAE={mae:.2f} post_MAE={new_mae:.2f})",
                              current_date)
                else:
                    print(f"  ⏭️ {sku}: drift in cooldown on {current_date.date()}")
                    log_event("COOLDOWN",
                              f"{sku} drift detected but in cooldown period",
                              current_date)
            else:
                pass  # no log for stable (avoids log explosion)

            demand_today = int(test_df["Demand"].iloc[0])
            new_stock    = update_inventory_after_demand(sku, demand_today, session)
            inv_data     = get_inventory_data(sku, session)
            lead_time    = inv_data["lead_time_days"]
            safety_stock = inv_data["safety_stock"]
            forecast_demand = float(preds[0])

            reorder_point = forecast_demand * lead_time + safety_stock
            reorder_qty   = max(0, int(reorder_point - new_stock))
            risk = "CRITICAL" if reorder_qty > 100 else "WARNING" if reorder_qty > 0 else "SAFE"

            inv_results.append({
                "Date": current_date,
                "SKU": sku,
                "Current_Stock": new_stock,
                "In_Transit": inv_data["in_transit"],
                "Recommended_Order_Qty": reorder_qty,
                "Risk_Level": risk,
                "Lead_Time_Days": lead_time,
                "Safety_Stock": safety_stock,
            })

            results.append({
                "Date":      test_df["Date"].iloc[0],
                "SKU":       sku,
                "Actual":    demand_today,
                "Predicted": round(forecast_demand, 4),
                "MAE":       round(mae, 4),
                "MAPE":      round(mape, 4),
                "RMSE":      round(rmse, 4),
                "Drift":     int(drift_detected),
                "Retrained": int(retrain_happened),
            })


    # --- Sync inventory_master.csv with DB ---
    from src.config import INVENTORY_FILE
    inventory_db = session.query(Inventory).all()
    inv_rows = []
    for inv in inventory_db:
        inv_rows.append({
            "SKU": inv.sku,
            "Current_Stock": inv.current_stock,
            "In_Transit": inv.in_transit,
            "Lead_Time_Days": inv.lead_time_days,
            "Safety_Stock": inv.safety_stock,
            "Stock_As_Of_Date": inv.last_updated.strftime("%Y-%m-%d") if inv.last_updated else ""
        })
    inv_df = pd.DataFrame(inv_rows)
    INVENTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(INVENTORY_FILE, index=False)
    print(f"Inventory DB exported to {INVENTORY_FILE}")

    # --- Regenerate inventory_recommendations.csv ---
    try:
        from src.inventory import load_data as load_inv_data, generate_inventory_recommendations, save_inventory
        forecast, inventory = load_inv_data()
        # Use the latest date in inventory as current_date
        current_date = inventory["Stock_As_Of_Date"].max() if not inventory.empty else pd.Timestamp.now()
        recs_df = generate_inventory_recommendations(forecast, inventory, current_date)
        save_inventory(recs_df)
        print("Inventory recommendations regenerated.")
    except Exception as e:
        print(f"Failed to regenerate inventory recommendations: {e}")

    session.close()

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv("data/processed/metrics.csv", index=False)
    pd.DataFrame(inv_results).to_csv("data/processed/inventory_recommendations.csv", index=False)
    pd.DataFrame(quality_log).to_csv("data/processed/data_quality.csv", index=False)

    # FIX: always write system_events.csv even if empty (CI/CD validate step needs it)
    event_log_path = Path("data/processed/system_events.csv")
    if not event_log_path.exists():
        pd.DataFrame(columns=["timestamp", "event_type", "message"]).to_csv(event_log_path, index=False)
        print("  ℹ  No drift events — empty system_events.csv written")

    print(f"\n✅ Pipeline done. {len(results)} predictions written.")

    # --- Generate forecast_2025.csv for dashboard and notebook validation ---
    try:
        from src.forecasting import run_forecasting
        print("\n🔮 Generating forecast_2025.csv for dashboard/notebook...")
        run_forecasting(df, silent=True)
        print("✅ forecast_2025.csv generated.")
    except Exception as e:
        print(f"❌ Failed to generate forecast_2025.csv: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",          required=True)
    parser.add_argument("--end",            required=True)
    parser.add_argument("--run_id",         default="manual")
    parser.add_argument("--drift_threshold",type=float, default=2.0)
    parser.add_argument("--cooldown_days",  type=int, default=7)
    args = parser.parse_args()
    run_pipeline(args.start, args.end, args.run_id, args.drift_threshold, args.cooldown_days)