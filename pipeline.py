import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
import random
import os
import warnings
from tqdm import tqdm
from datetime import timedelta, datetime
from typing import Dict, Optional, Any, List, cast

warnings.filterwarnings("ignore")

# MLflow
import mlflow
from mlflow import sklearn as mlflow_sklearn

MODEL_TYPE = os.getenv("MODEL_TYPE", "prophet")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

# ----------------------------
# Database setup (SQLite)
# ----------------------------
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
# Model functions
# ----------------------------
def train_baseline(df: pd.DataFrame) -> float:
    return float(df["Demand"].mean())

def predict_baseline(model: float, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), model)

def train_prophet(df: pd.DataFrame):
    if Prophet is None:
        raise RuntimeError("Prophet is not available. Install prophet or switch MODEL_TYPE.")
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
    actual = np.array(actual)
    pred = np.array(pred)
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
    if (df["Demand"] > mean + 3*std).any():
        issues.append("outlier_demand")
    return {"sku": sku, "date": date, "issues": issues, "pass": len(issues) == 0}

# ----------------------------
# Data loading (from your src.config)
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
# Event logger (safe)
# ----------------------------
try:
    from src.event_logger import log_event
    LOGGER_AVAILABLE = True
except ImportError:
    def log_event(*args, **kwargs):
        pass
    LOGGER_AVAILABLE = False

# ----------------------------
# Inventory helpers (with type ignores)
# ----------------------------
def process_in_transit_orders(current_date: pd.Timestamp, session: Session):
    orders = session.query(Order).filter(Order.restock_date <= current_date, Order.received == 0).all()
    for order in orders:
        inv = session.query(Inventory).filter_by(sku=order.sku).first()
        if inv:
            inv.current_stock += order.order_qty  # type: ignore
            inv.in_transit -= order.order_qty     # type: ignore
            order.received = 1                    # type: ignore
    session.commit()

def update_inventory_after_demand(sku: str, demand_quantity: int, session: Session) -> int:
    inv = session.query(Inventory).filter_by(sku=sku).first()
    if not inv:
        inv = Inventory(sku=sku, current_stock=0, lead_time_days=7, safety_stock=10)
        session.add(inv)
        session.commit()
        session.refresh(inv)
    current_stock = int(cast(Optional[int], inv.current_stock) or 0)
    new_stock = max(0, current_stock - demand_quantity)
    inv.current_stock = new_stock   # type: ignore
    inv.last_updated = datetime.utcnow()  # type: ignore
    session.commit()
    return new_stock

def get_inventory_data(sku: str, session: Session) -> Dict:
    inv = session.query(Inventory).filter_by(sku=sku).first()
    if not inv:
        return {"current_stock": 0, "lead_time_days": 7, "safety_stock": 10, "in_transit": 0}
    return {
        "current_stock": int(inv.current_stock),      # type: ignore
        "lead_time_days": int(inv.lead_time_days),    # type: ignore
        "safety_stock": int(inv.safety_stock),        # type: ignore
        "in_transit": int(inv.in_transit)             # type: ignore
    }

# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline(start: str, end: str, run_id: str = "manual",
                 drift_probability: float = 0.3, cooldown_days: int = 7) -> None:
    print(f"\n🚀 Pipeline | {start} → {end} | Drift prob={drift_probability*100}% | Cooldown={cooldown_days}d")
    ensure_data_range(start, end)
    df = load_data()
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]

    sku_model = {}
    sku_last_retrain = {}
    results = []
    inventory_results = []
    quality_log = []

    session = SessionLocal()

    for current_date in tqdm(sorted(df["Date"].unique()), desc="Processing days"):
        process_in_transit_orders(current_date, session)

        day_df = df[df["Date"] == current_date]
        for sku in day_df["SKU"].unique():
            hist = df[(df["SKU"] == sku) & (df["Date"] <= current_date)].sort_values("Date")
            if len(hist) < 3:
                continue
            train_df = hist.iloc[:-1]
            test_df = hist.iloc[-1:]

            quality = data_quality_check(train_df, sku, current_date)
            quality_log.append(quality)
            if not quality["pass"]:
                print(f"  ⚠️ {sku}: data quality issues {quality['issues']}")

            if sku not in sku_model:
                if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                    model = train_prophet(train_df)
                else:
                    model = train_baseline(train_df)
                sku_model[sku] = model
                sku_last_retrain[sku] = current_date
                print(f"  🆕 {sku}: initial model trained")

            model = sku_model[sku]
            if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                preds = predict_prophet(model, test_df)
            else:
                preds = predict_baseline(model, test_df)

            actual = test_df["Demand"].values
            mae = calculate_mae(actual, preds)
            mape = calculate_mape(actual, preds)
            rmse = calculate_rmse(actual, preds)

            random_drift = random.random() < drift_probability
            drift_detected = (mae > 2.5) or random_drift
            retrain_happened = False

            if drift_detected:
                days_since = (current_date - sku_last_retrain[sku]).days
                if days_since >= cooldown_days:
                    print(f"  🔄 {sku}: DRIFT (MAE={mae:.2f}) → retraining")
                    if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                        new_model = train_prophet(train_df)
                    else:
                        new_model = train_baseline(train_df)
                    sku_model[sku] = new_model
                    sku_last_retrain[sku] = current_date
                    retrain_happened = True

                    model_dir = Path("models") / ("prophet" if MODEL_TYPE == "prophet" else "baseline")
                    model_dir.mkdir(parents=True, exist_ok=True)
                    model_path = model_dir / f"{sku}_drift_{current_date.strftime('%Y%m%d')}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(new_model, f)

                    mlflow.set_tracking_uri("sqlite:///mlflow.db")
                    with mlflow.start_run(run_name=f"{sku}_{current_date.strftime('%Y%m%d')}"):
                        mlflow.log_param("sku", sku)
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("mape", mape)
                        mlflow.log_metric("rmse", rmse)
                        mlflow_sklearn.log_model(new_model, f"model_{sku}")

                    event_time = current_date + timedelta(hours=random.randint(9, 17), minutes=random.randint(0, 59))
                    log_event("DRIFT", f"{sku} drift detected (MAE={mae:.2f})", event_time)
                    log_event("RETRAIN", f"{sku} retrained and model saved", event_time)
                else:
                    print(f"  ⏭️ {sku}: DRIFT but cooldown active ({days_since} days ago)")
            else:
                print(f"  ✅ {sku}: stable (MAE={mae:.2f})")

            demand_today = int(test_df["Demand"].iloc[0])
            new_stock = update_inventory_after_demand(sku, demand_today, session)
            inv_data = get_inventory_data(sku, session)
            lead_time = inv_data["lead_time_days"]
            safety_stock = inv_data["safety_stock"]
            forecast_demand = float(preds[0])

            reorder_point = forecast_demand * lead_time + safety_stock
            reorder_qty = max(0, int(reorder_point - new_stock))
            risk = "CRITICAL" if reorder_qty > 100 else "WARNING" if reorder_qty > 0 else "SAFE"

            inventory_results.append({
                "Date": current_date,
                "SKU": sku,
                "Current_Stock": new_stock,
                "In_Transit": inv_data["in_transit"],
                "Recommended_Order_Qty": reorder_qty,
                "Risk_Level": risk,
                "Lead_Time_Days": lead_time,
                "Safety_Stock": safety_stock
            })

            results.append({
                "Date": test_df["Date"].iloc[0],
                "SKU": sku,
                "Actual": demand_today,
                "Predicted": forecast_demand,
                "MAE": mae,
                "MAPE": mape,
                "RMSE": rmse,
                "Drift": int(drift_detected),
                "Retrained": int(retrain_happened)
            })

    session.close()

    pd.DataFrame(results).to_csv("data/processed/metrics.csv", index=False)
    pd.DataFrame(inventory_results).to_csv("data/processed/inventory_recommendations.csv", index=False)
    pd.DataFrame(quality_log).to_csv("data/processed/data_quality.csv", index=False)
    print(f"\n✅ Pipeline finished. {len(results)} predictions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run_id", default="manual")
    parser.add_argument("--drift_probability", type=float, default=0.3)
    parser.add_argument("--cooldown_days", type=int, default=7)
    args = parser.parse_args()
    run_pipeline(args.start, args.end, args.run_id, args.drift_probability, args.cooldown_days)