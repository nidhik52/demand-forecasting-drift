import argparse
import numpy as np
import pandas as pd
import pickle

import mlflow
import mlflow.sklearn

from src.config import (
    DAILY_DEMAND_FILE,
    METRICS_FILE,
    INVENTORY_RECOMMENDATIONS_FILE,
    MODELS_DIR,
    DRIFT_THRESHOLD
)
from src.event_logger import log_event
from src.preprocessing import load_data as load_raw_data, preprocess_data, save_processed_data


def train_model(df):
    return df["Demand"].mean()


def predict(model, df):
    return np.full(len(df), model)


def calculate_mae(actual, pred):
    return np.mean(np.abs(actual - pred))


def load_data():
    df = pd.read_csv(DAILY_DEMAND_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def ensure_data_range(start, end):
    start_dt = pd.to_datetime(start, errors="coerce")
    end_dt = pd.to_datetime(end, errors="coerce")

    if pd.isna(start_dt) or pd.isna(end_dt):
        return

    needs_refresh = False

    if not DAILY_DEMAND_FILE.exists():
        needs_refresh = True
    else:
        try:
            existing = pd.read_csv(DAILY_DEMAND_FILE)
            existing["Date"] = pd.to_datetime(existing["Date"], errors="coerce")
            min_date = existing["Date"].min()
            max_date = existing["Date"].max()
            if pd.isna(min_date) or pd.isna(max_date):
                needs_refresh = True
            elif start_dt < min_date or end_dt > max_date:
                needs_refresh = True
        except Exception:
            needs_refresh = True

    if needs_refresh:
        raw_df = load_raw_data()
        demand = preprocess_data(raw_df)
        save_processed_data(demand)


def run_pipeline(start, end, run_id="manual"):
    print(f"\n🚀 Starting baseline pipeline | {start} → {end}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Drift-Aware Retail Forecasting")

    ensure_data_range(start, end)
    df = load_data()
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]

    results = []
    inventory_results = []

    for current_date in sorted(df["Date"].unique()):
        day_df = df[df["Date"] == current_date]

        for sku in day_df["SKU"].unique():
            sku_df = df[(df["SKU"] == sku) & (df["Date"] <= current_date)].sort_values("Date")

            if len(sku_df) < 3:
                continue

            train_df = sku_df.iloc[:-1]
            test_df = sku_df.iloc[-1:]

            model = train_model(train_df)
            preds = predict(model, test_df)
            mae = calculate_mae(test_df["Demand"].values, preds)

            drift_flag = 1 if mae > DRIFT_THRESHOLD else 0
            event_time = pd.to_datetime(current_date) + pd.Timedelta(minutes=5)
            timestamp_str = event_time.strftime("%Y%m%d_%H%M%S")

            if drift_flag:
                model_path = MODELS_DIR / f"{sku}_DRIFT_{timestamp_str}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            with mlflow.start_run(run_name=f"{sku}_{timestamp_str}"):
                mlflow.log_param("sku", sku)
                mlflow.log_param("start_date", str(start))
                mlflow.log_param("end_date", str(end))
                mlflow.log_param("model", "mean")

                mlflow.log_metric("mae", float(mae))
                mlflow.log_metric("drift", int(drift_flag))

                mlflow.set_tag("airflow_run_id", run_id)
                mlflow.set_tag("pipeline_stage", "drift_detection")

                if drift_flag:
                    mlflow.sklearn.log_model(model, f"model_{sku}")

            if drift_flag:
                log_event("DRIFT", f"{sku} drift detected ({mae:.2f})", event_time)
                log_event("RETRAIN", f"{sku} retrained and model saved", event_time)
            else:
                log_event("STABLE", f"{sku} stable ({mae:.2f})", event_time)

            results.append({
                "Date": test_df["Date"].iloc[0],
                "SKU": sku,
                "Actual": float(test_df["Demand"].iloc[0]),
                "Predicted": float(preds[0]),
                "MAE": float(mae),
                "Drift": drift_flag
            })

            current_stock = np.random.randint(0, 500)
            demand = float(preds[0])

            reorder_qty = max(0, int(demand * 1.2 - current_stock))

            if reorder_qty > 100:
                risk = "CRITICAL"
            elif reorder_qty > 0:
                risk = "WARNING"
            else:
                risk = "SAFE"

            inventory_results.append({
                "Date": current_date,
                "SKU": sku,
                "Product": f"Product_{sku}",
                "Current_Stock": current_stock,
                "Recommended_Order_Qty": reorder_qty,
                "Risk_Level": risk
            })

            print(f"[{event_time}] {sku} | MAE={mae:.2f} | Drift={drift_flag}")

    pd.DataFrame(results).to_csv(METRICS_FILE, index=False)
    pd.DataFrame(inventory_results).to_csv(INVENTORY_RECOMMENDATIONS_FILE, index=False)

    print("\n✅ Baseline pipeline completed\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run_id", default="manual")

    args = parser.parse_args()

    run_pipeline(args.start, args.end, args.run_id)
