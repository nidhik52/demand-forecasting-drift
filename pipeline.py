import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# -----------------------------
# IMPORT CONFIG FIRST (IMPORTANT FIX)
# -----------------------------
from src.config import (
    DAILY_DEMAND_FILE,
    FORECAST_FILE,
    MODELS_DIR,
    METRICS_FILE,
    INVENTORY_FILE,
    DRIFT_THRESHOLD
)

from src.event_logger import log_event

from src.inventory import (
    load_data as load_inventory_data,
    generate_inventory_recommendations,
    save_inventory
)

# -----------------------------
# CREATE DIRECTORIES
# -----------------------------
MODELS_DIR.mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    return pd.read_csv(DAILY_DEMAND_FILE)

# -----------------------------
# SIMPLE FORECAST MODEL
# -----------------------------
def train_model(data):
    # Simple moving average (replaceable later)
    return data["demand"].mean()

def forecast(model):
    return model + random.uniform(-2, 2)

# -----------------------------
# DRIFT DETECTION
# -----------------------------
def detect_drift(actual, predicted):
    error = abs(actual - predicted)
    return error, error > DRIFT_THRESHOLD

# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model, sku, timestamp):
    filename = f"{sku}_{timestamp}.pkl"
    path = MODELS_DIR / filename

    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"💾 Saving model at {path}")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline(start_date, end_date):
    df = load_data()

    # Normalize columns
    df.columns = df.columns.str.lower()

    df["date"] = pd.to_datetime(df["date"])

    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    while current_date <= end_date:

        print(f"\n📡 Processing {current_date.date()}")

        daily_data = df[df["date"] == current_date]

        metrics = []

        for sku in daily_data["sku"].unique():

            sku_data = daily_data[daily_data["sku"] == sku]

            actual = sku_data["demand"].values[0]
    

            # -----------------------------
            # TRAIN MODEL (INITIAL)
            # -----------------------------
            model = train_model(sku_data)

            predicted = forecast(model)

            error, drift = detect_drift(actual, predicted)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # -----------------------------
            # DRIFT HANDLING
            # -----------------------------
            if drift:
                log_event("DRIFT", f"Drift detected for {sku} ({actual:.2f} → {predicted:.2f})", datetime.now())

                model = train_model(sku_data)

                save_model(model, sku, timestamp)

                log_event("RETRAIN", f"{sku} retrained and saved model", datetime.now())

                print(f"✅ {sku} retrained")

            else:
                log_event("NO_DRIFT", f"{sku} stable, model reused", datetime.now())

                print(f"ℹ️ {sku} no drift, using existing model")

            # -----------------------------
            # STORE METRICS
            # -----------------------------
            metrics.append({
                "Date": current_date,
                "SKU": sku,
                "Actual": actual,
                "Predicted": predicted,
                "Error": error
            })

        # -----------------------------
        # SAVE METRICS
        # -----------------------------
        metrics_df = pd.DataFrame(metrics)

        if Path(METRICS_FILE).exists():
            old = pd.read_csv(METRICS_FILE)
            metrics_df = pd.concat([old, metrics_df], ignore_index=True)

        metrics_df.to_csv(METRICS_FILE, index=False)

        # -----------------------------
        # INVENTORY UPDATE
        # -----------------------------
        inv_df = load_inventory_data()
        recommendations = generate_inventory_recommendations(inv_df, metrics_df)
        save_inventory(recommendations)

        print("📦 Inventory updated")

        current_date += timedelta(days=1)

    print("\n✅ Pipeline completed")

# -----------------------------
# CLI ENTRY
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)

    args = parser.parse_args()

    run_pipeline(args.start, args.end)