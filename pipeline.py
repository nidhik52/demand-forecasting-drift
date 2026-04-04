import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
from pathlib import Path

from pathlib import Path
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

from src.config import (
    DAILY_DEMAND_FILE,
    FORECAST_FILE,
    MODELS_DIR,
    METRICS_FILE,
    INVENTORY_FILE
)

from src.event_logger import log_event

from src.inventory import (
    load_data as load_inventory_data,
    generate_inventory_recommendations,
    save_inventory
)

# -----------------------------
# CONFIG
# -----------------------------
DRIFT_THRESHOLD = 1.4
MIN_HISTORY = 30
COOLDOWN_DAYS = 5


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    daily = pd.read_csv(DAILY_DEMAND_FILE)
    forecast = pd.read_csv(FORECAST_FILE)

    daily["Date"] = pd.to_datetime(daily["Date"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])

    return daily, forecast


# -----------------------------
# ERROR FUNCTION
# -----------------------------
def compute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))


# -----------------------------
# SIMPLE MODEL
# -----------------------------
def train_model(data):
    return data.mean()


def predict(model, steps):
    return np.array([model] * steps)


# -----------------------------
# REALISTIC TIMESTAMP
# -----------------------------
def generate_event_time(current_date):
    return datetime(
        current_date.year,
        current_date.month,
        current_date.day,
        random.randint(9, 18),
        random.randint(0, 59),
        random.randint(0, 59)
    )


# -----------------------------
# 🔥 INITIALIZE INVENTORY IF EMPTY
# -----------------------------
def initialize_inventory_if_empty(inventory, daily):
    if inventory.empty:
        print("⚠ Inventory empty → initializing...")

        skus = daily["SKU"].unique()

        inventory = pd.DataFrame({
            "SKU": skus,
            "Product": skus,
            "Current_Stock": np.random.randint(100, 500, size=len(skus)),
            "Lead_Time_Days": np.random.randint(5, 10, size=len(skus)),
            "Stock_As_Of_Date": "2025-01-01"
        })

        inventory.to_csv(INVENTORY_FILE, index=False)

        print("✅ Inventory initialized")

    return inventory


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline(start_date, end_date):

    print("\n🚀 Starting pipeline\n")

    MODELS_DIR.mkdir(exist_ok=True)

    daily, forecast_df = load_data()
    dates = pd.date_range(start=start_date, end=end_date)

    # 🔥 Load inventory ONCE
    forecast, inventory = load_inventory_data()

    # 🔥 Fix empty inventory
    inventory = initialize_inventory_if_empty(inventory, daily)

    last_retrain = {}
    all_metrics = []

    for current_date in dates:

        print(f"\n📡 Processing {current_date.date()}")

        for sku in daily["SKU"].unique():

            sku_data = daily[daily["SKU"] == sku]
            history = sku_data[sku_data["Date"] < current_date]["Demand"]

            if len(history) < MIN_HISTORY:
                continue

            # COOLDOWN
            if sku in last_retrain:
                days_since = (current_date - last_retrain[sku]).days
                if days_since < COOLDOWN_DAYS:
                    continue

            # BASE MODEL
            base_model = train_model(history)

            actual = history.values[-7:]
            base_pred = predict(base_model, len(actual))
            base_error = compute_error(actual, base_pred)

            # RECENT MODEL
            recent_model = train_model(history.tail(30))
            recent_pred = predict(recent_model, len(actual))
            recent_error = compute_error(actual, recent_pred)

            # METRICS
            all_metrics.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "SKU": sku,
                "MAE": round(recent_error, 4)
            })

            # DRIFT
            if recent_error > base_error * DRIFT_THRESHOLD:

                event_time = generate_event_time(current_date)

                log_event(
                    "DRIFT",
                    f"Drift detected for {sku} ({base_error:.2f} → {recent_error:.2f})",
                    event_time
                )

                model_name = f"{sku}_{event_time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
                model_path = MODELS_DIR / model_name

                with open(model_path, "wb") as f:
                    pickle.dump(recent_model, f)

                log_event(
                    "RETRAIN",
                    f"{sku} retrained and saved model",
                    event_time
                )

                last_retrain[sku] = current_date

                print(f"✅ {sku} retrained")

            else:
                print(f"✔ {sku} stable")

        # -------------------------
        # INVENTORY UPDATE
        # -------------------------
        recs = generate_inventory_recommendations(
            forecast,
            inventory,
            current_date
        )

        if recs is not None and not recs.empty:

            # ✅ Update recommendation date
            recs["Stock_As_Of_Date"] = current_date.strftime("%Y-%m-%d")

            save_inventory(recs)

            # ✅ Update MASTER inventory date
            inventory["Stock_As_Of_Date"] = current_date.strftime("%Y-%m-%d")

            inventory.to_csv(INVENTORY_FILE, index=False)

            print("📦 Inventory updated")

        else:
            print("⚠ No recommendations generated")

    # -----------------------------
    # SAVE METRICS
    # -----------------------------
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)

        try:
            old_df = pd.read_csv(METRICS_FILE)
            metrics_df = pd.concat([old_df, metrics_df], ignore_index=True)
        except:
            pass

        metrics_df.to_csv(METRICS_FILE, index=False)

    print("\n✅ Pipeline completed\n")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)

    args = parser.parse_args()

    run_pipeline(args.start, args.end)