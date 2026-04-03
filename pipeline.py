import pandas as pd
import numpy as np
import pickle
import random
from datetime import datetime
from pathlib import Path

from src.config import (
    DAILY_DEMAND_FILE,
    FORECAST_FILE,
    MODELS_DIR
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
COOLDOWN_DAYS = 5   # ✅ FINAL CHANGE


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
# ERROR
# -----------------------------
def compute_error(actual, predicted):
    return np.mean(np.abs(actual - predicted))


# -----------------------------
# MODEL
# -----------------------------
def train_model(data):
    return data.mean()


def predict(model, steps):
    return np.array([model] * steps)


# -----------------------------
# GENERATE REALISTIC TIMESTAMP
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
# PIPELINE
# -----------------------------
def run_pipeline(start_date, end_date):

    print("\n🚀 Starting pipeline\n")

    MODELS_DIR.mkdir(exist_ok=True)

    daily, forecast_df = load_data()
    dates = pd.date_range(start=start_date, end=end_date)

    # Track last retrain per SKU
    last_retrain = {}

    for current_date in dates:

        print(f"\n📡 Processing {current_date.date()}")

        for sku in daily["SKU"].unique():

            sku_data = daily[daily["SKU"] == sku]

            history = sku_data[sku_data["Date"] < current_date]["Demand"]

            if len(history) < MIN_HISTORY:
                continue

            # -------------------------
            # COOLDOWN CHECK
            # -------------------------
            if sku in last_retrain:
                days_since = (current_date - last_retrain[sku]).days
                if days_since < COOLDOWN_DAYS:
                    print(f"⏳ {sku} cooldown active ({days_since} days)")
                    continue

            # -------------------------
            # BASE MODEL
            # -------------------------
            base_model = train_model(history)

            actual = history.values[-7:]
            base_pred = predict(base_model, len(actual))
            base_error = compute_error(actual, base_pred)

            # -------------------------
            # RECENT MODEL
            # -------------------------
            recent_model = train_model(history.tail(30))

            recent_pred = predict(recent_model, len(actual))
            recent_error = compute_error(actual, recent_pred)

            # -------------------------
            # DRIFT DETECTION
            # -------------------------
            if recent_error > base_error * DRIFT_THRESHOLD:

                event_time = generate_event_time(current_date)

                log_event(
                    "DRIFT",
                    f"Drift detected for {sku} ({base_error:.2f} → {recent_error:.2f})",
                    event_time
                )

                # -------------------------
                # RETRAIN
                # -------------------------
                model_name = f"{sku}_{event_time.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
                model_path = MODELS_DIR / model_name

                with open(model_path, "wb") as f:
                    pickle.dump(recent_model, f)

                log_event(
                    "RETRAIN",
                    f"{sku} retrained and saved model",
                    event_time
                )

                print(f"✅ {sku} retrained & saved")

                # Update cooldown tracker
                last_retrain[sku] = current_date

            else:
                print(f"✔ {sku} stable")

        # -------------------------
        # INVENTORY UPDATE
        # -------------------------
        forecast, inventory = load_inventory_data()

        if forecast.empty or inventory.empty:
            print("⚠ Inventory skipped (missing data)")
            continue

        recs = generate_inventory_recommendations(
            forecast,
            inventory,
            current_date
        )

        if recs is not None and not recs.empty:
            save_inventory(recs)
            print("📦 Inventory updated")
        else:
            print("⚠ No inventory recommendations generated")

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