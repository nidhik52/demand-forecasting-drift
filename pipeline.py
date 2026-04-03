import pandas as pd
import os
from datetime import datetime
from prophet import Prophet
import joblib

from src.config import DAILY_DEMAND_FILE, FORECAST_FILE, INVENTORY_FILE, DRIFT_THRESHOLD
from src.event_logger import log_event


# ----------------------------------------
# LOAD DATA
# ----------------------------------------

def load_data():
    daily = pd.read_csv(DAILY_DEMAND_FILE)
    forecast = pd.read_csv(FORECAST_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    daily["Date"] = pd.to_datetime(daily["Date"])
    forecast["Date"] = pd.to_datetime(forecast["Date"])
    inventory["Stock_As_Of_Date"] = pd.to_datetime(inventory["Stock_As_Of_Date"])

    return daily, forecast, inventory


# ----------------------------------------
# DRIFT DETECTION
# ----------------------------------------

def detect_drift(actual, forecast):
    if actual == 0:
        return False

    error_ratio = abs(actual - forecast) / actual
    return error_ratio > DRIFT_THRESHOLD


# ----------------------------------------
# RETRAIN MODEL
# ----------------------------------------

def retrain_model(sku_df, sku, current_date):
    print(f"🔁 Retraining model for {sku}")

    if len(sku_df) < 10:
        print(f"⚠ Not enough data for {sku}, skipping retrain")
        return None

    prophet_df = sku_df.rename(columns={"Date": "ds", "Demand": "y"})

    model = Prophet()
    model.fit(prophet_df)

    model_name = f"models/prophet_{sku}_{current_date.date()}.pkl"
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, model_name)

    print(f"✅ Model saved → {model_name}")

    return model


# ----------------------------------------
# UPDATE FORECAST
# ----------------------------------------

def update_forecast(model, sku, sku_name):
    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)

    out = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Forecast_Demand"})
    out["SKU"] = sku
    out["SKU_Name"] = sku_name

    return out


# ----------------------------------------
# INVENTORY UPDATE (FIXED)
# ----------------------------------------

def update_inventory(inventory, daily_slice, forecast_slice, current_date):

    for _, row in daily_slice.iterrows():
        sku = row["SKU"]
        demand = row["Demand"]

        idx = inventory[inventory["SKU"] == sku].index[0]

        # reduce stock
        inventory.loc[idx, "Current_Stock"] -= demand
        inventory.loc[idx, "Current_Stock"] = max(0, inventory.loc[idx, "Current_Stock"])

        # update stock date
        inventory.loc[idx, "Stock_As_Of_Date"] = current_date

        # forecast demand next 7 days
        f = forecast_slice[forecast_slice["SKU"] == sku].head(7)
        future_demand = f["Forecast_Demand"].sum()

        current_stock = inventory.loc[idx, "Current_Stock"]

        if current_stock < future_demand:
            order_qty = int(future_demand - current_stock)

            inventory.loc[idx, "Current_Stock"] += order_qty

            log_event(
                event_type="RESTOCK",
                message=f"{sku} restocked {order_qty}",
                date=current_date
            )

    return inventory


# ----------------------------------------
# MAIN PIPELINE
# ----------------------------------------

def run_pipeline(start_date, end_date):

    print("\n🚀 Starting pipeline\n")

    daily, forecast, inventory = load_data()

    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    metrics = []

    while current_date <= end_date:

        print(f"\n📡 Streaming {current_date.date()}")

        daily_slice = daily[daily["Date"] == current_date]
        forecast_slice = forecast[forecast["Date"] == current_date]

        for _, row in daily_slice.iterrows():

            sku = row["SKU"]
            actual = row["Demand"]

            pred_row = forecast_slice[forecast_slice["SKU"] == sku]

            if pred_row.empty:
                continue

            predicted = pred_row["Forecast_Demand"].values[0]

            error = abs(actual - predicted)

            metrics.append({
                "Date": current_date,
                "SKU": sku,
                "Actual": actual,
                "Predicted": predicted,
                "Error": error
            })

            # DRIFT
            if detect_drift(actual, predicted):

                print(f"⚠ Drift detected for {sku}")

                log_event(
                    event_type="DRIFT",
                    message=f"Drift detected for {sku}",
                    date=current_date
                )

                sku_df = daily[(daily["SKU"] == sku) & (daily["Date"] <= current_date)]

                model = retrain_model(sku_df, sku, current_date)

                if model:
                    new_forecast = update_forecast(model, sku, row["SKU_Name"])

                    forecast = forecast[forecast["SKU"] != sku]
                    forecast = pd.concat([forecast, new_forecast])

                    log_event(
                        event_type="RETRAIN",
                        message=f"{sku} retrained",
                        date=current_date
                    )

        # INVENTORY UPDATE
        inventory = update_inventory(inventory, daily_slice, forecast, current_date)

        current_date += pd.Timedelta(days=1)

    # SAVE FILES
    pd.DataFrame(metrics).to_csv("data/processed/metrics.csv", index=False)
    forecast.to_csv("data/processed/forecast_2025.csv", index=False)
    inventory.to_csv("data/processed/inventory_master.csv", index=False)

    print("\n✅ Pipeline completed")


# ----------------------------------------
# CLI
# ----------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)

    args = parser.parse_args()

    run_pipeline(args.start, args.end)