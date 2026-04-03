from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import subprocess
import sys
import datetime

from src.config import (
    EVENT_LOG_FILE,
    INVENTORY_RECOMMENDATIONS_FILE,
    METRICS_FILE,
    PROJECT_ROOT,
    FORECAST_FILE,
    INVENTORY_FILE,
    ORDERS_FILE
)

from src.event_logger import log_event

app = FastAPI()

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# RUN PIPELINE
# ---------------------------
@app.post("/run_pipeline")
def run_pipeline(start: str, end: str):
    cmd = [sys.executable, str(PROJECT_ROOT / "pipeline.py"), "--start", start, "--end", end]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    return {"message": f"Pipeline run from {start} to {end}"}


# ---------------------------
# FORECAST
# ---------------------------
@app.get("/forecast")
def get_forecast(sku: str, start: str, end: str):

    df = pd.read_csv(FORECAST_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        (df["SKU"] == sku) &
        (df["Date"] >= start) &
        (df["Date"] <= end)
    ]

    return df.sort_values("Date").to_dict(orient="records")


# ---------------------------
# METRICS (Actual vs Predicted)
# ---------------------------
@app.get("/metrics")
def get_metrics(sku: str, start: str, end: str):

    df = pd.read_csv(METRICS_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        (df["SKU"] == sku) &
        (df["Date"] >= start) &
        (df["Date"] <= end)
    ]

    return df.sort_values("Date").to_dict(orient="records")


# ---------------------------
# EVENTS (DRIFT + RETRAIN + ORDER)
# ---------------------------
@app.get("/events")
def get_events(sku: str, start: str, end: str):

    df = pd.read_csv(EVENT_LOG_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end) &
        (df["message"].str.contains(sku))
    ]

    return df.sort_values("timestamp").to_dict(orient="records")


# ---------------------------
# INVENTORY (RECOMMENDATIONS)
# ---------------------------
@app.get("/inventory")
def get_inventory():
    df = pd.read_csv(INVENTORY_RECOMMENDATIONS_FILE)
    return df.to_dict(orient="records")


# ---------------------------
# INVENTORY MASTER (ACTUAL STOCK)
# ---------------------------
@app.get("/inventory-master")
def get_inventory_master():
    df = pd.read_csv(INVENTORY_FILE)
    return df.to_dict(orient="records")


# ---------------------------
# PLACE ORDER
# ---------------------------
@app.post("/order")
def place_order(sku: str, qty: int):

    ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        orders_df = pd.read_csv(ORDERS_FILE)
    except:
        orders_df = pd.DataFrame(columns=["SKU", "Order_Qty", "Order_Date", "Restock_Date"])

    try:
        inv = pd.read_csv(INVENTORY_FILE)
        lead = int(inv.loc[inv["SKU"] == sku, "Lead_Time_Days"].iloc[0])
    except:
        lead = 7

    order_date = pd.Timestamp.now()
    restock_date = order_date + pd.Timedelta(days=lead)

    new_order = {
        "SKU": sku,
        "Order_Qty": int(qty),
        "Order_Date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
        "Restock_Date": restock_date.strftime("%Y-%m-%d %H:%M:%S"),
    }

    orders_df = pd.concat([orders_df, pd.DataFrame([new_order])], ignore_index=True)
    orders_df.to_csv(ORDERS_FILE, index=False)

    # LOG EVENT (FIXED TIMESTAMP)
    log_event(
        event_type="ORDER",
        message=f"{sku} order placed for {qty}",
        event_time=order_date
    )

    return {"message": "Order placed successfully"}


# ---------------------------
# GET ORDERS (for dashboard tab)
# ---------------------------
@app.get("/orders")
def get_orders():
    try:
        df = pd.read_csv(ORDERS_FILE)
        return df.to_dict(orient="records")
    except:
        return []