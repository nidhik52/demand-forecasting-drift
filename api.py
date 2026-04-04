from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import subprocess
import sys
from pathlib import Path
from fastapi.staticfiles import StaticFiles

from src.config import (
    EVENT_LOG_FILE,
    INVENTORY_RECOMMENDATIONS_FILE,
    METRICS_FILE,
    PROJECT_ROOT,
    ORDERS_FILE,
    INVENTORY_FILE
)
from src.event_logger import log_event

app = FastAPI()
FRONTEND_BUILD_DIR = PROJECT_ROOT / "dashboard" / "build"

# ---------------------------
# CORS (IMPORTANT)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# SAFE CSV LOADER
# ---------------------------
def safe_read_csv(path, columns=None):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame(columns=columns if columns else [])


# ---------------------------
# RUN PIPELINE
# ---------------------------
@app.post("/run_pipeline")
def run_pipeline(start: str, end: str):
    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "pipeline.py"),
            "--start", start,
            "--end", end
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        return {
            "status": "success",
            "logs": result.stdout[-1000:]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------
# GET SKUs (FIX DROPDOWN)
# ---------------------------
@app.get("/skus")
def get_skus():
    df = safe_read_csv(METRICS_FILE, ["SKU"])
    if df.empty:
        return []
    return sorted(df["SKU"].unique().tolist())


# ---------------------------
# METRICS
# ---------------------------
@app.get("/metrics")
def get_metrics(sku: str, start: str, end: str):
    df = safe_read_csv(METRICS_FILE)

    if df.empty:
        return []

    df["Date"] = pd.to_datetime(df["Date"])

    df = df[
        (df["SKU"] == sku) &
        (df["Date"] >= start) &
        (df["Date"] <= end)
    ]

    return df.sort_values("Date").to_dict(orient="records")


# ---------------------------
# EVENTS
# ---------------------------
@app.get("/events")
def get_events(sku: str, start: str, end: str):
    df = safe_read_csv(EVENT_LOG_FILE)

    if df.empty:
        return []

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end) &
        (df["message"].str.contains(sku))
    ]

    return df.sort_values("timestamp", ascending=False).to_dict(orient="records")


# ---------------------------
# INVENTORY
# ---------------------------
@app.get("/inventory")
def get_inventory():
    df = safe_read_csv(INVENTORY_RECOMMENDATIONS_FILE)
    return df.to_dict(orient="records")


# ---------------------------
# PLACE ORDER
# ---------------------------
@app.post("/order")
def place_order(sku: str, qty: int):

    ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)

    orders_df = safe_read_csv(
        ORDERS_FILE,
        ["SKU", "Order_Qty", "Order_Date", "Restock_Date"]
    )

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

    log_event("ORDER", f"{sku} order placed for {qty} units", order_date)

    return {"status": "success"}


if FRONTEND_BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="frontend")