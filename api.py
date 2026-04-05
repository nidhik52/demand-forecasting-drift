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
    inv_df = safe_read_csv(INVENTORY_FILE, ["SKU", "Product"])
    if not inv_df.empty and "SKU" in inv_df.columns:
        inv_df = inv_df.drop_duplicates(subset=["SKU"]).sort_values("SKU")
        return inv_df[["SKU", "Product"]].fillna("").to_dict(orient="records")

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

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")
    start_dt = pd.to_datetime(start, errors="coerce")
    end_dt = pd.to_datetime(end, errors="coerce")

    if pd.isna(start_dt) or pd.isna(end_dt):
        return []

    df = df.dropna(subset=["Date"])

    df = df[
        (df["SKU"] == sku) &
        (df["Date"] >= start_dt) &
        (df["Date"] <= end_dt)
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

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    df = df[
        (df["timestamp"] >= start_dt) &
        (df["timestamp"] <= end_dt) &
        (df["message"].str.contains(sku))
    ]

    return df.sort_values("timestamp", ascending=False).to_dict(orient="records")


# ---------------------------
# INVENTORY
# ---------------------------
@app.get("/inventory")
def get_inventory(end: str):
    df = safe_read_csv(INVENTORY_RECOMMENDATIONS_FILE)

    if df.empty:
        return []

    df["Stock_As_Of_Date"] = pd.to_datetime(df["Stock_As_Of_Date"])
    end_dt = pd.to_datetime(end)

    df = df[df["Stock_As_Of_Date"] <= end_dt]

    return (
        df.sort_values("Stock_As_Of_Date", ascending=False)
        .drop_duplicates("SKU")
        .to_dict(orient="records")
    )


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
        inv = None
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

    if inv is not None:
        inv.loc[inv["SKU"] == sku, "Current_Stock"] -= int(qty)
        inv.to_csv(INVENTORY_FILE, index=False)

    log_event("ORDER", f"{sku} order placed for {qty} units", order_date)

    return {
        "status": "success",
        "restock_date": restock_date.strftime("%Y-%m-%d")
    }


########################################################
#  Monitoring using Grafana
########################################################

@app.get("/monitoring")
def monitoring():

    def safe_read(path):
        try:
            return pd.read_csv(path)
        except:
            return pd.DataFrame()

    metrics = safe_read(METRICS_FILE)
    drift = safe_read("data/processed/drift_summary.csv")
    runs = safe_read("data/processed/pipeline_runs.csv")

    return {
        "total_records": len(metrics),
        "avg_mae": float(metrics["MAE"].mean()) if not metrics.empty else 0,
        "drift_count": len(drift),
        "pipeline_runs": len(runs),
        "last_run": runs.tail(1).to_dict(orient="records") if not runs.empty else []
    }


if FRONTEND_BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="frontend")