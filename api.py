from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import subprocess
import sys

from src.config import (
    METRICS_FILE,
    EVENT_LOG_FILE,
    INVENTORY_RECOMMENDATIONS_FILE,
    PROJECT_ROOT,
    ORDERS_FILE,
    INVENTORY_FILE
)

from src.event_logger import log_event

app = FastAPI(title="Drift-Aware Retail API")

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
# HELPERS
# ---------------------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()


def normalize_event_columns(df):
    if df.empty:
        return df

    normalized = df.copy()

    if "timestamp" not in normalized.columns and "date" in normalized.columns:
        normalized["timestamp"] = normalized["date"]

    if "timestamp" not in normalized.columns:
        normalized["timestamp"] = pd.NaT

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")

    if "event_type" not in normalized.columns:
        normalized["event_type"] = "UNKNOWN"

    if "message" not in normalized.columns:
        normalized["message"] = ""

    return normalized


def get_pipeline_reference_date(sku=None):
    inventory_df = safe_read_csv(INVENTORY_FILE)
    metrics_df = safe_read_csv(METRICS_FILE)

    # 1) Prefer inventory snapshot date for requested SKU.
    if not inventory_df.empty and "Stock_As_Of_Date" in inventory_df.columns:
        scoped = inventory_df

        if sku and "SKU" in inventory_df.columns:
            scoped = inventory_df[inventory_df["SKU"] == sku]
            if scoped.empty:
                scoped = inventory_df

        inv_date = pd.to_datetime(scoped["Stock_As_Of_Date"], errors="coerce").max()
        if pd.notna(inv_date):
            return inv_date

    # 2) Fall back to latest metrics date.
    if not metrics_df.empty and "Date" in metrics_df.columns:
        metrics_date = pd.to_datetime(metrics_df["Date"], errors="coerce").max()
        if pd.notna(metrics_date):
            return metrics_date

    # 3) Last resort.
    return pd.Timestamp.now()

# ---------------------------
# HOME
# ---------------------------
@app.get("/")
def home():
    return {"message": "API running 🚀"}

# ---------------------------
# RUN PIPELINE
# ---------------------------
@app.post("/run_pipeline")
def run_pipeline(start: str, end: str):
    started_at = datetime.now()
    log_event("PIPELINE_START", f"Pipeline started for range {start} to {end}", started_at)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline.py"),
        "--start", start,
        "--end", end
    ]

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        failed_at = datetime.now()
        log_event("PIPELINE_FAILED", f"Pipeline failed for range {start} to {end}", failed_at)
        raise HTTPException(status_code=500, detail="Pipeline execution failed") from exc

    completed_at = datetime.now()
    log_event("PIPELINE_COMPLETE", f"Pipeline completed for range {start} to {end}", completed_at)

    return {
        "status": "Pipeline executed",
        "started_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_at": completed_at.strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/pipeline_status")
def pipeline_status():
    events = normalize_event_columns(safe_read_csv(EVENT_LOG_FILE))
    metrics = safe_read_csv(METRICS_FILE)

    if events.empty:
        return {
            "last_run_at": None,
            "last_event_type": None,
            "last_event_message": None,
            "latest_metrics_date": None
        }

    pipeline_events = events[
        events["event_type"].isin(["PIPELINE_START", "PIPELINE_COMPLETE", "PIPELINE_FAILED"])
    ].sort_values("timestamp")

    last_any_event = events.sort_values("timestamp").tail(1)
    last_pipeline_event = pipeline_events.tail(1)

    latest_metrics_date = None
    if not metrics.empty and "Date" in metrics.columns:
        latest_metrics = pd.to_datetime(metrics["Date"], errors="coerce").max()
        if pd.notna(latest_metrics):
            latest_metrics_date = latest_metrics.strftime("%Y-%m-%d")

    source = last_pipeline_event if not last_pipeline_event.empty else last_any_event
    row = source.iloc[0]

    timestamp = row["timestamp"]
    last_run_at = timestamp.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(timestamp) else None

    return {
        "last_run_at": last_run_at,
        "last_event_type": str(row.get("event_type", "")),
        "last_event_message": str(row.get("message", "")),
        "latest_metrics_date": latest_metrics_date
    }

# ---------------------------
# METRICS
# ---------------------------
@app.get("/metrics")
def get_metrics(sku: str = None, start: str = None, end: str = None):

    df = safe_read_csv(METRICS_FILE)

    if df.empty:
        return []

    df["Date"] = pd.to_datetime(df["Date"])

    if sku:
        df = df[df["SKU"] == sku]

    if start:
        df = df[df["Date"] >= start]

    if end:
        df = df[df["Date"] <= end]

    return df.sort_values("Date").to_dict(orient="records")


@app.get("/skus")
def get_skus():
    metrics = safe_read_csv(METRICS_FILE)
    inventory = safe_read_csv(INVENTORY_RECOMMENDATIONS_FILE)

    sku_values = []

    if not metrics.empty and "SKU" in metrics.columns:
        sku_values.extend(metrics["SKU"].dropna().astype(str).tolist())

    if not inventory.empty and "SKU" in inventory.columns:
        sku_values.extend(inventory["SKU"].dropna().astype(str).tolist())

    unique_sorted = sorted({value.strip().upper() for value in sku_values if value and value.strip()})
    return {"skus": unique_sorted}

# ---------------------------
# EVENTS
# ---------------------------
@app.get("/events")
def get_events(sku: str = None, start: str = None, end: str = None):

    df = normalize_event_columns(safe_read_csv(EVENT_LOG_FILE))

    if df.empty:
        return []

    if start:
        df = df[df["timestamp"] >= start]

    if end:
        df = df[df["timestamp"] <= end]

    if sku:
        df = df[df["message"].str.contains(sku, case=False, na=False)]

    return df.sort_values("timestamp").to_dict(orient="records")

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

    try:
        orders_df = pd.read_csv(ORDERS_FILE)
    except:
        orders_df = pd.DataFrame(columns=["SKU", "Order_Qty", "Order_Date", "Restock_Date"])

    try:
        inv = pd.read_csv(INVENTORY_FILE)
        lead = int(inv.loc[inv["SKU"] == sku, "Lead_Time_Days"].iloc[0])
    except:
        lead = 7

    reference_date = get_pipeline_reference_date(sku)
    order_date = pd.Timestamp(reference_date).normalize()
    restock_date = order_date + pd.Timedelta(days=lead)

    new_order = {
        "SKU": sku,
        "Order_Qty": qty,
        "Order_Date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
        "Restock_Date": restock_date.strftime("%Y-%m-%d %H:%M:%S")
    }

    orders_df = pd.concat([orders_df, pd.DataFrame([new_order])], ignore_index=True)
    orders_df.to_csv(ORDERS_FILE, index=False)

    log_event("ORDER", f"{sku} order placed for {qty} units", order_date)

    return {
        "status": "Order placed",
        "sku": sku,
        "qty": qty,
        "order_date": new_order["Order_Date"],
        "restock_date": new_order["Restock_Date"]
    }