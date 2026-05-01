from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, cast
import os

from src.config import PROJECT_ROOT, METRICS_FILE, EVENT_LOG_FILE
from src.db import SessionLocal, Inventory, Order


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve React dashboard static files from /dashboard/build if present
import os
import pathlib
from fastapi.staticfiles import StaticFiles
dashboard_build_dir = pathlib.Path(__file__).parent / "dashboard" / "build"
if dashboard_build_dir.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_build_dir), html=True), name="dashboard")
else:
    import logging
    logging.error("React build not found at %s. Please run 'npm run build' in dashboard/ before building the Docker image.", dashboard_build_dir)
    # FIX 10: log warning instead of crashing — Render raw deploy may not have build dir
    import logging
    logging.warning(
        "React build not found at %s. "
        "Dashboard will not be served. API endpoints are still available.",
        dashboard_build_dir
    )

# Root endpoint: redirect to /dashboard
from fastapi.responses import RedirectResponse, PlainTextResponse
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/dashboard")

# Health check endpoint
@app.get("/health", include_in_schema=False)
def health():
    if not dashboard_build_dir.is_dir():
        raise HTTPException(status_code=500, detail="React dashboard build missing")
    return {"status": "ok"}

INVENTORY_FILE = PROJECT_ROOT / "data" / "processed" / "inventory_recommendations.csv"

def safe_read_csv(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns or [])

def _as_int(value, default: int) -> int:
    return int(value) if value is not None else default

# ----------------------------
# /skus
# Returns list of unique SKUs from metrics.csv
# ----------------------------
@app.get("/skus")
def get_skus() -> List[Dict]:
    df = safe_read_csv(METRICS_FILE, ["SKU"])
    if df.empty:
        # fallback to DB
        session = SessionLocal()
        inv = session.query(Inventory).all()
        session.close()
        if inv:
            return [{"SKU": i.sku, "Product": f"Product_{i.sku}"} for i in inv]
        return []
    return [{"SKU": s, "Product": s} for s in sorted(df["SKU"].unique())]

# ----------------------------
# /metrics
# FIX: date filtering uses sim dates (match pipeline output)
# ----------------------------
@app.get("/metrics")
def get_metrics(sku: str, start: str, end: str) -> List[Dict]:
    df = safe_read_csv(METRICS_FILE)
    if df.empty:
        return []
    df["Date"] = pd.to_datetime(df["Date"])
    start_dt   = pd.to_datetime(start)
    end_dt     = pd.to_datetime(end)
    filtered   = df[
        (df["SKU"] == sku) &
        (df["Date"] >= start_dt) &
        (df["Date"] <= end_dt)
    ].sort_values("Date")
    # Ensure Drift is integer for JS
    if "Drift" in filtered.columns:
        filtered["Drift"] = filtered["Drift"].fillna(0).astype(int)
    return filtered.to_dict("records")

# ----------------------------
# /events
# FIX: filter by sim date column (event_date), not timestamp
# events.csv has: timestamp (HH:MM:SS variation), event_type, message
# We parse the DATE portion of timestamp for filtering
# ----------------------------
@app.get("/events")
def get_events(sku: str, start: str, end: str) -> List[Dict]:
    df = safe_read_csv(EVENT_LOG_FILE)
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Extract just the date for range filtering
    df["_date"]     = df["timestamp"].dt.normalize()
    start_dt        = pd.to_datetime(start)
    end_dt          = pd.to_datetime(end)
    filtered = df[
        (df["_date"] >= start_dt) &
        (df["_date"] <= end_dt) &
        (df["message"].str.contains(sku, na=False))
    ].drop(columns=["_date"])
    return filtered.sort_values("timestamp", ascending=False).to_dict("records")

# ----------------------------
# /drift-events
# Returns only DRIFT-type events for a SKU + date range
# Used by App.js to plot red drift markers on chart
# ----------------------------
@app.get("/drift-events")
def get_drift_events(sku: str, start: str, end: str) -> List[Dict]:
    df = safe_read_csv(EVENT_LOG_FILE)
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_date"]     = df["timestamp"].dt.normalize()
    start_dt        = pd.to_datetime(start)
    end_dt          = pd.to_datetime(end)
    filtered = df[
        (df["event_type"].str.upper() == "DRIFT") &
        (df["_date"] >= start_dt) &
        (df["_date"] <= end_dt) &
        (df["message"].str.contains(sku, na=False))
    ].drop(columns=["_date"])
    # Return date string for chart matching
    filtered = filtered.copy()
    filtered["date"] = filtered["timestamp"].dt.strftime("%Y-%m-%d")
    return filtered.to_dict("records")

# ----------------------------
# /inventory
# FIX: reads from CSV (written by pipeline), not DB
# DB is only used as fallback when CSV doesn't exist yet
# ----------------------------
@app.get("/inventory")
def get_inventory(end: Optional[str] = None) -> List[Dict]:
    # Try latest pipeline CSV output first
    if INVENTORY_FILE.exists():
        df = safe_read_csv(INVENTORY_FILE)
        if not df.empty:
            # Get most recent record per SKU
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                if end:
                    df = df[df["Date"] <= pd.to_datetime(end)]
                df = df.sort_values("Date").groupby("SKU").last().reset_index()
            return df.to_dict("records")

    # Fallback to DB
    session = SessionLocal()
    inv     = session.query(Inventory).all()
    session.close()
    result  = []
    for i in inv:
        current  = _as_int(cast(Optional[int], i.current_stock), 0)
        safety   = _as_int(cast(Optional[int], i.safety_stock), 10)
        risk     = "CRITICAL" if current < safety // 2 else "WARNING" if current < safety else "SAFE"
        result.append({
            "SKU":                  i.sku,
            "Current_Stock":        current,
            "In_Transit":           _as_int(cast(Optional[int], i.in_transit), 0),
            "Lead_Time_Days":       _as_int(cast(Optional[int], i.lead_time_days), 7),
            "Safety_Stock":         safety,
            "Risk_Level":           risk,
            "Recommended_Order_Qty": max(0, safety - current),
        })
    return result

# ----------------------------
# /order (unchanged — DB write)
# ----------------------------
@app.post("/order")
def place_order(sku: str, qty: int, end: Optional[str] = None) -> Dict:
    session = SessionLocal()
    inv = session.query(Inventory).filter_by(sku=sku).first()
    if not inv:
        inv = Inventory(sku=sku, current_stock=0, lead_time_days=7, safety_stock=10)
        session.add(inv)
        session.commit()
        session.refresh(inv)
    in_transit   = _as_int(cast(Optional[int], inv.in_transit), 0)
    inv.in_transit = in_transit + qty  # type: ignore
    order_date   = pd.to_datetime(end).to_pydatetime() if end else datetime.utcnow()
    lead         = _as_int(cast(Optional[int], inv.lead_time_days), 7)
    restock_date = order_date + timedelta(days=lead)
    order = Order(sku=sku, order_qty=qty, order_date=order_date,
                  restock_date=restock_date, received=0)
    session.add(order)
    session.commit()
    session.close()
    try:
        from src.event_logger import log_event
        log_event("ORDER", f"{sku} order placed for {qty} units (restock {restock_date.date()})", order_date)
    except ImportError:
        pass
    return {
        "status":       "success",
        "sku":          sku,
        "qty":          qty,
        "restock_date": restock_date.strftime("%Y-%m-%d"),
        "message":      f"{qty} units ordered for {sku} and will arrive by {restock_date.strftime('%Y-%m-%d')}",
    }

# ----------------------------
# /run_pipeline
# FIX: pass drift_threshold arg to pipeline
# ----------------------------
@app.post("/run_pipeline")
def run_pipeline(start: str, end: str, model: str = "prophet",
                 drift_threshold: float = 2.0) -> Dict:
    try:
        cmd = [
            sys.executable, "pipeline.py",
            "--start", start,
            "--end",   end,
            "--drift_threshold", str(drift_threshold),
        ]
        env             = os.environ.copy()
        env["MODEL_TYPE"] = "prophet" if model == "prophet" else "baseline"
        subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)
        return {"status": "started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ----------------------------
# /monitoring
# FIX: adds drift_count from metrics CSV
# ----------------------------
@app.get("/monitoring")
def monitoring() -> Dict:
    metrics = safe_read_csv(METRICS_FILE)
    drift_count = 0
    avg_mae     = 0.0
    total       = len(metrics)
    if not metrics.empty:
        avg_mae     = float(metrics["MAE"].mean())
        if "Drift" in metrics.columns:
            drift_count = int(metrics["Drift"].sum())
    return {
        "total_records":  total,
        "avg_mae":        round(avg_mae, 4),
        "drift_count":    drift_count,
        "pipeline_runs":  total,   # proxy metric
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)