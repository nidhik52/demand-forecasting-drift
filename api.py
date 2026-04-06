from fastapi import FastAPI
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

def safe_read_csv(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame(columns=columns or [])

def _as_int(value: Optional[int], default: int) -> int:
    return int(value) if value is not None else default

@app.get("/skus")
def get_skus() -> List[Dict]:
    session = SessionLocal()
    inv = session.query(Inventory).all()
    session.close()
    if inv:
        return [{"SKU": i.sku, "Product": f"Product_{i.sku}"} for i in inv]
    df = safe_read_csv(METRICS_FILE, ["SKU"])
    if df.empty:
        return []
    return [{"SKU": s} for s in df["SKU"].unique()]

@app.get("/metrics")
def get_metrics(sku: str, start: str, end: str) -> List[Dict]:
    df = safe_read_csv(METRICS_FILE)
    if df.empty:
        return []
    df["Date"] = pd.to_datetime(df["Date"])
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    filtered = df[(df["SKU"] == sku) & (df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    return filtered.sort_values("Date").to_dict("records")

@app.get("/events")
def get_events(sku: str, start: str, end: str) -> List[Dict]:
    df = safe_read_csv(EVENT_LOG_FILE)
    if df.empty:
        return []
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    filtered = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt) & (df["message"].str.contains(sku))]
    return filtered.sort_values("timestamp", ascending=False).to_dict("records")

@app.get("/inventory")
def get_inventory(end: str) -> List[Dict]:
    session = SessionLocal()
    inv = session.query(Inventory).all()
    session.close()
    result = []
    for i in inv:
        current = _as_int(cast(Optional[int], i.current_stock), 0)
        safety = _as_int(cast(Optional[int], i.safety_stock), 10)
        risk = "CRITICAL" if current < safety // 2 else "WARNING" if current < safety else "SAFE"
        result.append({
            "SKU": i.sku,
            "Current_Stock": current,
            "In_Transit": _as_int(cast(Optional[int], i.in_transit), 0),
            "Lead_Time_Days": _as_int(cast(Optional[int], i.lead_time_days), 7),
            "Safety_Stock": safety,
            "Risk_Level": risk
        })
    return result

@app.post("/order")
def place_order(sku: str, qty: int, end: Optional[str] = None) -> Dict:
    session = SessionLocal()
    inv = session.query(Inventory).filter_by(sku=sku).first()
    if not inv:
        inv = Inventory(sku=sku, current_stock=0, lead_time_days=7, safety_stock=10)
        session.add(inv)
        session.commit()
        session.refresh(inv)
    in_transit = _as_int(cast(Optional[int], inv.in_transit), 0)
    inv.in_transit = in_transit + qty  # type: ignore
    order_date = pd.to_datetime(end).to_pydatetime() if end else datetime.utcnow()
    lead = _as_int(cast(Optional[int], inv.lead_time_days), 7)
    restock_date = order_date + timedelta(days=lead)
    order = Order(sku=sku, order_qty=qty, order_date=order_date, restock_date=restock_date, received=0)
    session.add(order)
    session.commit()
    session.close()
    # Safe event logging
    try:
        from src.event_logger import log_event
        log_event("ORDER", f"{sku} order placed for {qty} units (restock {restock_date.date()})", order_date)
    except ImportError:
        pass
    return {"status": "success", "restock_date": restock_date.strftime("%Y-%m-%d")}

@app.post("/run_pipeline")
def run_pipeline(start: str, end: str, model: str = "prophet") -> Dict:
    try:
        cmd = [sys.executable, "pipeline.py", "--start", start, "--end", end]
        env = os.environ.copy()
        env["MODEL_TYPE"] = "prophet" if model == "prophet" else "baseline"
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            return {"status": "error", "message": result.stderr[-1000:] or "Pipeline failed"}
        return {"status": "success", "logs": result.stdout[-1000:]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/monitoring")
def monitoring() -> Dict:
    metrics = safe_read_csv(METRICS_FILE)
    return {
        "total_records": len(metrics),
        "avg_mae": float(metrics["MAE"].mean()) if not metrics.empty else 0,
        "pipeline_runs": len(safe_read_csv(Path("data/processed/pipeline_runs.csv")))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)