"""
api/main.py
───────────────────────────────────────────────────────────────────────────────
FastAPI backend — serves forecast, drift, and inventory data to the dashboard.

Endpoints
─────────
  GET /                    Health check
  GET /forecast            2026 demand forecast (filterable by SKU)
  GET /forecast/skus       List of all available SKU codes
  GET /drift               Logged concept-drift events
  GET /inventory           Inventory replenishment recommendations

Run locally
───────────
  uvicorn api.main:app --reload --port 8000

With Docker
───────────
  Handled by docker-compose (see docker-compose.yml)
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── Output file paths (written by pipeline.py) ────────────────────────────────
FORECAST_PATH  = Path("data/processed/forecast_2026.csv")
DRIFT_LOG_PATH = Path("data/drift_logs/drift_events.csv")
INVENTORY_PATH = Path("data/processed/inventory_recommendations.csv")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Demand Forecasting API",
    description=(
        "Drift-Aware Continuous Learning Framework for "
        "Demand Forecasting and Inventory Replenishment — M.Tech Project"
    ),
    version="1.0.0",
)

# Allow the React dev server (Vite default: 5173) and any deployed origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Internal helper ───────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame:
    """Load a CSV or raise HTTP 404 with a helpful message."""
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"File not found: {path}. "
                "Run `python pipeline.py` first to generate pipeline outputs."
            ),
        )
    return pd.read_csv(path)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Liveness probe — returns service status."""
    return {"status": "ok", "service": "demand-forecasting-api", "version": "1.0.0"}


@app.get("/forecast", tags=["Forecast"])
def get_forecast(
    sku: str | None = Query(
        default=None,
        description="Filter results to a single SKU code (e.g. ELEC-001)",
    ),
    limit: int = Query(
        default=2000,
        le=50000,
        description="Maximum number of rows to return",
    ),
):
    """
    Return 2026 demand forecast data.

    - Without `sku`: returns forecast for all SKUs (up to `limit` rows).
    - With `sku`: returns the full 365-day forecast for that SKU.

    Response columns: **Date**, **SKU**, **forecast_demand**
    """
    df = _read_csv(FORECAST_PATH)

    if sku:
        df = df[df["SKU"] == sku.upper()]
        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"SKU '{sku.upper()}' not found in forecast."
            )

    return df.head(limit).to_dict(orient="records")


@app.get("/forecast/skus", tags=["Forecast"])
def get_available_skus():
    """Return the list of all SKU codes present in the forecast file."""
    df = _read_csv(FORECAST_PATH)
    return sorted(df["SKU"].unique().tolist())


@app.get("/drift", tags=["Drift"])
def get_drift_events(
    sku: str | None = Query(
        default=None,
        description="Filter drift events to a specific SKU",
    ),
):
    """
    Return logged concept-drift events.

    Returns an empty list if no drift has been detected yet (not an error).

    Response columns: **timestamp**, **sku**, **date**, **rolling_mae**,
    **threshold**, **drift_detected**
    """
    if not DRIFT_LOG_PATH.exists():
        return []   # pipeline has not streamed yet — normal state

    df = _read_csv(DRIFT_LOG_PATH)

    if sku:
        df = df[df["sku"] == sku.upper()]

    return df.to_dict(orient="records")


@app.get("/inventory", tags=["Inventory"])
def get_inventory(
    needs_order: bool = Query(
        default=False,
        description="When true, return only SKUs that require replenishment",
    ),
):
    """
    Return inventory replenishment recommendations.

    Response columns: **SKU**, **Current_Stock**, **Safety_Stock**,
    **Reorder_Point**, **Recommended_Order_Qty**, **Recommended_Order_Date**
    """
    df = _read_csv(INVENTORY_PATH)

    if needs_order:
        df = df[df["Recommended_Order_Qty"] > 0]

    return df.to_dict(orient="records")
