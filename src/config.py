from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
MODELS_DIR = PROJECT_ROOT / "models"

FORECAST_DAYS = 90

STREAM_START_DATE = "2025-02-01"
STREAM_END_DATE = "2025-04-01"

DRIFT_THRESHOLD = 0.35

RAW_SALES_FILE = RAW_DIR / "sales_with_sku.csv"
DAILY_DEMAND_FILE = PROCESSED_DIR / "daily_demand.csv"
METRICS_FILE = PROCESSED_DIR / "metrics.csv"
EVENT_LOG_FILE = PROCESSED_DIR / "system_events.csv"
INVENTORY_FILE = PROCESSED_DIR / "inventory_master.csv"
FORECAST_FILE = PROCESSED_DIR / "forecast_2025.csv"
ORDERS_FILE = PROCESSED_DIR / "orders.csv"
INVENTORY_RECOMMENDATIONS_FILE = PROCESSED_DIR / "inventory_recommendations.csv"