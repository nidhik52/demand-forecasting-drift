import pandas as pd
from src.event_logger import log_event
from src.config import (
    FORECAST_FILE,
    INVENTORY_FILE,
    INVENTORY_RECOMMENDATIONS_FILE,
    ORDERS_FILE,
)

# Map Stock_Status values written by generate_inventory.py → Risk_Level used by dashboard
_STATUS_TO_RISK = {
    "insufficient": "CRITICAL",
    "borderline":   "WARNING",
    "sufficient":   "SAFE",
}


def load_data():
    forecast  = pd.read_csv(FORECAST_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    forecast["Date"]               = pd.to_datetime(forecast["Date"])
    inventory["Stock_As_Of_Date"]  = pd.to_datetime(inventory["Stock_As_Of_Date"])

    return forecast, inventory


def generate_inventory_recommendations(forecast, inventory, current_date):
    """
    Build one recommendation row per SKU.

    Risk_Level is derived from Stock_Status when present (ensures the
    sufficient / borderline / insufficient mix set by generate_inventory is
    preserved).  Falling back to demand-based logic when the column is absent
    keeps backward-compatibility with older inventory files.
    """
    current_date = pd.to_datetime(current_date)
    recs = []

    for _, item in inventory.iterrows():
        sku        = item["SKU"]
        stock      = max(0.0, float(item.get("Current_Stock", 0) or 0))
        lead_time  = int(item.get("Lead_Time_Days", 7))
        safety_stk = int(item.get("Safety_Stock", 10))
        in_transit = int(item.get("In_Transit", 0) or 0)

        horizon_end  = current_date + pd.Timedelta(days=lead_time)
        sku_forecast = forecast[
            (forecast["SKU"]  == sku)
            & (forecast["Date"] >  current_date)
            & (forecast["Date"] <= horizon_end)
        ]

        if sku_forecast.empty:
            demand          = float(lead_time * 10)   # sensible fallback
            recommended_qty = int(max(0, demand - stock))
        else:
            demand          = float(sku_forecast["Forecast_Demand"].clip(lower=0).sum())
            if demand == 0:
                demand = 0.01
            safety_demand   = 0.2 * demand
            reorder_point   = demand + safety_demand
            recommended_qty = int(max(0, reorder_point - stock))

        # Prefer the explicit status column set during inventory generation
        stock_status = str(item.get("Stock_Status", "")).strip().lower()
        if stock_status in _STATUS_TO_RISK:
            risk = _STATUS_TO_RISK[stock_status]
        else:
            # Fallback: derive from stock vs demand
            if stock <= 0:
                risk = "CRITICAL"
            elif stock < demand:
                risk = "CRITICAL"
            elif stock < (demand * 1.2):
                risk = "WARNING"
            else:
                risk = "SAFE"

        recs.append({
            "Date":                  current_date.strftime("%Y-%m-%d"),
            "SKU":                   sku,
            "Current_Stock":         int(stock),
            "In_Transit":            in_transit,
            "Recommended_Order_Qty": recommended_qty,
            "Risk_Level":            risk,
            "Lead_Time_Days":        lead_time,
            "Safety_Stock":          safety_stk,
        })

    return pd.DataFrame(recs)


def save_inventory(df):
    INVENTORY_RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INVENTORY_RECOMMENDATIONS_FILE, index=False)