import pandas as pd
from src.event_logger import log_event
from src.config import (
    FORECAST_FILE,
    INVENTORY_FILE,
    INVENTORY_RECOMMENDATIONS_FILE,
    ORDERS_FILE,
)

def load_data():
    forecast = pd.read_csv(FORECAST_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    forecast["Date"] = pd.to_datetime(forecast["Date"])
    inventory["Stock_As_Of_Date"] = pd.to_datetime(inventory["Stock_As_Of_Date"])

    return forecast, inventory


def generate_inventory_recommendations(forecast, inventory, current_date):

    current_date = pd.to_datetime(current_date)
    recs = []

    for _, item in inventory.iterrows():
        sku = item["SKU"]
        product = item.get("Product", "")
        stock = max(0.0, float(item.get("Current_Stock", 0) or 0))
        lead_time = int(item.get("Lead_Time_Days", 7))
        horizon_end = current_date + pd.Timedelta(days=lead_time)
        sku_forecast = forecast[
            (forecast["SKU"] == sku)
            & (forecast["Date"] > current_date)
            & (forecast["Date"] <= horizon_end)
        ]
        if sku_forecast.empty:
            # No forecast for this SKU in the window
            demand = 0
            safety_stock = 0
            reorder_point = 0
            recommended_qty = 0
            risk = "NO DATA"
            recommendation = "No forecast available for this SKU"
        else:
            demand = sku_forecast["Forecast_Demand"].clip(lower=0).sum()
            if demand == 0:
                demand = 0.01
            safety_stock = 0.2 * demand
            reorder_point = demand + safety_stock
            recommended_qty = int(max(0, reorder_point - stock))
            if stock <= 0:
                risk = "CRITICAL"
                recommendation = "URGENT: reorder immediately"
            elif stock < demand:
                risk = "CRITICAL"
                recommendation = "Stock will run out"
            elif stock < reorder_point:
                risk = "WARNING"
                recommendation = "Plan reorder"
            else:
                risk = "SAFE"
                recommendation = "Stock sufficient"
        recs.append({
            "SKU": sku,
            "Product": product,
            "Current_Stock": int(stock),
            "Stock_As_Of_Date": current_date,
            "Recommended_Order_Qty": recommended_qty,
            "Risk_Level": risk,
            "Recommendation": recommendation
        })

    return pd.DataFrame(recs)


def save_inventory(df):
    INVENTORY_RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INVENTORY_RECOMMENDATIONS_FILE, index=False)