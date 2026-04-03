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


def compute_inventory(forecast, inventory, current_date):
    current_date = pd.to_datetime(current_date)
    recs = []

    for _, item in inventory.iterrows():
        sku = item["SKU"]
        # Parse stock robustly: treat missing/invalid values as 0.0
        try:
            stock = float(item.get("Current_Stock", 0))
            if pd.isna(stock):
                stock = 0.0
        except Exception:
            stock = 0.0
        stock = max(0.0, stock)
        lead_time = int(item.get("Lead_Time_Days", 7))

        # demand = sum of forecast for the next `lead_time` days (inclusive)
        horizon_end = current_date + pd.Timedelta(days=max(lead_time - 1, 0))
        sku_forecast = forecast[
            (forecast["SKU"] == sku)
            & (forecast["Date"] >= current_date)
            & (forecast["Date"] <= horizon_end)
        ]

        horizon_values = sku_forecast["Forecast_Demand"].clip(lower=0)

        demand = float(horizon_values.sum()) if len(horizon_values) > 0 else 0.0

        safety_stock = 0.10 * demand
        required_stock = demand + safety_stock

        recommended_qty = int(max(0.0, required_stock - stock))

        # Zero or negative stock -> CRITICAL
        if stock <= 0.0:
            risk_level = "CRITICAL"
            recommendation = "URGENT: reorder immediately"
        elif stock < (0.5 * required_stock):
            risk_level = "WARNING"
            recommendation = "Plan reorder soon"
        else:
            risk_level = "SAFE"
            recommendation = "Stock sufficient"

        # Provide concise debug line (can be disabled by pipeline via silent flag)
        print(
            f"[INVENTORY_DEBUG] SKU={sku} current_stock={stock:.2f} "
            f"demand={demand:.2f} required_stock={required_stock:.2f} "
            f"recommended={recommended_qty}"
        )

        recs.append(
            {
                "SKU": sku,
                "Current_Stock": int(stock),
                "Stock_As_Of_Date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
                "Recommended_Order_Qty": recommended_qty,
                "Risk_Level": risk_level,
                "Recommendation": recommendation,
            }
        )

    return pd.DataFrame(recs)


def save_inventory(inventory_df):
    INVENTORY_RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(INVENTORY_RECOMMENDATIONS_FILE, index=False)


def apply_restock(current_date):
    try:
        orders = pd.read_csv(ORDERS_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return

    inventory = pd.read_csv(INVENTORY_FILE)

    current_date = pd.to_datetime(current_date)

    updated_orders = []

    for _, order in orders.iterrows():
        restock_date = pd.to_datetime(order["Restock_Date"])

        if restock_date <= current_date:
            sku = order["SKU"]
            qty = order["Order_Qty"]

            inventory.loc[inventory["SKU"] == sku, "Current_Stock"] += qty

            # 🔥 RESTOCK EVENT
            log_event(
                event_time=current_date,
                event_type="RESTOCK",
                message=f"{sku} restocked with {qty} units"
            )
        else:
            updated_orders.append(order)

    inventory.to_csv(INVENTORY_FILE, index=False)
    pd.DataFrame(updated_orders).to_csv(ORDERS_FILE, index=False)