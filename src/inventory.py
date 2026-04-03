import pandas as pd
from src.event_logger import log_event
from src.config import (
    FORECAST_FILE,
    INVENTORY_FILE,
    INVENTORY_RECOMMENDATIONS_FILE,
    ORDERS_FILE,
)


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    forecast = pd.read_csv(FORECAST_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)

    forecast["Date"] = pd.to_datetime(forecast["Date"])
    inventory["Stock_As_Of_Date"] = pd.to_datetime(inventory["Stock_As_Of_Date"])

    return forecast, inventory


# -------------------------------
# INVENTORY RECOMMENDATION
# -------------------------------
def generate_inventory_recommendations(forecast, inventory, current_date):

    current_date = pd.to_datetime(current_date)
    recs = []

    for _, item in inventory.iterrows():

        sku = item["SKU"]

        # Safe parsing
        try:
            stock = float(item.get("Current_Stock", 0))
            if pd.isna(stock):
                stock = 0.0
        except:
            stock = 0.0

        stock = max(0.0, stock)

        lead_time = int(item.get("Lead_Time_Days", 7))

        # Forecast window
        horizon_end = current_date + pd.Timedelta(days=lead_time)

        sku_forecast = forecast[
            (forecast["SKU"] == sku)
            & (forecast["Date"] > current_date)
            & (forecast["Date"] <= horizon_end)
        ]

        demand = sku_forecast["Forecast_Demand"].clip(lower=0).sum()

        # -----------------------
        # CORE LOGIC
        # -----------------------
        safety_stock = 0.2 * demand
        reorder_point = demand + safety_stock

        recommended_qty = int(max(0, reorder_point - stock))

        # Risk logic
        if stock <= 0:
            risk = "CRITICAL"
            recommendation = "URGENT: reorder immediately"
        elif stock < demand:
            risk = "CRITICAL"
            recommendation = "Stock will run out before lead time"
        elif stock < reorder_point:
            risk = "WARNING"
            recommendation = "Plan reorder soon"
        else:
            risk = "SAFE"
            recommendation = "Stock sufficient"

        recs.append({
            "SKU": sku,
            "Current_Stock": int(stock),
            "Stock_As_Of_Date": current_date.strftime("%Y-%m-%d %H:%M:%S"),
            "Recommended_Order_Qty": recommended_qty,
            "Risk_Level": risk,
            "Recommendation": recommendation
        })

        print(
            f"[INVENTORY] {sku} | Stock={stock:.1f} | Demand={demand:.1f} | "
            f"Reorder={recommended_qty} | Risk={risk}"
        )

    return pd.DataFrame(recs)


# -------------------------------
# SAVE INVENTORY
# -------------------------------
def save_inventory(df):
    INVENTORY_RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INVENTORY_RECOMMENDATIONS_FILE, index=False)


# -------------------------------
# APPLY RESTOCK
# -------------------------------
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

            inventory.loc[inventory["SKU"] == sku, "Stock_As_Of_Date"] = current_date

            # ✅ CLEAN LOG FORMAT
            log_event(
                event_type="RESTOCK",
                message=f"{sku} restocked {qty} units",
                event_time=current_date
            )

        else:
            updated_orders.append(order)

    inventory.to_csv(INVENTORY_FILE, index=False)
    pd.DataFrame(updated_orders).to_csv(ORDERS_FILE, index=False)