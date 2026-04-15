
import pandas as pd
import numpy as np
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR

np.random.seed(42)

# Pipeline date range start — inventory is "as of" this date
INVENTORY_AS_OF_DATE = "2025-07-01"
SAFETY_STOCK_DEFAULT = 10


def generate_inventory(output_path=None):

    df = pd.read_csv(PROCESSED_DIR / "daily_demand.csv")
    forecast = pd.read_csv(PROCESSED_DIR / "forecast_2025.csv")
    forecast_skus = set(forecast["SKU"].unique())
    products = df[["SKU", "SKU_Name"]].drop_duplicates()
    products = products[products["SKU"].isin(forecast_skus)]
    inventory = []

    # Ensure a realistic mix:
    #   insufficient  → very low / zero stock  (maps to CRITICAL risk)
    #   borderline    → 2–6 days supply        (maps to WARNING risk)
    #   sufficient    → 15–40 days supply      (maps to SAFE risk)
    n = len(products)
    n_insufficient = max(1, n // 3)
    n_borderline   = max(1, n // 3)
    n_sufficient   = n - n_insufficient - n_borderline

    profiles = (
        ["insufficient"] * n_insufficient
        + ["borderline"]  * n_borderline
        + ["sufficient"]  * n_sufficient
    )
    np.random.shuffle(profiles)

    for (_, row), profile in zip(products.iterrows(), profiles):
        sku = row["SKU"]
        avg_demand = df[df["SKU"] == sku]["Demand"].mean()


        if profile == "insufficient":
            # 0 to ~2 days of supply — clearly below safety stock
            stock = np.random.randint(0, max(1, int(avg_demand * 2)))
            stock_status = "insufficient"
        elif profile == "borderline":
            # 2–6 days of supply — close to reorder point
            days = np.random.randint(2, 7)
            stock = max(0, avg_demand * days)
            stock_status = "borderline"
        else:  # sufficient
            # 15–40 days of supply — comfortably above safety stock
            days = np.random.randint(15, 41)
            stock = max(0, avg_demand * days)
            stock_status = "sufficient"

        # Ensure stock is a valid integer, not bytes or numpy type
        try:
            stock = int(float(stock))
        except Exception:
            stock = 0

        inventory.append({
            "SKU":              sku,
            "Current_Stock":    stock,
            "In_Transit":       0,
            "Lead_Time_Days":   int(np.random.choice([5, 7, 10])),
            "Safety_Stock":     SAFETY_STOCK_DEFAULT,
            "Stock_Status":     stock_status,
            "Stock_As_Of_Date": INVENTORY_AS_OF_DATE,
        })

    inventory_df = pd.DataFrame(inventory)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = PROCESSED_DIR / "inventory_master.csv"

    inventory_df.to_csv(output_path, index=False)

    print(f"Inventory generated -> {output_path}")
    print(inventory_df["Stock_Status"].value_counts().to_string())

    return inventory_df


if __name__ == "__main__":
    generate_inventory()