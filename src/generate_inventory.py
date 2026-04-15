
import pandas as pd
import numpy as np
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR

np.random.seed(42)


def generate_inventory(output_path=None):

    df = pd.read_csv(PROCESSED_DIR / "daily_demand.csv")
    forecast = pd.read_csv(PROCESSED_DIR / "forecast_2025.csv")
    forecast_skus = set(forecast["SKU"].unique())
    products = df[["SKU", "SKU_Name"]].drop_duplicates()
    products = products[products["SKU"].isin(forecast_skus)]
    inventory = []
    # Ensure a mix: 1/3 CRITICAL (zero), 1/3 WARNING (low), 1/3 SAFE (high)
    n = len(products)
    n_critical = max(1, n // 3)
    n_warning = max(1, n // 3)
    n_safe = n - n_critical - n_warning
    profiles = (["zero"] * n_critical) + (["low"] * n_warning) + (["high"] * n_safe)
    np.random.shuffle(profiles)
    for (_, row), profile in zip(products.iterrows(), profiles):
        sku = row["SKU"]
        avg_demand = df[df["SKU"] == sku]["Demand"].mean()
        if profile == "zero":
            stock = 0
        elif profile == "low":
            days = np.random.randint(1, 6)
            stock = int(max(0, avg_demand * days))
        else:  # high
            days = np.random.randint(30, 61)
            stock = int(max(0, avg_demand * days))
        inventory.append({
            "SKU": sku,
            "Product": row["SKU_Name"],
            "Current_Stock": stock,
            "Lead_Time_Days": np.random.choice([5, 7, 10]),
            "Stock_As_Of_Date": "2025-07-01",
        })

    inventory_df = pd.DataFrame(inventory)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = PROCESSED_DIR / "inventory_master.csv"

    inventory_df.to_csv(output_path, index=False)

    print(f"Inventory generated -> {output_path}")

    return inventory_df


if __name__ == "__main__":
    generate_inventory()