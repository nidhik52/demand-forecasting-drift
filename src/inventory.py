"""
inventory.py
────────────
Inventory Decision Engine.

Reads the 2026 demand forecast and computes, for every SKU:

  Average daily demand  (μ)
  Demand std deviation  (σ)
  Safety Stock          = 1.65 × σ × √(lead_time)
  Reorder Point         = μ × lead_time + safety_stock
  Recommended Order Qty = reorder_point − current_stock  (if below)

The 1.65 z-score corresponds to a 95 % service level — a common
industry standard for fast-moving retail goods.

Current stock is simulated with a reproducible random draw so the
pipeline can be demonstrated without a live ERP system.

Output
──────
  data/processed/inventory_recommendations.csv
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
FORECAST_PATH = Path("data/processed/forecast_2026.csv")
OUTPUT_PATH   = Path("data/processed/inventory_recommendations.csv")

# ── Parameters ────────────────────────────────────────────────────────────────
LEAD_TIME = 7       # days from order placement to goods arrival
Z_SCORE   = 1.65    # 95 % service level
RNG_SEED  = 42      # reproducible random stock simulation


def compute_inventory(
    forecast_df: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute replenishment recommendations from the 2026 demand forecast.

    Parameters
    ──────────
    forecast_df : pre-loaded forecast DataFrame (Date | SKU | forecast_demand).
                  If None, the file at FORECAST_PATH is read automatically.
    save        : write the recommendations CSV when True.

    Returns
    ───────
    DataFrame with columns:
      SKU | Current_Stock | Safety_Stock | Reorder_Point |
      Recommended_Order_Qty | Recommended_Order_Date
    """
    if forecast_df is None:
        forecast_df = pd.read_csv(FORECAST_PATH, parse_dates=["Date"])

    print("[Inventory] Computing replenishment recommendations …")

    rng  = np.random.default_rng(RNG_SEED)
    rows = []

    for sku, grp in forecast_df.groupby("SKU"):
        demand = grp["forecast_demand"].values

        avg_demand = demand.mean()
        std_demand = demand.std(ddof=1) if len(demand) > 1 else 0.0

        # Safety Stock buffers against demand variability during lead time
        safety_stock  = Z_SCORE * std_demand * np.sqrt(LEAD_TIME)

        # Reorder Point: expected demand consumed while waiting for delivery
        reorder_point = avg_demand * LEAD_TIME + safety_stock

        # Simulate current inventory level (0 … 2× reorder point)
        current_stock = float(rng.uniform(0, 2 * reorder_point))

        if current_stock < reorder_point:
            order_qty        = reorder_point - current_stock
            recommended_date = (date.today() + timedelta(days=1)).isoformat()
        else:
            order_qty        = 0.0
            recommended_date = "—"

        rows.append({
            "SKU":                    sku,
            "Current_Stock":          round(current_stock, 1),
            "Safety_Stock":           round(safety_stock, 1),
            "Reorder_Point":          round(reorder_point, 1),
            "Recommended_Order_Qty":  round(order_qty, 1),
            "Recommended_Order_Date": recommended_date,
        })

    recommendations = pd.DataFrame(rows)
    needs_order     = (recommendations["Recommended_Order_Qty"] > 0).sum()
    print(f"  {needs_order} / {len(recommendations)} SKUs flagged for replenishment.")

    if save:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(OUTPUT_PATH, index=False)
        print(f"  Saved → {OUTPUT_PATH}")

    return recommendations


if __name__ == "__main__":
    compute_inventory()
