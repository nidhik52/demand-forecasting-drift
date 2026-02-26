"""
Inventory Replenishment Module
==============================
Project : Drift-Aware Continuous Learning Framework
File    : src/inventory/replenishment.py

HOW IT WORKS (plain language)
------------------------------
After Prophet forecasts the next 30 days of demand, this module
converts those forecasts into three actionable inventory decisions:

  1. Safety Stock  — how many extra units to hold as a buffer
                     against uncertainty in the forecast

  2. Reorder Point — when stock drops to this level, place an order
                     (ensures stock arrives before you run out)

  3. Order Qty     — how many units to order per replenishment cycle

These formulas use Prophet's uncertainty bands (yhat_upper - yhat_lower)
as the measure of forecast uncertainty. This is why XGBoost could not
replace Prophet — it has no uncertainty bands.

WHY THIS MATTERS FOR THE PROJECT
---------------------------------
Without drift detection and retraining:
  - Model forecasts Rs.10K/day for Electronics
  - Inventory module computes reorder point based on Rs.10K
  - Actual demand is Rs.24K/day (drift!)
  - Stock runs out → stockout during Black Friday

With drift detection and retraining:
  - Drift detected on Nov 6
  - Model retrained, now forecasts Rs.24K/day
  - Inventory module recomputes → correct reorder point
  - No stockout

This is the business value of the entire pipeline.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────
# Data class — inventory decision for one category
# ─────────────────────────────────────────────────────────

@dataclass
class InventoryDecision:
    category:          str
    forecast_date:     str           # date decision was made
    avg_daily_demand:  float         # mean forecast demand (Rs./day)
    forecast_std:      float         # uncertainty std from Prophet
    safety_stock:      float         # buffer stock (Rs.)
    reorder_point:     float         # order when stock hits this (Rs.)
    order_quantity:    float         # how much to order (Rs.)
    service_level:     float         # 0.95 = 95% chance of no stockout
    lead_time_days:    int           # days until order arrives
    model_version:     int           # which model produced this forecast

    def to_dict(self):
        return {
            'category'        : self.category,
            'forecast_date'   : self.forecast_date,
            'avg_daily_demand': round(self.avg_daily_demand, 0),
            'forecast_std'    : round(self.forecast_std, 0),
            'safety_stock'    : round(self.safety_stock, 0),
            'reorder_point'   : round(self.reorder_point, 0),
            'order_quantity'  : round(self.order_quantity, 0),
            'service_level'   : self.service_level,
            'lead_time_days'  : self.lead_time_days,
            'model_version'   : self.model_version,
        }

    def summary(self) -> str:
        return (
            f"{self.category} [{self.forecast_date}]\n"
            f"  Avg daily demand : Rs.{self.avg_daily_demand:>10,.0f}\n"
            f"  Forecast std     : Rs.{self.forecast_std:>10,.0f}\n"
            f"  Safety stock     : Rs.{self.safety_stock:>10,.0f}\n"
            f"  Reorder point    : Rs.{self.reorder_point:>10,.0f}\n"
            f"  Order quantity   : Rs.{self.order_quantity:>10,.0f}\n"
            f"  Lead time        : {self.lead_time_days} days\n"
            f"  Service level    : {self.service_level*100:.0f}%\n"
            f"  Model version    : v{self.model_version}"
        )


# ─────────────────────────────────────────────────────────
# InventoryCalculator — core class
# ─────────────────────────────────────────────────────────

class InventoryCalculator:
    """
    Computes safety stock, reorder point, and order quantity
    from a Prophet forecast.

    Parameters
    ----------
    service_level  : float — target service level (default 0.95 = 95%)
                     0.95 → Z = 1.65 standard deviations
                     0.99 → Z = 2.33 (higher stock, less stockout risk)
    lead_time_days : int   — days from placing order to receiving it
    cycle_days     : int   — replenishment cycle length (default 30)

    Formulas
    --------
    forecast_std  = (yhat_upper - yhat_lower) / 4  # Prophet CI to std
                    (dividing by 4 converts 95% CI to ~1 std dev)

    safety_stock  = Z x forecast_std x sqrt(lead_time_days)

    reorder_point = avg_daily_demand x lead_time_days + safety_stock

    order_qty     = avg_daily_demand x cycle_days
    """

    # Z-score lookup for common service levels
    Z_SCORES = {
        0.90: 1.28,
        0.95: 1.65,
        0.98: 2.05,
        0.99: 2.33,
    }

    def __init__(
        self,
        service_level:  float = 0.95,
        lead_time_days: int   = 7,
        cycle_days:     int   = 30,
    ):
        self.service_level  = service_level
        self.lead_time_days = lead_time_days
        self.cycle_days     = cycle_days
        self.Z = self.Z_SCORES.get(service_level, 1.65)

    def compute(
        self,
        category:      str,
        forecast_df:   pd.DataFrame,   # Prophet output: ds, yhat, yhat_lower, yhat_upper
        forecast_date: str  = '',
        model_version: int  = 0,
    ) -> InventoryDecision:
        """
        Compute inventory decisions from a Prophet forecast dataframe.

        Parameters
        ----------
        category      : product category name
        forecast_df   : Prophet forecast output with yhat, yhat_lower, yhat_upper
        forecast_date : date string for logging
        model_version : which model version produced this forecast

        Returns
        -------
        InventoryDecision dataclass
        """
        # Clip negative predictions to 0 (demand can't be negative)
        yhat       = np.maximum(forecast_df['yhat'].to_numpy(dtype=float), 0)
        yhat_lower = np.maximum(forecast_df['yhat_lower'].to_numpy(dtype=float), 0)
        yhat_upper = np.maximum(forecast_df['yhat_upper'].to_numpy(dtype=float), 0)

        # Average daily demand over forecast horizon
        avg_daily_demand = float(np.mean(yhat))

        # Forecast uncertainty — convert Prophet CI to std
        # Prophet's interval_width=0.95 means [yhat_lower, yhat_upper] is a 95% CI
        # A 95% CI = mean ± 1.96*std, so width = 2*1.96*std = 3.92*std
        # We use 4 as a round approximation
        ci_widths    = yhat_upper - yhat_lower
        avg_ci_width = float(np.mean(ci_widths))
        forecast_std = avg_ci_width / 4.0

        # Safety stock
        # Buffer = Z * std * sqrt(lead_time)
        # sqrt(lead_time) accounts for uncertainty accumulating over lead time
        safety_stock = self.Z * forecast_std * np.sqrt(self.lead_time_days)

        # Reorder point
        # Expected demand during lead time + safety buffer
        reorder_point = avg_daily_demand * self.lead_time_days + safety_stock

        # Order quantity
        # Simple: cover one full replenishment cycle
        order_quantity = avg_daily_demand * self.cycle_days

        return InventoryDecision(
            category         = category,
            forecast_date    = forecast_date or str(pd.Timestamp.now().date()),
            avg_daily_demand = avg_daily_demand,
            forecast_std     = forecast_std,
            safety_stock     = safety_stock,
            reorder_point    = reorder_point,
            order_quantity   = order_quantity,
            service_level    = self.service_level,
            lead_time_days   = self.lead_time_days,
            model_version    = model_version,
        )

    def compute_stockout_impact(
        self,
        pre_drift_decision:  'InventoryDecision',
        post_drift_decision: 'InventoryDecision',
        drift_days:          int = 61,
    ) -> dict:
        """
        Compare inventory decisions before and after drift adaptation.
        Quantifies the business impact of NOT having drift detection.

        Parameters
        ----------
        pre_drift_decision  : decision from original model (wrong forecast)
        post_drift_decision : decision from retrained model (correct forecast)
        drift_days          : number of days the drift was active

        Returns
        -------
        dict with stockout_risk_days, excess_demand, revenue_at_risk
        """
        pre_demand  = pre_drift_decision.avg_daily_demand
        post_demand = post_drift_decision.avg_daily_demand
        daily_gap   = max(post_demand - pre_demand, 0)

        # Days at risk = days where old reorder point would trigger too late
        # Simplified: if old ROP < new demand * lead_time, stockout occurs
        old_rop     = pre_drift_decision.reorder_point
        new_rop     = post_drift_decision.reorder_point
        rop_gap     = max(new_rop - old_rop, 0)

        return {
            'pre_avg_daily_demand' : round(pre_demand, 0),
            'post_avg_daily_demand': round(post_demand, 0),
            'daily_demand_gap'     : round(daily_gap, 0),
            'pre_reorder_point'    : round(old_rop, 0),
            'post_reorder_point'   : round(new_rop, 0),
            'reorder_point_gap'    : round(rop_gap, 0),
            'total_unmet_demand'   : round(daily_gap * drift_days, 0),
            'stockout_revenue_risk': round(daily_gap * drift_days, 0),
            'interpretation': (
                f"Without drift detection, the model forecasted Rs.{pre_demand:,.0f}/day "
                f"while actual demand was Rs.{post_demand:,.0f}/day. "
                f"Over {drift_days} drift days, this represents Rs.{daily_gap*drift_days:,.0f} "
                f"in unmet demand due to understocking."
            )
        }
