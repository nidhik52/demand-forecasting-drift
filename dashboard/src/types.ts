// Shared TypeScript interfaces — imported by components and api.ts

export interface ForecastPoint {
  Date: string;
  SKU: string;
  forecast_demand: number;
}

export interface DriftEvent {
  timestamp: string;
  sku: string;
  date: string;
  rolling_mae: number;
  threshold: number;
  drift_detected: boolean;
}

export interface InventoryRow {
  SKU: string;
  Current_Stock: number;
  Safety_Stock: number;
  Reorder_Point: number;
  Recommended_Order_Qty: number;
  Recommended_Order_Date: string;
}
