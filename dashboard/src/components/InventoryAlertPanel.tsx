/**
 * InventoryAlertPanel.tsx
 *
 * Compact overview panel showing SKUs needing orders and those approaching
 * the reorder point.  Color-coded: red = order required, yellow = approaching,
 * green = safe.  Used on the Overview tab.
 */

import { useEffect, useState } from "react";
import { fetchInventory } from "../api";
import type { InventoryRow } from "../types";

interface Props { refreshTick: number; }

type Status = "red" | "yellow" | "green";

function getStatus(row: InventoryRow): Status {
  if (row.Recommended_Order_Qty > 0) return "red";
  if (row.Current_Stock <= row.Reorder_Point * 1.3) return "yellow";
  return "green";
}

export default function InventoryAlertPanel({ refreshTick }: Props) {
  const [rows, setRows]   = useState<InventoryRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchInventory()
      .then((data) => { setRows(data); setError(null); })
      .catch(() => setError("Could not reach API — run the backend first."));
  }, [refreshTick]);

  const redRows    = rows.filter((r) => getStatus(r) === "red");
  const yellowRows = rows.filter((r) => getStatus(r) === "yellow");
  const greenCount = rows.length - redRows.length - yellowRows.length;

  return (
    <div className="card">
      <div className="card-title">📦 Inventory Alerts</div>

      {/* Summary badges */}
      {rows.length > 0 && (
        <div className="inv-summary-row">
          <span className="inv-summary-badge inv-badge-red">
            🔴 {redRows.length} order required
          </span>
          <span className="inv-summary-badge inv-badge-yellow">
            🟡 {yellowRows.length} approaching
          </span>
          <span className="inv-summary-badge inv-badge-green">
            🟢 {greenCount} safe
          </span>
        </div>
      )}

      {error && <div className="no-alerts">{error}</div>}
      {!error && rows.length === 0 && (
        <div className="no-alerts">No inventory data. Run the pipeline first.</div>
      )}

      {/* Red — order required */}
      {redRows.length > 0 && (
        <>
          <div className="inv-section-label red-text">🔴 Order Required</div>
          {redRows.slice(0, 6).map((r) => (
            <div key={r.SKU} className="inv-alert-item inv-red">
              <span className="inv-sku">{r.SKU}</span>
              <div className="inv-details">
                <span>Stock&nbsp;{r.Current_Stock.toFixed(0)}</span>
                <span>Reorder&nbsp;@&nbsp;{r.Reorder_Point.toFixed(0)}</span>
                <span className="inv-order-qty">Order&nbsp;{r.Recommended_Order_Qty.toFixed(0)}&nbsp;units</span>
              </div>
            </div>
          ))}
          {redRows.length > 6 && (
            <div className="no-alerts" style={{ padding: "0.3rem 0" }}>
              +{redRows.length - 6} more — go to Inventory tab
            </div>
          )}
        </>
      )}

      {/* Yellow — approaching reorder */}
      {yellowRows.length > 0 && (
        <>
          <div className="inv-section-label yellow-text" style={{ marginTop: redRows.length ? "0.75rem" : 0 }}>
            🟡 Approaching Reorder Point
          </div>
          {yellowRows.slice(0, 4).map((r) => (
            <div key={r.SKU} className="inv-alert-item inv-yellow">
              <span className="inv-sku">{r.SKU}</span>
              <div className="inv-details">
                <span>Stock&nbsp;{r.Current_Stock.toFixed(0)}</span>
                <span>Reorder&nbsp;@&nbsp;{r.Reorder_Point.toFixed(0)}</span>
              </div>
            </div>
          ))}
        </>
      )}

      {/* All safe */}
      {rows.length > 0 && redRows.length === 0 && yellowRows.length === 0 && (
        <div className="no-alerts green-text">✅ All {rows.length} SKUs are fully stocked</div>
      )}
    </div>
  );
}
