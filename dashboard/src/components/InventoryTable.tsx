/**
 * InventoryTable.tsx
 *
 * Displays inventory replenishment recommendations for all SKUs.
 * SKUs needing an order are highlighted and can be filtered with a toggle.
 */

import { useEffect, useState } from "react";
import { fetchInventory } from "../api";
import type { InventoryRow } from "../types";

export default function InventoryTable() {
  const [rows, setRows]           = useState<InventoryRow[]>([]);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState<string | null>(null);
  const [filterAlert, setFilter]  = useState(false);
  const [search, setSearch]       = useState("");

  useEffect(() => {
    setLoading(true);
    fetchInventory()
      .then(setRows)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-msg">Loading inventory…</div>;
  if (error)   return <div className="error-msg">Error: {error}</div>;
  if (!rows.length) {
    return (
      <div className="card">
        <div className="loading-msg">
          No inventory data. Run <code>python pipeline.py</code> first.
        </div>
      </div>
    );
  }

  const alertCount = rows.filter((r) => r.Recommended_Order_Qty > 0).length;

  // Apply filters
  let visible = rows;
  if (filterAlert) visible = visible.filter((r) => r.Recommended_Order_Qty > 0);
  if (search)      visible = visible.filter((r) =>
    r.SKU.toUpperCase().includes(search.toUpperCase())
  );

  return (
    <>
      {/* Stat row */}
      <div className="stat-row">
        <div className="stat-card">
          <div className="stat-label">Total SKUs</div>
          <div className="stat-value">{rows.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Need Replenishment</div>
          <div className="stat-value red">{alertCount}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Stock OK</div>
          <div className="stat-value green">{rows.length - alertCount}</div>
        </div>
      </div>

      {/* Controls */}
      <div className="card" style={{ padding: "0.75rem 1rem" }}>
        <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
          <input
            type="text"
            placeholder="Search SKU…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="sku-select"
            style={{ minWidth: 200 }}
          />
          <label style={{ fontSize: "0.85rem", color: "#94a3b8", display: "flex", gap: "0.4rem", alignItems: "center", cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={filterAlert}
              onChange={(e) => setFilter(e.target.checked)}
            />
            Show alerts only
          </label>
          <span style={{ fontSize: "0.8rem", color: "#64748b", marginLeft: "auto" }}>
            {visible.length} / {rows.length} SKUs
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="card">
        <div className="card-title">Inventory Replenishment Recommendations</div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>SKU</th>
                <th>Current Stock</th>
                <th>Safety Stock</th>
                <th>Reorder Point</th>
                <th>Order Qty</th>
                <th>Order Date</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((r) => {
                const needsOrder = r.Recommended_Order_Qty > 0;
                return (
                  <tr key={r.SKU}>
                    <td><strong>{r.SKU}</strong></td>
                    <td
                      style={{ color: needsOrder ? "#ef4444" : "#10b981" }}
                    >
                      {r.Current_Stock.toFixed(0)}
                    </td>
                    <td style={{ color: "#94a3b8" }}>{r.Safety_Stock.toFixed(0)}</td>
                    <td>{r.Reorder_Point.toFixed(0)}</td>
                    <td style={{ color: needsOrder ? "#f59e0b" : "#64748b", fontWeight: needsOrder ? 700 : 400 }}>
                      {needsOrder ? r.Recommended_Order_Qty.toFixed(0) : "—"}
                    </td>
                    <td style={{ color: "#94a3b8" }}>{r.Recommended_Order_Date}</td>
                    <td>
                      <span className={`badge ${needsOrder ? "badge-red" : "badge-green"}`}>
                        {needsOrder ? "ORDER" : "OK"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
