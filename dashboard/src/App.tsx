import { useState, useEffect } from "react";
import { fetchSkus } from "./api";
import ForecastChart  from "./components/ForecastChart";
import DriftMonitor   from "./components/DriftMonitor";
import InventoryTable from "./components/InventoryTable";

type Tab = "forecast" | "drift" | "inventory";

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("forecast");
  const [skus, setSkus]           = useState<string[]>([]);
  const [selectedSku, setSelectedSku] = useState<string>("");

  // Load the SKU list once on mount
  useEffect(() => {
    fetchSkus()
      .then((list) => {
        setSkus(list);
        if (list.length) setSelectedSku(list[0]);
      })
      .catch(() => {/* API not ready yet — pipeline may not have run */});
  }, []);

  const tabs: { id: Tab; label: string }[] = [
    { id: "forecast",  label: "📈 Forecast 2026"   },
    { id: "drift",     label: "⚠️  Drift Monitor"   },
    { id: "inventory", label: "📦 Inventory Alerts" },
  ];

  return (
    <div className="app">
      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <header className="topbar">
        <div>
          <div className="topbar-title">Demand Forecasting Dashboard</div>
          <div className="topbar-sub">
            Drift-Aware Continuous Learning Framework — M.Tech Project
          </div>
        </div>
      </header>

      {/* ── Tab navigation ──────────────────────────────────────────────── */}
      <nav className="tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`tab-btn${activeTab === t.id ? " active" : ""}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* ── Main content ────────────────────────────────────────────────── */}
      <main className="main">
        {/* SKU selector — shown on forecast and drift tabs */}
        {(activeTab === "forecast" || activeTab === "drift") && (
          <div className="sku-select-row">
            <label htmlFor="sku-select">Select SKU:</label>
            <select
              id="sku-select"
              className="sku-select"
              value={selectedSku}
              onChange={(e) => setSelectedSku(e.target.value)}
            >
              {skus.length === 0 && (
                <option value="">— run pipeline.py first —</option>
              )}
              {skus.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        )}

        {activeTab === "forecast"  && <ForecastChart  sku={selectedSku} />}
        {activeTab === "drift"     && <DriftMonitor   sku={selectedSku} />}
        {activeTab === "inventory" && <InventoryTable />}
      </main>
    </div>
  );
}
