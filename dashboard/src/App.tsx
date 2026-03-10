import { useState, useEffect } from "react";
import { fetchSkus } from "./api";
import ForecastChart       from "./components/ForecastChart";
import DriftMonitor        from "./components/DriftMonitor";
import InventoryTable      from "./components/InventoryTable";
import DriftAlertPanel     from "./components/DriftAlertPanel";
import InventoryAlertPanel from "./components/InventoryAlertPanel";
import SystemEvents        from "./components/SystemEvents";

type Tab = "overview" | "forecast" | "drift" | "inventory";

const REFRESH_MS     = 5000;   // auto-refresh interval
const MAX_STREAM_DAY = 90;     // matches STREAM_DAYS in streaming.py

export default function App() {
  const [activeTab,   setActiveTab]   = useState<Tab>("overview");
  const [skus,        setSkus]        = useState<string[]>([]);
  const [selectedSku, setSelectedSku] = useState<string>("");
  const [refreshTick, setRefreshTick] = useState(0);
  const [streamDay,   setStreamDay]   = useState(1);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  // Load SKU list once on mount
  useEffect(() => {
    fetchSkus()
      .then((list) => { setSkus(list); if (list.length) setSelectedSku(list[0]); })
      .catch(() => {});
  }, []);

  // Auto-refresh every 5 seconds — bump tick so all components re-fetch
  useEffect(() => {
    const id = setInterval(() => {
      setRefreshTick((t) => t + 1);
      setStreamDay((d) => (d < MAX_STREAM_DAY ? d + 1 : d));
      setLastRefresh(new Date());
    }, REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  const progress = Math.round((streamDay / MAX_STREAM_DAY) * 100);

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview",  label: "🖥️ Overview"       },
    { id: "forecast",  label: "📈 Forecast"        },
    { id: "drift",     label: "⚠️  Drift Monitor"  },
    { id: "inventory", label: "📦 Inventory"       },
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

        {/* ── Right-side status widgets ──────────────────────────────────── */}
        <div className="topbar-right">
          {/* Streaming progress */}
          <div className="stream-badge">
            <span className="stream-label">Streaming Progress</span>
            <span className="stream-val">Day {streamDay} / {MAX_STREAM_DAY}</span>
            <div className="stream-progress-bar">
              <div className="stream-progress-fill" style={{ width: `${progress}%` }} />
            </div>
          </div>

          {/* Live pulse */}
          <div className="live-pill">
            <span className="live-dot" />
            LIVE
          </div>

          <span className="refresh-time">
            {lastRefresh.toLocaleTimeString()}
          </span>
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

        {/* ── Overview: all four panels visible at once ───────────────── */}
        {activeTab === "overview" && (
          <>
            {/* Full-width drift alert panel */}
            <DriftAlertPanel refreshTick={refreshTick} />

            {/* Two-column lower section */}
            <div className="two-col">
              <InventoryAlertPanel refreshTick={refreshTick} />
              <SystemEvents        refreshTick={refreshTick} />
            </div>
          </>
        )}

        {/* ── SKU selector (forecast + drift tabs) ────────────────────── */}
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
              {skus.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
        )}

        {activeTab === "forecast"  && <ForecastChart  sku={selectedSku} refreshTick={refreshTick} />}
        {activeTab === "drift"     && <DriftMonitor   sku={selectedSku} refreshTick={refreshTick} />}
        {activeTab === "inventory" && <InventoryTable refreshTick={refreshTick} />}
      </main>
    </div>
  );
}
