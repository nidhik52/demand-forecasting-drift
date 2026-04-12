import React, { useEffect, useMemo, useState, useCallback } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Legend, ResponsiveContainer,
  ReferenceLine
} from "recharts";
import "./App.css";

const API =
  process.env.REACT_APP_API_BASE ||
  (process.env.NODE_ENV === "production"
    ? "https://demand-forecasting-drift.onrender.com"
    : "");

function App() {
  const [skus,         setSkus]         = useState([]);
  const [selectedSKU,  setSelectedSKU]  = useState("");
  const [metrics,      setMetrics]      = useState([]);
  const [events,       setEvents]       = useState([]);
  const [driftEvents,  setDriftEvents]  = useState([]);   // FIX: dedicated drift-events
  const [inventory,    setInventory]    = useState([]);
  const [orderQty,     setOrderQty]     = useState({});
  const [loading,      setLoading]      = useState(false);
  const [refreshing,   setRefreshing]   = useState(false);
  const [apiHealthy,   setApiHealthy]   = useState(true);
  const [error,        setError]        = useState("");
  const [monitoring,   setMonitoring]   = useState(null);
  const [lastUpdated,  setLastUpdated]  = useState(null);
  const [eventQuery,   setEventQuery]   = useState("");
  const [eventType,    setEventType]    = useState("ALL");
  const [inventoryQuery, setInventoryQuery] = useState("");
  const [riskFilter,   setRiskFilter]   = useState("ALL");
  const [retrainQuery, setRetrainQuery] = useState("");
  const [modelChoice,  setModelChoice]  = useState("prophet");

  // FIX: default dates match pipeline data range (Jul-Dec 2025)
  const [start, setStart] = useState("2025-07-01");
  const [end,   setEnd]   = useState("2025-12-31");

  const computeRestockDate = (orderDate, leadDays) => {
    if (!orderDate || !leadDays) return "--";
    const date = new Date(orderDate);
    date.setDate(date.getDate() + leadDays);
    return date.toISOString().split("T")[0];
  };

  // Load SKUs once on mount
  useEffect(() => {
    const loadSkus = async () => {
      try {
        setError("");
        const res  = await axios.get(`${API}/skus`);
        const items = Array.isArray(res.data) ? res.data : [];
        setSkus(items);
        if (items.length > 0) setSelectedSKU(items[0].SKU);
        setApiHealthy(true);
      } catch {
        setError("Failed to load SKUs. Is the API running?");
        setApiHealthy(false);
      }
    };
    loadSkus();
  }, []);

  const fetchData = useCallback(async () => {
    if (!selectedSKU) return;
    try {
      setError("");
      setRefreshing(true);
      const [m, e, d, i] = await Promise.all([
        axios.get(`${API}/metrics`,       { params: { sku: selectedSKU, start, end } }),
        axios.get(`${API}/events`,        { params: { sku: selectedSKU, start, end } }),
        axios.get(`${API}/drift-events`,  { params: { sku: selectedSKU, start, end } }),  // FIX
        axios.get(`${API}/inventory`,     { params: { end } }),
      ]);
      setMetrics(    Array.isArray(m.data) ? m.data : []);
      setEvents(     Array.isArray(e.data) ? e.data : []);
      setDriftEvents(Array.isArray(d.data) ? d.data : []);
      setInventory(  Array.isArray(i.data) ? i.data : []);
      setApiHealthy(true);
      setLastUpdated(new Date());
    } catch {
      setError("Failed to load data.");
      setApiHealthy(false);
    } finally {
      setRefreshing(false);
    }
  }, [selectedSKU, start, end]);

  useEffect(() => { fetchData(); }, [fetchData]);

  // Monitoring poll every 30s
  useEffect(() => {
    const fetchMonitoring = async () => {
      try {
        const res = await axios.get(`${API}/monitoring`);
        setMonitoring(res.data || null);
      } catch { setMonitoring(null); }
    };
    fetchMonitoring();
    const timer = setInterval(fetchMonitoring, 30000);
    return () => clearInterval(timer);
  }, []);

  const runPipeline = async () => {
    try {
      setError("");
      setLoading(true);
      await axios.post(`${API}/run_pipeline`, null, {
        params: { start, end, model: modelChoice }
      });
      // Refresh after 3s to give pipeline time to write first results
      setTimeout(fetchData, 3000);
    } catch {
      setError("Failed to run pipeline.");
    } finally {
      setLoading(false);
    }
  };

  const placeOrder = async (sku, qty) => {
    try {
      setError("");
      const res = await axios.post(`${API}/order`, null, { params: { sku, qty, end } });
      alert(`✅ Order placed!\nRestock by: ${res.data.restock_date}`);
      fetchData();
    } catch {
      setError("Failed to place order.");
    }
  };

  // ── Chart data ───────────────────────────────────────────────────────────
  const chartData = useMemo(() =>
    metrics
      .filter(d => d && d.Date && d.Actual != null && d.Predicted != null)
      .map(d => ({
        date:      String(d.Date).split("T")[0],
        actual:    Number(d.Actual),
        predicted: Number(d.Predicted),
        mae:       Number(d.MAE || 0),
        drift:     Number(d.Drift || 0),
      })),
    [metrics]
  );

  // FIX: drift reference lines from dedicated /drift-events endpoint
  // Each unique drift date gets a vertical red reference line
  const driftDates = useMemo(() => {
    const dates = new Set(driftEvents.map(e => e.date));
    return [...dates];
  }, [driftEvents]);

  // ── Derived lists ────────────────────────────────────────────────────────
  const retrainEvents = useMemo(() =>
    events.filter(e => String(e.event_type || "").toUpperCase().includes("RETRAIN")),
    [events]
  );
  const driftEventsFiltered = useMemo(() =>
    events.filter(e => String(e.event_type || "").toUpperCase().includes("DRIFT")),
    [events]
  );

  const formatDate = value => {
    if (!value) return "--";
    const parsed = new Date(value);
    return isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
  };

  const kpiCards = [
    { title: "SKUs Monitored",       value: skus.length || "--" },
    { title: "Drift Events",         value: monitoring?.drift_count ?? driftEventsFiltered.length },
    { title: "Average MAE",          value: monitoring?.avg_mae != null ? Number(monitoring.avg_mae).toFixed(2) : "--" },
    { title: "Retrain Events",       value: retrainEvents.length },
  ];

  const riskClass = level => {
    const norm = String(level || "SAFE").toUpperCase();
    if (norm.includes("CRIT")) return "risk-badge risk-critical";
    if (norm.includes("WARN")) return "risk-badge risk-warning";
    return "risk-badge risk-low";
  };

  const filteredEvents = useMemo(() => {
    const query = eventQuery.trim().toLowerCase();
    return events.filter(event => {
      const typeVal    = String(event.event_type || "").toUpperCase();
      const matchType  = eventType === "ALL" || typeVal === eventType;
      const matchQuery = !query ||
        String(event.message   || "").toLowerCase().includes(query) ||
        String(event.timestamp || "").toLowerCase().includes(query);
      return matchType && matchQuery;
    });
  }, [events, eventQuery, eventType]);

  const filteredInventory = useMemo(() => {
    const query = inventoryQuery.trim().toLowerCase();
    return inventory.filter(item => {
      const risk       = String(item.Risk_Level || "SAFE").toUpperCase();
      const matchRisk  = riskFilter === "ALL" || risk === riskFilter;
      const matchQuery = !query ||
        String(item.SKU     || "").toLowerCase().includes(query) ||
        String(item.Product || "").toLowerCase().includes(query);
      return matchRisk && matchQuery;
    });
  }, [inventory, inventoryQuery, riskFilter]);

  const filteredRetrain = useMemo(() => {
    const query = retrainQuery.trim().toLowerCase();
    return retrainEvents.filter(e =>
      !query ||
      String(e.message   || "").toLowerCase().includes(query) ||
      String(e.timestamp || "").toLowerCase().includes(query)
    );
  }, [retrainEvents, retrainQuery]);

  // Custom tooltip for forecast chart
  const ForecastTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    return (
      <div className="chart-tooltip">
        <p style={{ marginBottom: 4 }}>{label}</p>
        {payload.map(item => (
          <p key={item.name} style={{ color: item.color, margin: "2px 0" }}>
            {item.name}: {typeof item.value === "number" ? item.value.toFixed(2) : item.value}
          </p>
        ))}
        {payload[0]?.payload?.drift === 1 && (
          <p style={{ color: "#ff7b7b", margin: "4px 0 0" }}>⚠ Drift detected</p>
        )}
      </div>
    );
  };

  return (
    <div className="dashboard-shell">
      <div className="dashboard-bg-glow" />
      <div className="dashboard-bg-grid" />

      {/* Header */}
      <header className="dashboard-header glass-card card-hover">
        <div>
          <h1>Drift-Aware Forecast Dashboard</h1>
          <p>Real-time drift signals, inventory posture, and pipeline health.</p>
        </div>
        <div className="status-pill-group">
          <span className="status-pill">
            <span className="status-dot"
              style={{ background: apiHealthy ? "#4cd7a0" : "#ffd166" }} />
            {apiHealthy ? "API Connected" : "API Degraded"}
          </span>
          <span className="status-pill">
            Updated {lastUpdated ? formatDate(lastUpdated) : "--"}
          </span>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {/* Filters */}
      <section className="filter-card glass-card">
        <p className="section-title">Filters</p>
        <div className="filter-grid">
          <label>
            SKU
            <select value={selectedSKU} onChange={e => setSelectedSKU(e.target.value)}>
              {skus.map(s => (
                <option key={s.SKU} value={s.SKU}>{s.SKU}</option>
              ))}
            </select>
          </label>
          <label>
            Model
            <select value={modelChoice} onChange={e => setModelChoice(e.target.value)}>
              <option value="prophet">Prophet</option>
              <option value="mean">Baseline (Mean)</option>
            </select>
          </label>
          <label>
            Start date
            <input className="date-input" type="date" value={start}
              onChange={e => setStart(e.target.value)} />
          </label>
          <label>
            End date
            <input className="date-input" type="date" value={end}
              onChange={e => setEnd(e.target.value)} />
          </label>
          <div className="button-row">
            <button className="btn btn-primary" onClick={runPipeline} disabled={loading}>
              {loading ? "Running…" : "Run Pipeline"}
            </button>
            <button className="btn btn-secondary" onClick={fetchData} disabled={refreshing}>
              {refreshing ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </div>
        <div className="pipeline-meta">
          <p>Total records: {monitoring?.total_records ?? "--"}</p>
          <p>Drift events: {monitoring?.drift_count ?? "--"}</p>
        </div>
      </section>

      {/* KPI Cards */}
      <section className="kpi-grid">
        {kpiCards.map(card => (
          <div key={card.title} className="kpi-card glass-card card-hover">
            <p className="kpi-title">{card.title}</p>
            <p className="kpi-value">{card.value}</p>
          </div>
        ))}
      </section>

      {/* Main charts */}
      <section className="main-grid">

        {/* Forecast vs Actual */}
        <div className="chart-card glass-card">
          <p className="section-title">Forecast vs Actual — {selectedSKU}</p>
          <div className="chart-area">
            {refreshing ? (
              <div className="overlay">
                <div className="spinner-group">
                  <span className="spinner-circle" />
                  <span className="spinner-circle spinner-circle-2" />
                  <span className="spinner-circle spinner-circle-3" />
                </div>
                <p>Refreshing series…</p>
              </div>
            ) : chartData.length === 0 ? (
              <div className="overlay">
                <p>No data for this range. Run the pipeline first.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#1c2b4a" strokeDasharray="3 3" />
                  <XAxis dataKey="date" stroke="#c2d9ff" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#c2d9ff" tick={{ fontSize: 11 }} />
                  <Tooltip content={<ForecastTooltip />} />
                  <Legend />

                  <Line dataKey="actual"    name="Actual"    stroke="#60a5fa"
                    strokeWidth={2} dot={false} />
                  <Line dataKey="predicted" name="Predicted" stroke="#34d399"
                    strokeWidth={2} dot={false} strokeDasharray="5 4" />

                  {/* FIX: drift reference lines from /drift-events (vertical red lines) */}
                  {driftDates.map(date => (
                    <ReferenceLine
                      key={date}
                      x={date}
                      stroke="#ff7b7b"
                      strokeWidth={1.5}
                      strokeDasharray="4 3"
                      label={{ value: "⚠", position: "top", fill: "#ff7b7b", fontSize: 11 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Events panel */}
        <div className="chart-card glass-card">
          <div className="panel-header">
            <p className="section-title">Drift &amp; Events</p>
            <div className="panel-filters">
              <select value={eventType} onChange={e => setEventType(e.target.value)}>
                <option value="ALL">All types</option>
                <option value="DRIFT">Drift</option>
                <option value="RETRAIN">Retrain</option>
                <option value="COOLDOWN">Cooldown</option>
                <option value="ORDER">Order</option>
              </select>
              <input type="text" value={eventQuery}
                onChange={e => setEventQuery(e.target.value)}
                placeholder="Search events" />
            </div>
          </div>
          <div className="scroll-panel">
            {filteredEvents.length === 0
              ? <p className="empty-text">No events for this range.</p>
              : filteredEvents.map((event, idx) => (
                <div className="event-row" key={`${event.timestamp}-${idx}`}>
                  <time>{formatDate(event.timestamp)}</time>
                  <p>
                    <strong>{event.event_type || "EVENT"}</strong> — {event.message || ""}
                  </p>
                </div>
              ))}
          </div>
        </div>
      </section>

      {/* MAE trend + Inventory */}
      <section className="main-grid lower-grid">

        {/* MAE over time */}
        <div className="chart-card glass-card">
          <p className="section-title">MAE Trend — {selectedSKU}</p>
          <div className="chart-area">
            {chartData.length === 0
              ? <div className="overlay"><p>No MAE data yet.</p></div>
              : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke="#1c2b4a" strokeDasharray="3 3" />
                    <XAxis dataKey="date" stroke="#c2d9ff" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#c2d9ff" tick={{ fontSize: 11 }} />
                    <Tooltip formatter={v => Number(v).toFixed(3)} />
                    <Legend />
                    <Line dataKey="mae" name="MAE" stroke="#ffd166"
                      strokeWidth={1.8} dot={false} />
                    {driftDates.map(date => (
                      <ReferenceLine key={date} x={date}
                        stroke="#ff7b7b" strokeWidth={1} strokeDasharray="4 3" />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              )}
          </div>
        </div>

        {/* Inventory risk */}
        <div className="chart-card glass-card">
          <div className="panel-header">
            <p className="section-title">Inventory Risk</p>
            <div className="panel-filters">
              <select value={riskFilter} onChange={e => setRiskFilter(e.target.value)}>
                <option value="ALL">All risk</option>
                <option value="CRITICAL">Critical</option>
                <option value="WARNING">Warning</option>
                <option value="SAFE">Safe</option>
              </select>
              <input type="text" value={inventoryQuery}
                onChange={e => setInventoryQuery(e.target.value)}
                placeholder="Search SKU" />
            </div>
          </div>
          <div className="inventory-list">
            {filteredInventory.length === 0
              ? <p className="empty-text">No inventory data yet.</p>
              : filteredInventory.map(item => (
                <div key={item.SKU} className="inventory-item">
                  <div className="inventory-row">
                    <div>
                      <h3>{item.SKU}</h3>
                      <p>{item.Product || ""}</p>
                    </div>
                    <span className={riskClass(item.Risk_Level)}>
                      {item.Risk_Level || "SAFE"}
                    </span>
                  </div>
                  <p>
                    Stock: {item.Current_Stock ?? "--"} |
                    Rec: {item.Recommended_Order_Qty ?? "--"} |
                    Lead: {item.Lead_Time_Days ?? "--"}d
                  </p>
                  <div className="inventory-order-row">
                    <input type="number" min="0"
                      value={orderQty[item.SKU] || ""}
                      onChange={e => setOrderQty(prev => ({ ...prev, [item.SKU]: e.target.value }))}
                      placeholder="Order qty" />
                    <button className="btn btn-order"
                      onClick={() => placeOrder(item.SKU, orderQty[item.SKU] || 0)}>
                      Order
                    </button>
                  </div>
                  {orderQty[item.SKU] > 0 && item.Lead_Time_Days && (
                    <div className="restock-note">
                      ⏱ Restock approx: {computeRestockDate(end, item.Lead_Time_Days)}
                    </div>
                  )}
                </div>
              ))}
          </div>
        </div>
      </section>

      {/* Retrain timeline */}
      <section className="main-grid" style={{ marginTop: "1.2rem" }}>
        <div className="chart-card glass-card">
          <div className="panel-header">
            <p className="section-title">Retrain Timeline</p>
            <input type="text" value={retrainQuery}
              onChange={e => setRetrainQuery(e.target.value)}
              placeholder="Search retrain" />
          </div>
          <div className="scroll-panel compact">
            {filteredRetrain.length === 0
              ? <p className="empty-text">No retrain events yet.</p>
              : filteredRetrain.map((event, idx) => (
                <div className="retrain-row" key={`${event.timestamp}-${idx}`}>
                  <time>{formatDate(event.timestamp)}</time>
                  <span>{event.message || event.event_type || "RETRAIN"}</span>
                </div>
              ))}
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;