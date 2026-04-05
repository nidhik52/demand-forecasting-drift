import React, { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Scatter, Legend, ResponsiveContainer
} from "recharts";
import "./App.css";

const API =
  process.env.REACT_APP_API_BASE ||
  (process.env.NODE_ENV === "production"
    ? "https://demand-forecasting-drift.onrender.com"
    : "http://localhost:8000");

function App() {

  const [skus, setSkus] = useState([]);
  const [selectedSKU, setSelectedSKU] = useState("");
  const [metrics, setMetrics] = useState([]);
  const [events, setEvents] = useState([]);
  const [inventory, setInventory] = useState([]);
  const [orderQty, setOrderQty] = useState({});
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [apiHealthy, setApiHealthy] = useState(true);
  const [error, setError] = useState("");
  const [monitoring, setMonitoring] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);

  const [start, setStart] = useState("2025-07-01");
  const [end, setEnd] = useState("2025-07-31");

  // LOAD SKUs
  useEffect(() => {
    const loadSkus = async () => {
      try {
        setError("");
        const res = await axios.get(`${API}/skus`);
        const items = Array.isArray(res.data) ? res.data : [];
        setSkus(items);
        if (items.length > 0) {
          setSelectedSKU(items[0].SKU);
        }
        setApiHealthy(true);
      } catch (err) {
        setError("Failed to load SKUs. Check API URL and CORS.");
        setApiHealthy(false);
      }
    };

    loadSkus();
  }, []);

  const fetchData = async () => {
    if (!selectedSKU) return;

    try {
      setError("");
      setRefreshing(true);
      const [m, e, i] = await Promise.all([
        axios.get(`${API}/metrics`, { params: { sku: selectedSKU, start, end } }),
        axios.get(`${API}/events`, { params: { sku: selectedSKU, start, end } }),
        axios.get(`${API}/inventory`, { params: { end } })
      ]);

      setMetrics(Array.isArray(m.data) ? m.data : []);
      setEvents(Array.isArray(e.data) ? e.data : []);
      setInventory(Array.isArray(i.data) ? i.data : []);
      setApiHealthy(true);
      setLastUpdated(new Date());
    } catch (err) {
      setError("Failed to load data. Check API URL and server logs.");
      setApiHealthy(false);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedSKU, start, end]);

  useEffect(() => {
    const fetchMonitoring = async () => {
      try {
        const res = await axios.get(`${API}/monitoring`);
        setMonitoring(res.data || null);
      } catch (err) {
        setMonitoring(null);
      }
    };

    fetchMonitoring();
    const timer = setInterval(fetchMonitoring, 30000);
    return () => clearInterval(timer);
  }, []);

  const runPipeline = async () => {
    try {
      setError("");
      setLoading(true);
      await axios.post(`${API}/run_pipeline`, null, { params: { start, end } });
      setTimeout(fetchData, 2000);
    } catch (err) {
      setError("Failed to run pipeline. Check API URL and server logs.");
    } finally {
      setLoading(false);
    }
  };

  const placeOrder = async (sku, qty) => {
    try {
      setError("");
      const res = await axios.post(`${API}/order`, null, {
        params: { sku, qty }
      });

      alert(`✅ Order successful!\nRestock by: ${res.data.restock_date}`);
      fetchData();
    } catch (err) {
      setError("Failed to place order. Check API URL and server logs.");
    }
  };

  const chartData = metrics
    .filter(d => d && d.Date && d.Actual != null && d.Predicted != null)
    .map(d => ({
      date: String(d.Date).split("T")[0],
      actual: d.Actual,
      predicted: d.Predicted,
      error: Math.abs(d.Actual - d.Predicted)
    }));

  const driftPoints = events
    .filter(e => e.event_type?.toUpperCase() === "DRIFT")
    .map(e => {
      const date = e.timestamp.split(" ")[0];
      const match = chartData.find(d => d.date === date);
      return match ? { ...match } : null;
    })
    .filter(Boolean);

  const retrainEvents = useMemo(
    () => events.filter(e => String(e.event_type || "").toUpperCase().includes("RETRAIN")),
    [events]
  );

  const driftEvents = useMemo(
    () => events.filter(e => String(e.event_type || "").toUpperCase().includes("DRIFT")),
    [events]
  );

  const lastRun = monitoring?.last_run?.[0];

  const formatDate = value => {
    if (!value) return "--";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return parsed.toLocaleString();
  };

  const kpiCards = [
    {
      title: "Total SKUs Monitored",
      value: skus.length || monitoring?.total_records || "--"
    },
    {
      title: "Drift Events Detected",
      value: monitoring?.drift_count ?? driftEvents.length
    },
    {
      title: "Average MAE",
      value: monitoring?.avg_mae ? monitoring.avg_mae.toFixed(2) : "--"
    }
  ];

  const riskClass = level => {
    const normalized = String(level || "SAFE").toUpperCase();
    if (normalized.includes("CRIT")) return "risk-badge risk-critical";
    if (normalized.includes("WARN")) return "risk-badge risk-warning";
    return "risk-badge risk-low";
  };

  return (
    <div className="dashboard-shell">
      <div className="dashboard-bg-glow" />
      <div className="dashboard-bg-grid" />

      <header className="dashboard-header glass-card card-hover">
        <div>
          <h1>Drift-Aware Dashboard</h1>
          <p>Real-time drift signals, inventory posture, and pipeline health.</p>
        </div>
        <div className="status-pill-group">
          <span className="status-pill">
            <span
              className="status-dot"
              style={{ background: apiHealthy ? "#4cd7a0" : "#ffd166" }}
            />
            {apiHealthy ? "API Connected" : "API Degraded"}
          </span>
          <span className="status-pill">
            Updated {lastUpdated ? formatDate(lastUpdated) : "--"}
          </span>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <section className="filter-card glass-card">
        <p className="section-title">Filters</p>
        <div className="filter-grid">
          <label>
            SKU
            <select value={selectedSKU} onChange={e => setSelectedSKU(e.target.value)}>
              {skus.map(s => (
                <option key={s.SKU} value={s.SKU}>
                  {s.SKU} - {s.Product}
                </option>
              ))}
            </select>
          </label>
          <label>
            Start date
            <input
              className="date-input"
              type="date"
              value={start}
              onChange={e => setStart(e.target.value)}
            />
          </label>
          <label>
            End date
            <input
              className="date-input"
              type="date"
              value={end}
              onChange={e => setEnd(e.target.value)}
            />
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
          <p>Pipeline runs: {monitoring?.pipeline_runs ?? "--"}</p>
          <p>Last run: {lastRun ? formatDate(lastRun.timestamp || lastRun.time) : "--"}</p>
        </div>
      </section>

      <section className="kpi-grid">
        {kpiCards.map(card => (
          <div key={card.title} className="kpi-card glass-card card-hover">
            <p className="kpi-title">{card.title}</p>
            <p className="kpi-value">{card.value}</p>
          </div>
        ))}
      </section>

      <section className="main-grid">
        <div className="chart-card glass-card">
          <p className="section-title">Forecast vs Actual</p>
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
                <p>No forecast data for this range.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#1c2b4a" strokeDasharray="3 3" />
                  <XAxis dataKey="date" stroke="#c2d9ff" tick={{ fontSize: 12 }} />
                  <YAxis stroke="#c2d9ff" tick={{ fontSize: 12 }} />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload?.length) return null;
                      return (
                        <div className="chart-tooltip">
                          <p>{label}</p>
                          {payload.map(item => (
                            <p key={item.name} style={{ color: item.color }}>
                              {item.name}: {item.value}
                            </p>
                          ))}
                        </div>
                      );
                    }}
                  />
                  <Legend />
                  <Line dataKey="actual" stroke="#60a5fa" strokeWidth={2.3} dot={false} />
                  <Line dataKey="predicted" stroke="#34d399" strokeWidth={2.3} dot={false} />
                  <Scatter data={driftPoints} dataKey="actual" fill="#ff7b7b" name="Drift" />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="chart-card glass-card">
          <p className="section-title">Drift & Events</p>
          <div className="scroll-panel">
            {events.length === 0 ? (
              <p className="empty-text">No events logged for this range.</p>
            ) : (
              events.map((event, idx) => (
                <div className="event-row" key={`${event.timestamp}-${idx}`}>
                  <time>{formatDate(event.timestamp)}</time>
                  <p>{event.event_type || "EVENT"} • {event.message || ""}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="main-grid lower-grid">
        <div className="chart-card glass-card">
          <p className="section-title">Inventory Risk</p>
          <div className="inventory-list">
            {inventory.length === 0 ? (
              <p className="empty-text">No inventory data yet.</p>
            ) : (
              inventory.map(item => (
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
                    Stock: {item.Current_Stock ?? "--"}
                    <span className="divider-dot" />
                    Rec: {item.Recommended_Order_Qty ?? "--"}
                  </p>
                  <div className="inventory-order-row">
                    <input
                      type="number"
                      min="0"
                      value={orderQty[item.SKU] || ""}
                      onChange={e =>
                        setOrderQty(prev => ({
                          ...prev,
                          [item.SKU]: e.target.value
                        }))
                      }
                      placeholder="Order qty"
                    />
                    <button
                      className="btn btn-order"
                      onClick={() => placeOrder(item.SKU, orderQty[item.SKU] || 0)}
                    >
                      Order
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="chart-card glass-card">
          <p className="section-title">Retrain Timeline</p>
          <div className="scroll-panel compact">
            {retrainEvents.length === 0 ? (
              <p className="empty-text">No retrain events yet.</p>
            ) : (
              retrainEvents.map((event, idx) => (
                <div className="retrain-row" key={`${event.timestamp}-${idx}`}>
                  <time>{formatDate(event.timestamp)}</time>
                  <span>{event.message || event.event_type || "RETRAIN"}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;