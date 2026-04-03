import React, { useState, useEffect, useCallback, useMemo } from "react";
import "./App.css";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Legend
} from "recharts";

const API = (process.env.REACT_APP_API_BASE_URL || "").replace(/\/$/, "");

const getApiUrl = (path) => `${API}${path}`;

const formatDay = (raw) => {
  if (!raw) return "-";
  const value = String(raw);
  if (value.includes("T")) return value.split("T")[0];
  if (value.includes(" ")) return value.split(" ")[0];
  return value;
};

const formatDateLabel = (raw) => {
  if (!raw) return "-";
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return String(raw);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
};

const formatDateTimeLabel = (raw) => {
  if (!raw) return "-";
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return String(raw);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  });
};

const getRiskLevel = (item) => {
  const raw = item?.Risk || item?.Risk_Level || "LOW";
  const normalized = String(raw).toUpperCase();
  if (["CRITICAL", "HIGH"].includes(normalized)) return "CRITICAL";
  if (["WARNING", "MEDIUM"].includes(normalized)) return "WARNING";
  return "LOW";
};

const riskClassMap = {
  CRITICAL: "risk-critical",
  WARNING: "risk-warning",
  LOW: "risk-low"
};

const extractSkuFromMessage = (message = "") => {
  const text = String(message);
  const driftMatch = text.match(/for\s+([A-Z0-9-]+)/i);
  if (driftMatch?.[1]) return driftMatch[1].toUpperCase();

  const retrainMatch = text.match(/^([A-Z0-9-]+)\s+retrained/i);
  if (retrainMatch?.[1]) return retrainMatch[1].toUpperCase();

  return null;
};

const normalizeEvent = (event) => {
  const timestamp = event?.timestamp || event?.date || "";
  const eventType = String(event?.event_type || "UNKNOWN").toUpperCase();
  const message = String(event?.message || "");

  return {
    timestamp,
    event_type: eventType,
    message,
    dateKey: formatDay(timestamp),
    sku: extractSkuFromMessage(message)
  };
};

const fetchJson = async (path) => {
  const response = await fetch(getApiUrl(path));
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${path}`);
  }
  return response.json();
};

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  const maeValue = payload.find((entry) => entry.dataKey === "mae")?.value;
  const driftPayload = payload.find((entry) => entry.payload?.driftMessage)?.payload;

  return (
    <div className="chart-tooltip">
      <p>{formatDateLabel(label)}</p>
      <p>
        MAE: <strong>{Number(maeValue || 0).toFixed(3)}</strong>
      </p>
      {driftPayload?.driftMessage ? <p className="tooltip-drift">Drift: {driftPayload.driftMessage}</p> : null}
    </div>
  );
};

const KPI = ({ title, value }) => (
  <div className="glass-card card-hover kpi-card">
    <p className="kpi-title">{title}</p>
    <p className="kpi-value">{value}</p>
  </div>
);

function App() {
  const [sku, setSku] = useState("APPL-001");
  const [skuOptions, setSkuOptions] = useState([]);
  const [start, setStart] = useState("2025-07-01");
  const [end, setEnd] = useState("2025-07-31");

  const [metrics, setMetrics] = useState([]);
  const [events, setEvents] = useState([]);
  const [inventory, setInventory] = useState([]);
  const [orderQtyBySku, setOrderQtyBySku] = useState({});

  const [isLoading, setIsLoading] = useState(false);
  const [isRunningPipeline, setIsRunningPipeline] = useState(false);
  const [placingOrderSku, setPlacingOrderSku] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [pipelineStatus, setPipelineStatus] = useState({
    last_run_at: null,
    last_event_type: null,
    last_event_message: null,
    latest_metrics_date: null
  });
  const [toasts, setToasts] = useState([]);

  const pushToast = useCallback((type, title, body) => {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;

    setToasts((prev) => [...prev, { id, type, title, body }]);

    window.setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, 4500);
  }, []);

  const fetchInventoryOnly = useCallback(async () => {
    const inventoryData = await fetchJson("/inventory");
    setInventory(Array.isArray(inventoryData) ? inventoryData : []);
  }, []);

  const fetchSkuOptions = useCallback(async () => {
    const response = await fetchJson("/skus");
    const options = Array.isArray(response?.skus) ? response.skus : [];
    setSkuOptions(options);

    if (options.length > 0 && !options.includes(String(sku).toUpperCase())) {
      setSku(options[0]);
    }
  }, [sku]);

  const fetchData = useCallback(async () => {
    setIsLoading(true);
    setErrorMessage("");

    try {
      const encodedSku = encodeURIComponent(sku);
      const [metricsData, eventsData, inventoryData, statusData] = await Promise.all([
        fetchJson(`/metrics?sku=${encodedSku}&start=${start}&end=${end}`),
        fetchJson(`/events?start=${start}&end=${end}`),
        fetchJson("/inventory"),
        fetchJson("/pipeline_status")
      ]);

      setMetrics(Array.isArray(metricsData) ? metricsData : []);
      setEvents(Array.isArray(eventsData) ? eventsData : []);
      setInventory(Array.isArray(inventoryData) ? inventoryData : []);
      setPipelineStatus(statusData || {});
    } catch (error) {
      setMetrics([]);
      setEvents([]);
      setInventory([]);
      setErrorMessage(
        `Data unavailable. Start API on port 8000 or set REACT_APP_API_BASE_URL. (${error.message})`
      );
    } finally {
      setIsLoading(false);
    }
  }, [sku, start, end]);

  useEffect(() => {
    fetchData();
    fetchSkuOptions();
  }, [fetchData, fetchSkuOptions]);

  const runPipeline = async () => {
    setIsRunningPipeline(true);

    try {
      const response = await fetch(getApiUrl(`/run_pipeline?start=${start}&end=${end}`), {
        method: "POST"
      });

      if (!response.ok) {
        throw new Error(`Pipeline request failed (${response.status})`);
      }

      const result = await response.json();
      pushToast("success", "Pipeline completed", "Fresh metrics and events are now available.");

      if (result?.completed_at) {
        setPipelineStatus((prev) => ({
          ...prev,
          last_run_at: result.completed_at,
          last_event_type: "PIPELINE_COMPLETE",
          last_event_message: "Pipeline completed"
        }));
      }

      await fetchData();
    } catch (error) {
      pushToast("error", "Pipeline failed", error.message);
    } finally {
      setIsRunningPipeline(false);
    }
  };

  const placeOrder = async (targetSku, qty) => {
    const parsedQty = Number(qty);
    if (!Number.isFinite(parsedQty) || parsedQty <= 0) {
      pushToast("warning", "Invalid quantity", "Enter a valid quantity greater than 0.");
      return;
    }

    setPlacingOrderSku(targetSku);

    try {
      const response = await fetch(
        getApiUrl(`/order?sku=${encodeURIComponent(targetSku)}&qty=${parsedQty}`),
        { method: "POST" }
      );

      if (!response.ok) {
        throw new Error(`Order request failed (${response.status})`);
      }

      const result = await response.json();
      const restockDate = result?.restock_date || "Not available";

      pushToast(
        "success",
        `Order successful for ${targetSku}`,
        `Qty ${parsedQty} placed. Restock date: ${restockDate}`
      );

      // Only refresh inventory, so the graph and metrics remain stable.
      await fetchInventoryOnly();
    } catch (error) {
      pushToast("error", "Order failed", error.message);
    } finally {
      setPlacingOrderSku("");
    }
  };

  const normalizedEvents = useMemo(
    () => (Array.isArray(events) ? events.map(normalizeEvent) : []),
    [events]
  );

  const skuUpper = useMemo(() => String(sku || "").toUpperCase(), [sku]);

  const skuSpecificEvents = useMemo(
    () => normalizedEvents.filter((event) => event.sku && event.sku === skuUpper),
    [normalizedEvents, skuUpper]
  );

  const selectedEvents = skuSpecificEvents.length > 0 ? skuSpecificEvents : normalizedEvents;

  const totalDrift = selectedEvents.filter((event) => event.event_type === "DRIFT").length;

  const avgMAE =
    metrics.length > 0
      ? (metrics.reduce((sum, row) => sum + Number(row.MAE || 0), 0) / metrics.length).toFixed(2)
      : "0.00";

  const criticalCount = inventory.filter((item) => getRiskLevel(item) === "CRITICAL").length;

  const chartData = useMemo(
    () =>
      metrics
        .map((row) => ({
          date: formatDay(row.Date),
          mae: Number(row.MAE || 0)
        }))
        .filter((row) => Number.isFinite(row.mae))
        .sort((a, b) => new Date(a.date) - new Date(b.date)),
    [metrics]
  );

  const maxMae = useMemo(() => {
    if (!chartData.length) return 2;
    return Math.max(...chartData.map((point) => point.mae), 2);
  }, [chartData]);

  const yDomainMax = useMemo(() => {
    const padded = maxMae * 1.12;
    return Math.max(2, Math.ceil(padded * 4) / 4);
  }, [maxMae]);

  const driftEventsForChart = useMemo(
    () => selectedEvents.filter((event) => event.event_type === "DRIFT"),
    [selectedEvents]
  );

  const driftByDate = useMemo(() => {
    const map = new Map();
    driftEventsForChart.forEach((event) => {
      if (!map.has(event.dateKey)) {
        map.set(event.dateKey, []);
      }
      map.get(event.dateKey).push(event.message);
    });
    return map;
  }, [driftEventsForChart]);

  const chartDataWithDrift = useMemo(
    () =>
      chartData.map((point) => {
        const messages = driftByDate.get(point.date);
        return {
          ...point,
          driftMarker: messages ? point.mae : null,
          driftMessage: messages ? messages.join(" | ") : ""
        };
      }),
    [chartData, driftByDate]
  );

  const retrainEvents = useMemo(
    () => selectedEvents.filter((event) => event.event_type === "RETRAIN").slice(-10).reverse(),
    [selectedEvents]
  );

  const recentDriftEvents = useMemo(
    () => driftEventsForChart.slice(-12).reverse(),
    [driftEventsForChart]
  );

  const renderLoadingOverlay = (message) => (
    <div className="overlay">
      <div className="spinner-group" aria-hidden="true">
        <span className="spinner-circle spinner-circle-1" />
        <span className="spinner-circle spinner-circle-2" />
        <span className="spinner-circle spinner-circle-3" />
      </div>
      <p>{message}</p>
    </div>
  );

  const pipelineRunText = pipelineStatus?.last_run_at
    ? formatDateTimeLabel(pipelineStatus.last_run_at)
    : "No run logged yet";

  return (
    <div className="dashboard-shell">
      <div className="dashboard-bg-glow" />
      <div className="dashboard-bg-grid" />

      <header className="dashboard-header glass-card">
        <div>
          <h1>📊 Drift-Aware Retail Intelligence</h1>
          <p>Live monitoring for forecast quality, inventory risk, and retraining actions.</p>
        </div>
        <div className="status-pill-group">
          <div className="status-pill">
            <span className="status-dot" />
            {isRunningPipeline ? "Pipeline Running" : "System Ready"}
          </div>
        </div>
      </header>

      <section className="glass-card filter-card">
        <h2 className="section-title">🧭 Filters & Actions</h2>
        <div className="filter-grid">
          <label>
            SKU
            <select value={sku} onChange={(event) => setSku(event.target.value)} className="sku-select">
              {skuOptions.length === 0 ? (
                <option value={sku}>{sku}</option>
              ) : (
                skuOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))
              )}
            </select>
          </label>

          <label>
            Start Date
            <input className="date-input" type="date" value={start} onChange={(event) => setStart(event.target.value)} />
          </label>

          <label>
            End Date
            <input className="date-input" type="date" value={end} onChange={(event) => setEnd(event.target.value)} />
          </label>

          <div className="button-row">
            <button className="btn btn-secondary" onClick={fetchData} disabled={isLoading || isRunningPipeline}>
              {isLoading ? "Refreshing..." : "Refresh"}
            </button>
            <button className="btn btn-primary" onClick={runPipeline} disabled={isRunningPipeline}>
              {isRunningPipeline ? "Running..." : "Run Pipeline"}
            </button>
          </div>
        </div>

        <div className="pipeline-meta">
          <p>
            <strong>Last pipeline run:</strong> {pipelineRunText}
          </p>
          <p>
            <strong>Latest event:</strong> {pipelineStatus?.last_event_type || "-"}
            {pipelineStatus?.last_event_message ? ` - ${pipelineStatus.last_event_message}` : ""}
          </p>
          <p>
            <strong>Latest metrics date:</strong> {pipelineStatus?.latest_metrics_date || "-"}
          </p>
        </div>
      </section>

      {errorMessage && (
        <section className="error-banner" role="alert">
          <strong>Connection issue:</strong> {errorMessage}
        </section>
      )}

      {isRunningPipeline && renderLoadingOverlay("Pipeline is running. Please wait while the dashboard updates.")}

      <section className="kpi-grid">
        <KPI title="⚠ Drift Events" value={totalDrift} />
        <KPI title="📉 Avg MAE" value={avgMAE} />
        <KPI title="🚨 Critical SKUs" value={criticalCount} />
      </section>

      <section className="main-grid">
        <div className="glass-card card-hover chart-card">
          <div className="card-heading">
            <h2 className="section-title">📈 Forecast Error Trend</h2>
          </div>
          <div className="chart-area">
            {isLoading ? (
              renderLoadingOverlay("Loading metrics and drift signals...")
            ) : (
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={chartDataWithDrift} margin={{ top: 8, right: 20, left: 8, bottom: 8 }}>
                  <CartesianGrid stroke="rgba(154, 176, 214, 0.16)" strokeDasharray="3 3" />
                  <XAxis dataKey="date" tickFormatter={formatDateLabel} stroke="#9ab0d6" tickMargin={8} />
                  <YAxis
                    stroke="#9ab0d6"
                    domain={[0, yDomainMax]}
                    tickCount={6}
                    allowDecimals
                  />
                  <Tooltip content={<ChartTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="mae"
                    name="MAE"
                    stroke="#5eb9ff"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 4, strokeWidth: 0 }}
                  />
                  <Line
                    type="linear"
                    dataKey="driftMarker"
                    name="Drift"
                    stroke="#f7c948"
                    strokeWidth={0}
                    dot={{ r: 4, fill: "#f7c948", stroke: "#0e213e", strokeWidth: 1 }}
                    activeDot={{ r: 5, fill: "#f7c948", stroke: "#0e213e", strokeWidth: 1 }}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="glass-card card-hover">
          <h2 className="section-title">📦 Inventory Risk</h2>
          <div className="inventory-list">
            {inventory.slice(0, 10).map((item, idx) => {
              const riskLevel = getRiskLevel(item);
              const riskClass = riskClassMap[riskLevel] || "risk-low";

              return (
                <article key={`${item.SKU}-${idx}`} className={`inventory-item ${riskClass}`}>
                  <div className="inventory-row">
                    <h3>{item.SKU}</h3>
                    <span className={`risk-badge ${riskClass}`}>{riskLevel}</span>
                  </div>
                  <p>
                    Stock: <strong>{item.Current_Stock}</strong>
                    <span className="divider-dot" />
                    Suggested: <strong>{item.Recommended_Order_Qty}</strong>
                  </p>
                  <div className="inventory-order-row">
                    <input
                      type="number"
                      min="1"
                      placeholder="Order qty"
                      value={orderQtyBySku[item.SKU] || ""}
                      onChange={(event) =>
                        setOrderQtyBySku((prev) => ({ ...prev, [item.SKU]: event.target.value }))
                      }
                    />
                    <button
                      className="btn btn-order"
                      onClick={() => placeOrder(item.SKU, orderQtyBySku[item.SKU])}
                      disabled={placingOrderSku === item.SKU}
                    >
                      {placingOrderSku === item.SKU ? "Ordering..." : "Order"}
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      </section>

      <section className="main-grid lower-grid">
        <div className="glass-card card-hover">
          <h2 className="section-title">⚠ Drift Events</h2>
          <div className="scroll-panel">
            {recentDriftEvents.length === 0 ? (
              <p className="empty-text">No drift events in the selected range.</p>
            ) : (
              recentDriftEvents.map((event, index) => (
                <div key={`${event.timestamp}-${index}`} className="event-row">
                  <time>{formatDateTimeLabel(event.timestamp)}</time>
                  <p>{event.message}</p>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="glass-card card-hover">
          <h2 className="section-title">🔁 Retrain History</h2>
          <div className="scroll-panel compact">
            {retrainEvents.length === 0 ? (
              <p className="empty-text">No retraining runs in this range.</p>
            ) : (
              retrainEvents.map((event, index) => (
                <div key={`${event.timestamp}-${index}`} className="retrain-row">
                  <time>{formatDateTimeLabel(event.timestamp)}</time>
                  <span>{event.message}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <div className="toast-stack" aria-live="polite" aria-atomic="true">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast-item toast-${toast.type}`}>
            <p className="toast-title">{toast.title}</p>
            <p className="toast-body">{toast.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
