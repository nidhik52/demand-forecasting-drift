import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Scatter
} from "recharts";

const API = "";

const DRIFT_THRESHOLD = 2;

function App() {

  const [sku, setSku] = useState("APPL-001");
  const [start, setStart] = useState("2025-07-01");
  const [end, setEnd] = useState("2025-07-31");

  const [metrics, setMetrics] = useState([]);
  const [events, setEvents] = useState([]);
  const [inventory, setInventory] = useState([]);

  // ------------------
  // FETCH DATA
  // ------------------
  const fetchData = useCallback(async () => {
    const metricRes = await fetch(
      `${API}/metrics?sku=${encodeURIComponent(sku)}&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`
    );
    const eventRes = await fetch(
      `${API}/events?sku=${encodeURIComponent(sku)}&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`
    );
    const inventoryRes = await fetch(`${API}/inventory`);

    const m = await metricRes.json();
    const e = await eventRes.json();
    const i = await inventoryRes.json();

    const sorted = m.sort((a, b) => new Date(a.Date) - new Date(b.Date));

    setMetrics(sorted);
    setEvents(e);
    setInventory(i);
  }, [sku, start, end]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // ------------------
  // RUN PIPELINE
  // ------------------
  const runPipeline = async () => {
    await fetch(
      `${API}/run_pipeline?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`,
      { method: "POST" }
    );

    fetchData();
  };

  // ------------------
  // CHART DATA
  // ------------------
  const chartData = metrics.map(d => ({
    date: d.Date,
    error: d.Error,
    actual: d.Actual,
    predicted: d.Predicted,
    drift: d.Error > DRIFT_THRESHOLD ? d.Error : null
  }));

  return (
    <div style={{ padding: "20px", background: "#0f172a", color: "white" }}>

      <h2>📊 Retail Intelligence Dashboard</h2>

      {/* FILTER */}
      <div>
        SKU:
        <input value={sku} onChange={e => setSku(e.target.value)} />

        Start:
        <input type="date" value={start} onChange={e => setStart(e.target.value)} />

        End:
        <input type="date" value={end} onChange={e => setEnd(e.target.value)} />

        <button onClick={fetchData}>🔄 Refresh</button>
        <button onClick={runPipeline}>🚀 Run Pipeline</button>
      </div>

      {/* ERROR GRAPH */}
      <h3>Error Trend with Drift</h3>

      <LineChart width={900} height={300} data={chartData}>
        <CartesianGrid stroke="#2a3f5f" />
        <XAxis dataKey="date" stroke="#ccc" />
        <YAxis stroke="#ccc" />
        <Tooltip />

        <Line type="monotone" dataKey="error" stroke="#ff6b6b" />
        <Line type="monotone" dataKey="actual" stroke="#4ade80" />
        <Line type="monotone" dataKey="predicted" stroke="#60a5fa" />

        <Scatter dataKey="drift" fill="red" />
      </LineChart>

      {/* INVENTORY */}
      <h3>📦 Inventory</h3>
      {inventory.map((item, i) => (
        <div key={i}>
          {item.SKU} — {item.Risk_Level}  
          Stock: {item.Current_Stock}  
          Suggested: {item.Recommended_Order_Qty}
        </div>
      ))}

      {/* EVENTS */}
      <h3>⚠ Drift Events</h3>
      {events.map((e, i) => (
        <div key={i}>
          {e.timestamp} — {e.message}
        </div>
      ))}

    </div>
  );
}

export default App;