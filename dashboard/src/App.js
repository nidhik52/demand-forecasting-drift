import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, ResponsiveContainer, Scatter
} from "recharts";

const API = (process.env.REACT_APP_API || "").replace(/\/$/, "");

function App() {

  const [sku, setSku] = useState("");
  const [skus, setSkus] = useState([]);

  const [start, setStart] = useState("2025-07-01");
  const [end, setEnd] = useState("2025-07-31");

  const [metrics, setMetrics] = useState([]);
  const [events, setEvents] = useState([]);
  const [inventory, setInventory] = useState([]);

  const [loading, setLoading] = useState(false);

  // --------------------------
  // LOAD SKUs
  // --------------------------
  useEffect(() => {
    fetch(`${API}/skus`)
      .then(res => res.json())
      .then(data => {
        setSkus(data);
        if (data.length > 0) setSku(data[0]);
      });
  }, []);

  // --------------------------
  // FETCH DATA
  // --------------------------
  const fetchData = useCallback(async () => {

    if (!sku) return;

    try {
      const [mRes, eRes, iRes] = await Promise.all([
        fetch(`${API}/metrics?sku=${sku}&start=${start}&end=${end}`),
        fetch(`${API}/events?sku=${sku}&start=${start}&end=${end}`),
        fetch(`${API}/inventory`)
      ]);

      setMetrics(await mRes.json());
      setEvents(await eRes.json());
      setInventory(await iRes.json());

    } catch (err) {
      console.error("Fetch error:", err);
    }

  }, [sku, start, end]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // --------------------------
  // RUN PIPELINE
  // --------------------------
  const runPipeline = async () => {
    setLoading(true);

    try {
      await fetch(`${API}/run_pipeline?start=${start}&end=${end}`, {
        method: "POST"
      });

      // wait a bit for backend to update files
      setTimeout(fetchData, 2000);

    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  // --------------------------
  // PLACE ORDER
  // --------------------------
  const placeOrder = async (sku, qty) => {
    if (qty === 0) return;

    await fetch(`${API}/order?sku=${sku}&qty=${qty}`, {
      method: "POST"
    });

    alert(`Order placed for ${sku}`);
    fetchData();
  };

  // --------------------------
  // KPI
  // --------------------------
  const totalDrift = events.filter(e => e.event_type === "DRIFT").length;

  const avgMAE =
    metrics.length > 0
      ? (metrics.reduce((s, x) => s + Number(x.MAE), 0) / metrics.length).toFixed(2)
      : 0;

  const criticalCount = inventory.filter(i => i.Risk_Level === "CRITICAL").length;

  // --------------------------
  // CHART
  // --------------------------
  const chartData = metrics.map(d => ({
    date: d.Date,
    mae: Number(d.MAE)
  }));

  const driftPoints = events
    .filter(e => e.event_type === "DRIFT")
    .map(e => ({
      date: e.timestamp.split(" ")[0],
      mae: Math.max(...chartData.map(d => d.mae), 2)
    }));

  return (
    <div style={container}>

      <h1>📊 Drift-Aware Retail Intelligence</h1>

      {/* FILTER */}
      <div style={filterBar}>
        <select value={sku} onChange={e => setSku(e.target.value)}>
          {skus.map(s => <option key={s}>{s}</option>)}
        </select>

        <input type="date" value={start} onChange={e => setStart(e.target.value)} />
        <input type="date" value={end} onChange={e => setEnd(e.target.value)} />

        <button onClick={fetchData}>Refresh</button>

        <button onClick={runPipeline} disabled={loading}>
          {loading ? "Running..." : "Run Pipeline"}
        </button>
      </div>

      {/* LOADING */}
      {loading && <p>⏳ Pipeline running... please wait</p>}

      {/* KPI */}
      <div style={grid3}>
        <Card>⚠ Drift Events: {totalDrift}</Card>
        <Card>📉 Avg MAE: {avgMAE}</Card>
        <Card>🚨 Critical SKUs: {criticalCount}</Card>
      </div>

      {/* MAIN */}
      <div style={gridMain}>

        {/* CHART */}
        <Card>
          <h3>Error Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid stroke="#2a3f5f" />
              <XAxis dataKey="date" stroke="#ccc" />
              <YAxis stroke="#ccc" />
              <Tooltip />
              <Line dataKey="mae" stroke="#ff6b6b" dot={false} />
              <Scatter data={driftPoints} fill="#facc15" />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* INVENTORY */}
        <Card>
          <h3>Inventory</h3>
          {inventory.slice(0, 10).map((i, idx) => (
            <div key={idx} style={{
              marginBottom: "10px",
              color:
                i.Risk_Level === "CRITICAL" ? "#ff6b6b" :
                i.Risk_Level === "WARNING" ? "#facc15" :
                "#4ade80"
            }}>
              <b>{i.SKU}</b> ({i.Risk_Level})<br />
              Stock: {i.Current_Stock} | Suggested: {i.Recommended_Order_Qty}

              <br />

              <button
                onClick={() => placeOrder(i.SKU, i.Recommended_Order_Qty)}
                disabled={i.Recommended_Order_Qty === 0}
              >
                Order
              </button>
            </div>
          ))}
        </Card>

      </div>

      {/* EVENTS */}
      <div style={grid2}>
        <Card>
          <h3>Drift Events</h3>
          {events.filter(e => e.event_type === "DRIFT").slice(-10).map((e, i) => (
            <div key={i}>{e.timestamp} — {e.message}</div>
          ))}
        </Card>

        <Card>
          <h3>Retraining</h3>
          {events.filter(e => e.event_type === "RETRAIN").slice(-10).map((e, i) => (
            <div key={i}>{e.timestamp} — {e.message}</div>
          ))}
        </Card>
      </div>

    </div>
  );
}

// --------------------------
const Card = ({ children }) => (
  <div style={{
    background: "#1e293b",
    padding: "15px",
    borderRadius: "12px",
    boxShadow: "0 4px 10px rgba(0,0,0,0.3)"
  }}>
    {children}
  </div>
);

// --------------------------
const container = {
  padding: "20px",
  background: "#0f172a",
  color: "white",
  minHeight: "100vh"
};

const filterBar = {
  marginBottom: "20px",
  display: "flex",
  gap: "10px"
};

const grid3 = {
  display: "grid",
  gridTemplateColumns: "repeat(3,1fr)",
  gap: "20px",
  marginBottom: "20px"
};

const gridMain = {
  display: "grid",
  gridTemplateColumns: "2fr 1fr",
  gap: "20px"
};

const grid2 = {
  display: "grid",
  gridTemplateColumns: "1fr 1fr",
  gap: "20px",
  marginTop: "20px"
};

export default App;