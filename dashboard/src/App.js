import React, { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  CartesianGrid, Scatter, Legend
} from "recharts";

const API =
  process.env.NODE_ENV === "production"
    ? "https://demand-forecasting-drift.onrender.com"
    : "http://localhost:8000";

function App() {

  const [skus, setSkus] = useState([]);
  const [selectedSKU, setSelectedSKU] = useState("");
  const [metrics, setMetrics] = useState([]);
  const [events, setEvents] = useState([]);
  const [inventory, setInventory] = useState([]);
  const [orderQty, setOrderQty] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

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
      } catch (err) {
        setError("Failed to load SKUs. Check API URL and CORS.");
      }
    };

    loadSkus();
  }, []);

  const fetchData = async () => {
    if (!selectedSKU) return;

    try {
      setError("");
      const [m, e, i] = await Promise.all([
        axios.get(`${API}/metrics`, { params: { sku: selectedSKU, start, end } }),
        axios.get(`${API}/events`, { params: { sku: selectedSKU, start, end } }),
        axios.get(`${API}/inventory`, { params: { end } })
      ]);

      setMetrics(Array.isArray(m.data) ? m.data : []);
      setEvents(Array.isArray(e.data) ? e.data : []);
      setInventory(Array.isArray(i.data) ? i.data : []);
    } catch (err) {
      setError("Failed to load data. Check API URL and server logs.");
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedSKU, start, end]);

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

  return (
    <div style={{ padding: 20, background: "#0b1220", color: "white" }}>

      <h2>📊 Drift-Aware Dashboard</h2>

      {error && <div style={{ color: "#fca5a5" }}>{error}</div>}

      <select value={selectedSKU} onChange={e => setSelectedSKU(e.target.value)}>
        {skus.map(s => (
          <option key={s.SKU} value={s.SKU}>
            {s.SKU} - {s.Product}
          </option>
        ))}
      </select>

      <input type="date" value={start} onChange={e => setStart(e.target.value)} />
      <input type="date" value={end} onChange={e => setEnd(e.target.value)} />

      <button onClick={runPipeline}>Run Pipeline</button>
      {loading && <span> ⏳ Running...</span>}

      <LineChart width={800} height={300} data={chartData}>
        <CartesianGrid stroke="#444" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />

        <Line dataKey="actual" stroke="#60a5fa" />
        <Line dataKey="predicted" stroke="#34d399" />

        <Scatter data={driftPoints} dataKey="actual" fill="red" name="Drift" />
      </LineChart>

      <h3>Events</h3>
      <ul>
        {events.map((e, i) => (
          <li key={i}>
            {e.timestamp} - {e.event_type} - {e.message}
          </li>
        ))}
      </ul>

      <h3>Inventory</h3>
      {inventory.map((item, i) => (
        <div key={i}>
          {item.SKU} | Stock: {item.Current_Stock} | Rec: {item.Recommended_Order_Qty}

          <input
            type="number"
            onChange={e =>
              setOrderQty(prev => ({
                ...prev,
                [item.SKU]: e.target.value
              }))
            }
          />

          <button onClick={() => placeOrder(item.SKU, orderQty[item.SKU] || 0)}>
            Order
          </button>
        </div>
      ))}
    </div>
  );
}

export default App;