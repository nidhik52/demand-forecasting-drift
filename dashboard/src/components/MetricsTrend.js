import React, { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Scatter
} from "recharts";

export default function MetricsTrend() {

  const [data, setData] = useState([]);
  const [events, setEvents] = useState([]);
  const [sku, setSku] = useState("");
  const [skuList, setSkuList] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {

    fetch("/metrics")
      .then((res) => {
        if (!res.ok) throw new Error(`Metrics API returned ${res.status}`);
        return res.json();
      })
      .then((d) => {
        setData(d);
        const skus = [...new Set(d.map((x) => x.SKU))];
        setSkuList(skus);
        setSku(skus[0]);
        setError("");
      })
      .catch(() => {
        setData([]);
        setSkuList([]);
        setError("Metrics API unreachable. Ensure backend is running on port 8000.");
      });

    fetch("/events")
      .then((res) => {
        if (!res.ok) throw new Error(`Events API returned ${res.status}`);
        return res.json();
      })
      .then(setEvents)
      .catch(() => {
        setEvents([]);
      });

  }, []);

  const filtered = data
    .filter(d => d.SKU === sku)
    .map(d => ({
      Date: d.Date,
      Error: Number(d.Error)
    }));

  const driftPoints = events
    .filter(e => e.event_type === "DRIFT")
    .filter(e => e.message.includes(sku))   // ✅ MATCH SKU
    .map(e => ({
      Date: e.date,
      Error: 2.5   // marker level
    }));

  return (
    <div className="glass-card">
      <h3>📉 Error Trend with Drift</h3>

      {error && <p style={{ color: "#ff6b6b" }}>{error}</p>}

      <select value={sku} onChange={e => setSku(e.target.value)}>
        {skuList.map(s => <option key={s}>{s}</option>)}
      </select>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={filtered}>
          <XAxis dataKey="Date" />
          <YAxis />
          <Tooltip />

          <Line dataKey="Error" stroke="#ff6b6b" dot={false} />

          {/* ✅ DRIFT POINTS */}
          <Scatter data={driftPoints} fill="#f1c40f" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}