import React, { useEffect, useState } from "react";
import Papa from "papaparse";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";

const dataPath = (fileName) => `${process.env.PUBLIC_URL || ""}/data/${fileName}`;

export default function ForecastChart() {
  const [allData, setAllData] = useState([]);
  const [skuMap, setSkuMap] = useState({});
  const [sku, setSku] = useState("");
  const [startDate, setStartDate] = useState("2025-07-01");
  const [endDate, setEndDate] = useState("2025-07-31");

  useEffect(() => {
    Papa.parse(dataPath("merged.csv"), {
      download: true,
      header: true,
      complete: (result) => {
        const cleaned = result.data.filter(d => d.SKU && d.Date);
        setAllData(cleaned);
      }
    });

    Papa.parse(dataPath("sku_map.csv"), {
      download: true,
      header: true,
      complete: (result) => {
        const map = {};
        result.data.forEach(d => {
          if (d.SKU) map[d.SKU] = d.SKU_Name;
        });
        setSkuMap(map);
        setSku(Object.keys(map)[0]);
      }
    });

  }, []);

  const filtered = allData
    .filter(d => d.SKU === sku)
    .map(d => ({
      ...d,
      Date: new Date(d.Date),
      Demand: Number(d.Demand),
      Forecast_Demand: Number(d.Forecast_Demand)
    }))
    .filter(d => d.Date >= new Date(startDate) && d.Date <= new Date(endDate))
    .sort((a, b) => a.Date - b.Date);

  return (
    <div className="glass-card">
      <h3>📊 Forecast vs Actual</h3>

      <div style={{ display: "flex", gap: "10px", marginBottom: "10px" }}>
        <select value={sku} onChange={(e) => setSku(e.target.value)}>
          {Object.keys(skuMap).map(key => (
            <option key={key} value={key}>
              {key} — {skuMap[key]}
            </option>
          ))}
        </select>

        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={filtered}>
          <XAxis dataKey="Date" tickFormatter={(d) => new Date(d).toLocaleDateString()} />
          <YAxis />
          <Tooltip />
          <Line dataKey="Demand" stroke="#64ffda" dot={false} />
          <Line dataKey="Forecast_Demand" stroke="#8892b0" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}