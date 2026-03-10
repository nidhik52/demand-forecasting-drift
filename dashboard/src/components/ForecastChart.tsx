/**
 * ForecastChart.tsx
 *
 * Shows the 2026 demand forecast for a selected SKU as a smooth area chart.
 * Displays stat cards for average, min, and max forecast demand.
 */

import { useEffect, useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// Recharts 2.x ReferenceLine has a JSX type incompatibility with React 18
const RefLine = ReferenceLine as any;
import { fetchForecast } from "../api";
import type { ForecastPoint } from "../types";

interface Props { sku: string; refreshTick?: number; }

// ── Custom tooltip ────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <div style={{ color: "#94a3b8", marginBottom: 2 }}>{label}</div>
      <div style={{ color: "#4f8ef7", fontWeight: 600 }}>
        {Number(payload[0].value).toFixed(1)} units
      </div>
    </div>
  );
};

export default function ForecastChart({ sku, refreshTick }: Props) {
  const [data, setData]       = useState<ForecastPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    if (!sku) return;
    setLoading(true);
    setError(null);
    fetchForecast(sku)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sku, refreshTick]);

  if (loading) return <div className="loading-msg">Loading forecast…</div>;
  if (error)   return <div className="error-msg">Error: {error}</div>;
  if (!data.length) return (
    <div className="loading-msg">
      No forecast data for {sku}. Run <code>pipeline.py</code> first.
    </div>
  );

  // Stat calculations
  const values = data.map((d) => d.forecast_demand);
  const avg    = values.reduce((a, b) => a + b, 0) / values.length;
  const min    = Math.min(...values);
  const max    = Math.max(...values);

  // Format date labels compactly (MMM DD)
  const chartData = data.map((d) => ({
    date:   new Date(d.Date).toLocaleDateString("en-GB", { month: "short", day: "numeric" }),
    demand: +d.forecast_demand.toFixed(2),
  }));

  return (
    <>
      {/* ── Stats row ───────────────────────────────────────────────────── */}
      <div className="stat-row">
        <div className="stat-card">
          <div className="stat-label">SKU</div>
          <div className="stat-value" style={{ fontSize: "1.2rem" }}>{sku}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Daily Demand</div>
          <div className="stat-value green">{avg.toFixed(1)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Peak Demand</div>
          <div className="stat-value yellow">{max.toFixed(1)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Min Demand</div>
          <div className="stat-value">{min.toFixed(1)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Forecast Days</div>
          <div className="stat-value">{data.length}</div>
        </div>
      </div>

      {/* ── Chart ───────────────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-title">
          2026 Demand Forecast — {sku}
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="demandGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#4f8ef7" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#4f8ef7" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={false}
              interval={29}   /* show roughly monthly labels */
            />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip content={<CustomTooltip />} />
            <RefLine
              y={avg}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              label={{ value: "avg", fill: "#f59e0b", fontSize: 10 }}
            />
            <Area
              type="monotone"
              dataKey="demand"
              stroke="#4f8ef7"
              strokeWidth={2}
              fill="url(#demandGrad)"
              dot={false}
              activeDot={{ r: 4, fill: "#4f8ef7" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}
