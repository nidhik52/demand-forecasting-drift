/**
 * DriftMonitor.tsx
 *
 * Visualises the rolling MAE over time for a selected SKU and lists all
 * drift events in a table.  A red reference line marks the drift threshold.
 */

import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Legend,
} from "recharts";

// Recharts 2.x ReferenceLine has a JSX type incompatibility with React 18
const RefLine = ReferenceLine as any;
import { fetchDrift } from "../api";
import type { DriftEvent } from "../types";

interface Props { sku: string; refreshTick?: number; }

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  const mae = payload.find((p: any) => p.dataKey === "rolling_mae");
  return (
    <div className="custom-tooltip">
      <div style={{ color: "#94a3b8", marginBottom: 4 }}>{label}</div>
      {mae && (
        <div style={{ color: mae.color, fontWeight: 600 }}>
          MAE: {Number(mae.value).toFixed(2)}
        </div>
      )}
    </div>
  );
};

export default function DriftMonitor({ sku, refreshTick }: Props) {
  const [events, setEvents]   = useState<DriftEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  useEffect(() => {
    if (!sku) return;
    setLoading(true);
    setError(null);
    fetchDrift(sku)
      .then(setEvents)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [sku, refreshTick]);

  if (loading) return <div className="loading-msg">Loading drift data…</div>;
  if (error)   return <div className="error-msg">Error: {error}</div>;

  if (!events.length) {
    return (
      <div className="card">
        <div className="card-title">Drift Monitor — {sku}</div>
        <div className="loading-msg" style={{ padding: "2rem 0" }}>
          ✅  No drift detected for <strong>{sku}</strong> yet.
          <br />
          <small style={{ color: "#94a3b8" }}>
            Run <code>python pipeline.py --step stream</code> to simulate streaming.
          </small>
        </div>
      </div>
    );
  }

  const threshold = events[0]?.threshold ?? 20;
  const totalDrift = events.filter((e) => e.drift_detected).length;

  const chartData = events.map((e) => ({
    date:        e.date,
    rolling_mae: +e.rolling_mae.toFixed(2),
    threshold,
  }));

  return (
    <>
      {/* Stat row */}
      <div className="stat-row">
        <div className="stat-card">
          <div className="stat-label">SKU</div>
          <div className="stat-value" style={{ fontSize: "1.2rem" }}>{sku}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Drift Events</div>
          <div className={`stat-value ${totalDrift > 0 ? "red" : "green"}`}>
            {totalDrift}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">MAE Threshold</div>
          <div className="stat-value yellow">{threshold}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Peak MAE</div>
          <div className="stat-value red">
            {Math.max(...events.map((e) => e.rolling_mae)).toFixed(2)}
          </div>
        </div>
      </div>

      {/* Rolling MAE chart */}
      <div className="card">
        <div className="card-title">Rolling MAE Over Time — {sku}</div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis
              dataKey="date"
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: "#94a3b8", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              width={40}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
            {/* Drift threshold line */}
            <RefLine
              y={threshold}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label={{ value: "threshold", fill: "#ef4444", fontSize: 10 }}
            />
            <Line
              type="monotone"
              dataKey="rolling_mae"
              stroke="#f59e0b"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
              name="Rolling MAE"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Drift events table */}
      <div className="card">
        <div className="card-title">Drift Event Log</div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>SKU</th>
                <th>Date</th>
                <th>Rolling MAE</th>
                <th>Threshold</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {events.map((e, i) => (
                <tr key={i}>
                  <td style={{ color: "#94a3b8" }}>
                    {new Date(e.timestamp).toLocaleString()}
                  </td>
                  <td><strong>{e.sku}</strong></td>
                  <td>{e.date}</td>
                  <td style={{ color: "#f59e0b" }}>{e.rolling_mae.toFixed(2)}</td>
                  <td>{e.threshold}</td>
                  <td>
                    <span className={`badge ${e.drift_detected ? "badge-red" : "badge-green"}`}>
                      {e.drift_detected ? "DRIFT" : "OK"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
