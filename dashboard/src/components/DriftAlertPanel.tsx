/**
 * DriftAlertPanel.tsx
 *
 * Displays recent drift detection events as highlighted alerts.
 * Refreshed on every refreshTick (every 5 s in auto-refresh mode).
 */

import { useEffect, useState } from "react";
import { fetchDrift } from "../api";
import type { DriftEvent } from "../types";

interface Props { refreshTick: number; }

export default function DriftAlertPanel({ refreshTick }: Props) {
  const [alerts, setAlerts] = useState<DriftEvent[]>([]);
  const [error, setError]   = useState<string | null>(null);

  useEffect(() => {
    fetchDrift()
      .then((all) => {
        // Keep only actual drift detections, most-recent first, cap at 8
        const detected = all
          .filter((e) => e.drift_detected)
          .slice(-8)
          .reverse();
        setAlerts(detected);
        setError(null);
      })
      .catch(() => setError("Could not reach API — run the backend first."));
  }, [refreshTick]);

  return (
    <div className="card alert-card">
      <div className="card-title">⚠️ Drift Alert Panel</div>

      {error && (
        <div className="no-alerts" style={{ color: "var(--muted)" }}>{error}</div>
      )}

      {!error && alerts.length === 0 && (
        <div className="no-alerts">
          ✅ No drift events detected yet.
          <br />
          <small>Run <code>python pipeline.py --step stream</code> to simulate streaming.</small>
        </div>
      )}

      {alerts.map((e, i) => (
        <div key={i} className="alert-item">
          <div className="alert-icon">⚠</div>
          <div className="alert-body">
            <div className="alert-title">
              Drift detected for SKU <strong>{e.sku}</strong>
            </div>
            <div className="alert-sub">
              Model retrained automatically · MAE {e.rolling_mae.toFixed(2)} · threshold {e.threshold}
            </div>
          </div>
          <div className="alert-meta">
            <div className="alert-sku-badge">{e.sku}</div>
            <div className="alert-time">
              {new Date(e.timestamp).toLocaleTimeString()}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
