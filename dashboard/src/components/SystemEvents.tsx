/**
 * SystemEvents.tsx
 *
 * Aggregates drift detections and inventory recommendations into a single
 * chronological event log, simulating a live monitoring feed.
 */

import { useEffect, useState } from "react";
import { fetchDrift, fetchInventory } from "../api";

interface SystemEvent {
  time: string;
  message: string;
  kind: "drift" | "retrain" | "inventory";
}

const ICON: Record<string, string> = {
  drift:     "🔴",
  retrain:   "🔄",
  inventory: "📦",
};

interface Props { refreshTick: number; }

export default function SystemEvents({ refreshTick }: Props) {
  const [events, setEvents] = useState<SystemEvent[]>([]);

  useEffect(() => {
    Promise.all([fetchDrift(), fetchInventory()])
      .then(([driftEvents, inventory]) => {
        const sys: SystemEvent[] = [];

        // Drift detections → also synthesise a "retrained" event
        driftEvents
          .filter((e) => e.drift_detected)
          .forEach((e) => {
            const t = new Date(e.timestamp).toLocaleTimeString();
            sys.push({ time: t, message: `Drift detected for ${e.sku}`, kind: "drift" });
            sys.push({ time: t, message: `Model retrained for ${e.sku}`, kind: "retrain" });
          });

        // Inventory orders as events (use order date as the time label)
        inventory
          .filter((r) => r.Recommended_Order_Qty > 0)
          .slice(0, 5)
          .forEach((r) => {
            sys.push({
              time: r.Recommended_Order_Date,
              message: `Inventory recommendation: order ${r.Recommended_Order_Qty.toFixed(0)} units of ${r.SKU}`,
              kind: "inventory",
            });
          });

        // Most-recent first, cap at 12 entries
        setEvents(sys.slice(-12).reverse());
      })
      .catch(() => {});
  }, [refreshTick]);

  return (
    <div className="card" style={{ height: "100%" }}>
      <div className="card-title">🗂️ Recent System Events</div>
      {events.length === 0 && (
        <div className="no-alerts">No events yet — run the pipeline to generate activity.</div>
      )}
      <div className="event-log">
        {events.map((e, i) => (
          <div key={i} className="event-item">
            <span className="event-dot">{ICON[e.kind]}</span>
            <span className="event-time">{e.time}</span>
            <span className="event-msg">{e.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
