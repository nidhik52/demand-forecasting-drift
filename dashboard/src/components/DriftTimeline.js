import React, { useEffect, useState } from "react";
import Papa from "papaparse";

const dataPath = (fileName) => `${process.env.PUBLIC_URL || ""}/data/${fileName}`;

export default function DriftTimeline() {
  const [data, setData] = useState([]);

  useEffect(() => {
    Papa.parse(dataPath("system_events.csv"), {
      download: true,
      header: true,
      complete: (result) => {
        const drift = result.data
          .filter(d => d.event_type === "DRIFT")
          .slice(-10);

        setData(drift);
      }
    });
  }, []);

  return (
    <div className="glass-card">
      <h3>⚠ Drift Events</h3>

      {data.map((d, i) => (
        <div key={i}>
          {d.timestamp} — {d.message}
        </div>
      ))}
    </div>
  );
}