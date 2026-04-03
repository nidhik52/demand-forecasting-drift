import React, { useEffect, useState } from "react";
import Papa from "papaparse";

const dataPath = (fileName) => `${process.env.PUBLIC_URL || ""}/data/${fileName}`;

export default function RetrainPanel() {
  const [data, setData] = useState([]);

  useEffect(() => {
    Papa.parse(dataPath("system_events.csv"), {
      download: true,
      header: true,
      complete: (result) => {
        const retrain = result.data
          .filter(d => d.event_type === "RETRAIN")
          .slice(-5);

        setData(retrain);
      }
    });
  }, []);

  return (
    <div className="glass-card">
      <h3>🔁 Retraining History</h3>

      {data.map((d, i) => (
        <div key={i}>
          {d.date} — {d.message}
        </div>
      ))}
    </div>
  );
}