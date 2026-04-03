import React, { useState } from "react";

export default function RunPipelineButton() {
  const [start, setStart] = useState("2025-07-01");
  const [end, setEnd] = useState("2025-07-31");

  const runPipeline = () => {
    fetch(`/run-pipeline?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`, {
      method: "POST"
    })
      .then((res) => {
        if (!res.ok) throw new Error(`Pipeline API returned ${res.status}`);
        return res.json();
      })
      .then(() => alert("Pipeline executed"))
      .catch(() => alert("Pipeline trigger failed. Check backend availability."));
  };

  return (
    <div className="glass-card">
      <h3>🚀 Run Pipeline</h3>

      <input type="date" value={start} onChange={e => setStart(e.target.value)} />
      <input type="date" value={end} onChange={e => setEnd(e.target.value)} />

      <button onClick={runPipeline}>Run</button>
    </div>
  );
}