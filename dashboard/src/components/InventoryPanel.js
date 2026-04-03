import React, { useEffect, useState } from "react";
export default function InventoryPanel() {
  const [data, setData] = useState([]);
  const [orders, setOrders] = useState({});
  const [error, setError] = useState("");

  const loadData = () => {
    fetch("/inventory")
      .then((res) => {
        if (!res.ok) throw new Error(`Inventory API returned ${res.status}`);
        return res.json();
      })
      .then((rows) => {
        setData(rows);
        setError("");
      })
      .catch(() => {
        setData([]);
        setError("API unreachable. Ensure backend is running on port 8000.");
      });
  };

  useEffect(() => {
    loadData();
  }, []);

  const placeOrder = (sku) => {
    const qty = Number(orders[sku] || 0);
    if (!Number.isFinite(qty) || qty <= 0) {
      alert("Enter a valid order quantity greater than 0");
      return;
    }

    fetch(`/order?sku=${encodeURIComponent(sku)}&qty=${qty}`, {
      method: "POST"
    })
      .then((res) => {
        if (!res.ok) throw new Error(`Order API returned ${res.status}`);
        return res.json();
      })
      .then(() => {
        alert("Order placed");
        loadData();
      })
      .catch(() => {
        alert("Order failed. Check backend availability.");
      });
  };

  return (
    <div className="glass-card">
      <h3>📦 Inventory</h3>

      {error && <p style={{ color: "#ff6b6b" }}>{error}</p>}

      {data.slice(0, 10).map((item, i) => (
        <div key={i} style={{ marginBottom: "12px" }}>
          <strong>{item.SKU}</strong> - {item.Risk_Level}
          <br />
          Stock: {item.Current_Stock} | Suggested: {item.Recommended_Order_Qty}

          <div style={{ marginTop: "6px" }}>
            <input
              type="number"
              min="1"
              placeholder="Qty"
              onChange={(e) =>
                setOrders({ ...orders, [item.SKU]: e.target.value })
              }
            />
            <button onClick={() => placeOrder(item.SKU)} style={{ marginLeft: "8px" }}>
              Order
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}