# Kafka Real-Time Simulation

Streams historical retail transactions through Apache Kafka to simulate a live POS feed, demonstrating drift detection and automatic Prophet retraining in **real time**.

---

## What It Does

| Component | File | Role |
|---|---|---|
| `docker-compose.kafka.yml` | Zookeeper + Kafka + Kafka UI | Message broker infrastructure |
| `producer.py` | Sends 1 row/transaction to `sales_events` | Simulates live POS stream |
| `consumer.py` | Aggregates daily demand, detects drift, retrains | MLOps pipeline on live events |

---

## Quick Start

```bash
# 1. Start Kafka stack
docker compose -f kafka/docker-compose.kafka.yml up -d

# 2. Wait ~20 seconds for Kafka to be healthy, then open the UI
#    http://localhost:8080  — Kafka UI (browse topics, messages)

# 3. In terminal A — start the consumer (waits for messages):
python kafka/consumer.py

# 4. In terminal B — stream the drift window at 200 rows/sec:
python kafka/producer.py --start-date 2025-11-01 --speed 200

# 5. Watch drift being detected in terminal A in near-real time
#    MLflow runs appear live at: mlflow ui --backend-store-uri mlruns/
```

---

## What You'll See

The drift window (Nov 1 – Dec 31, 2025) has two injected events:
- **Electronics & Tech** — abrupt +50% demand surge (Black Friday / holiday)
- **Health & Personal Care** — gradual +40% ramp (growing wellness trend)

Consumer terminal output:
```
  2025-11-01  ✅Elec 0.82x  |  ✅Ente 0.71x  |  ✅Fash 0.65x  | ...
  2025-11-02  🔴Elec 1.63x  |  ✅Ente 0.74x  |  ...
  2025-11-03  🔴Elec 1.71x  |  ...

  ⚡ RETRAIN TRIGGERED  Electronics & Tech  (2025-11-03)
     Decision : ACCEPTED  new MAE 5,234
```

---

## Options

### Producer
```
--bootstrap   Kafka broker      (default: localhost:9092)
--topic       Topic name        (default: sales_events)
--speed       Rows per second   (default: 200)
--start-date  Stream from date  (default: all data, format YYYY-MM-DD)
--dry-run     Print 5 messages without connecting
```

### Consumer
```
--bootstrap   Kafka broker               (default: localhost:9092)
--topic       Topic name                 (default: sales_events)
--threshold   Drift MAE ratio threshold  (default: 1.5)
--model-dir   .pkl model directory       (default: models/)
--val-days    Baseline calc window       (default: 30)
--idle-flush  Flush last day after N sec (default: 5)
```

---

## Teardown

```bash
docker compose -f kafka/docker-compose.kafka.yml down -v
```

---

## MLflow Integration

Retrain runs triggered by the consumer are logged to a **separate** MLflow experiment:  
`demand_forecasting_drift_kafka`

This keeps real-time triggered runs distinct from CI/CD batch runs.

```bash
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000
# → Switch to 'demand_forecasting_drift_kafka' experiment
```
