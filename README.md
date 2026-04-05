# 🚀 Drift-Aware Continuous Learning for Retail Demand Forecasting

## 📌 Overview

This project implements a **drift-aware, end-to-end machine learning system** for retail demand forecasting with continuous learning capabilities.

The system forecasts product demand, monitors model performance over time, detects **concept drift**, and automatically retrains models when performance degrades. It also generates **inventory replenishment recommendations** based on updated forecasts.

In addition, the project integrates **modern MLOps practices** including experiment tracking, orchestration, CI/CD, and monitoring.

---

## 🎯 Key Features

* 📈 Demand forecasting using historical sales data
* 🔍 Concept drift detection using prediction error (MAE)
* 🔁 Automated model retraining on drift
* 💾 Model versioning with timestamped artifacts
* 📊 Metrics tracking over time
* 🧾 System event logging (drift, retrain, orders)
* 📦 Inventory replenishment recommendations
* 🌐 FastAPI backend for serving data
* ⚛️ React dashboard for visualization
* 🧪 MLflow integration for experiment tracking
* ⏱️ Airflow DAG for pipeline orchestration
* 🔄 CI/CD using GitHub Actions

---

## 🧠 System Architecture

```text
Historical Data
     ↓
Data Processing
     ↓
Model Training
     ↓
Prediction + Evaluation
     ↓
Drift Detection (MAE Threshold)
     ↓
Retraining (if drift detected)
     ↓
Model Versioning + MLflow Logging
     ↓
Inventory Recommendation
     ↓
API + Dashboard + Monitoring
```

---

## 📁 Project Structure

```text
demand-forecasting-drift/
│
├── data/
│   └── processed/
│       ├── daily_demand.csv
│       ├── metrics.csv
│       ├── system_events.csv
│       └── inventory_recommendations.csv
│
├── models/
│   └── (saved models with SKU + timestamp)
│
├── src/
│   ├── config.py
│   ├── event_logger.py
│   └── inventory.py
│
├── dashboard/        # React frontend
├── api.py            # FastAPI backend
├── pipeline.py       # Core pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate Environment

```bash
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the System

### 🔹 Step 1: Run Pipeline

```bash
python pipeline.py --start 2025-07-01 --end 2025-07-10
```

This will:

* Train models
* Detect drift
* Retrain if needed
* Save models
* Update metrics
* Generate inventory recommendations

---

### 🔹 Step 2: Start Backend (FastAPI)

```bash
uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

### 🔹 Step 3: Start Frontend (React Dashboard)

```bash
cd dashboard
npm install
npm start
```

---

### 🔹 Step 4: Run MLflow (Experiment Tracking)

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

### 🔹 Step 5: Run Airflow (Pipeline Orchestration)

```bash
airflow webserver --port 8080
airflow scheduler
```

---

## 📊 Outputs Generated

After running the pipeline:

| File                            | Description                         |
| ------------------------------- | ----------------------------------- |
| `metrics.csv`                   | MAE over time (for drift detection) |
| `system_events.csv`             | Drift + retrain logs                |
| `inventory_recommendations.csv` | Stock decisions                     |
| `models/`                       | Saved models (only on drift)        |

---

## 🧪 Drift Detection Logic

Drift is detected using:

```text
MAE > Threshold
```

If drift is detected:

* Event is logged
* Model is retrained
* New model is saved
* MLflow run is created

---

## 📦 Inventory Logic

Inventory recommendations are based on:

* Forecasted demand
* Current stock
* Lead time

Risk levels:

* 🟢 SAFE
* 🟡 WARNING
* 🔴 CRITICAL

---

## 📊 Dashboard Features

* SKU-wise demand visualization
* Drift points highlighted on graph
* Metrics trend (MAE over time)
* Inventory risk panel
* Order placement system
* Event timeline (drift + retrain logs)

---

## 🔁 CI/CD Pipeline

Implemented using **GitHub Actions**:

* Automated pipeline execution
* Output validation
* API testing
* Frontend build verification

---

## 🧠 MLOps Components Implemented

| Component                    | Status          |
| ---------------------------- | --------------  |
| MLflow (Experiment Tracking) | ✅              |
| Airflow (Orchestration)      | ✅              |
| CI/CD (GitHub Actions)       | ✅              |
| FastAPI (Serving)            | ✅              |
| React Dashboard              | ✅              |
| Grafana (Monitoring)         | ✅              |
| Docker / K8s / Kafka         | 📌 Future Work  |

---

## 🚧 Future Work

* Kafka for real-time streaming
* Kubernetes deployment
* SageMaker integration
* Advanced drift detection methods
* Real-time monitoring with Prometheus + Grafana

---

## 👩‍💻 Author

**Nidhi Kambadkone**
M.Tech Data Science

---

## 💡 Final Note

This project demonstrates how **machine learning systems evolve in production**, handling changing data patterns through **drift-aware continuous learning and MLOps practices**.
