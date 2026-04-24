# 🚀 Drift-Aware Continuous Learning for Retail Demand Forecasting

## 📌 Overview


This project implements a **drift-aware, end-to-end machine learning system** for retail demand forecasting with continuous learning and MLOps capabilities.


---

## 🗃️ Data & Preprocessing

The system uses a synthetic retail dataset of **160,000 transaction records** across **72 SKUs** (2024–2025). Each record includes sales, product, region, and customer details. Data is aggregated to daily demand per SKU, with missing dates forward-filled. Four controlled concept drift windows (12 days each) are injected at random intervals, affecting ~1/6 of SKUs per window, using a demand multiplier (1.6–2.4×) for reproducibility.

---

The system forecasts product demand, monitors model performance over time, detects **concept drift**, and automatically retrains models when performance degrades. It also generates **inventory replenishment recommendations** based on updated forecasts, ensuring optimal stock levels and reducing the risk of stockouts or overstocking.


In addition, the project integrates **modern MLOps practices** including experiment tracking, orchestration, CI/CD, and monitoring, making it production-ready and scalable.

---


## 🎯 Key Features


- 📈 Demand forecasting using historical sales data

- 🗃️ Synthetic data generation, preprocessing, and drift injection
- 🔍 Concept drift detection using prediction error (MAE)
- 🕑 Dual-window (7-day & 30-day) rolling MAE drift detection
- 🔁 Automated model retraining on drift
- 🛡️ Safety gate: retrain only if new model outperforms previous on holdout
- 💾 Model versioning with timestamped artifacts
- 📊 Metrics tracking over time
- 🧾 System event logging (drift, retrain, orders)
- 📦 Inventory replenishment recommendations
- 🌐 FastAPI backend for serving data
- ⚛️ React dashboard for visualization
- 🧪 MLflow integration for experiment tracking
- ⏱️ Airflow DAG for pipeline orchestration
- 🔄 CI/CD using GitHub Actions
- 🛡️ Robust error handling and logging
- 🧩 Modular, extensible codebase

---

## 🤖 Forecasting

Forecasts are generated per SKU using **Facebook Prophet** with yearly and weekly seasonality (multiplicative mode), changepoint prior scale 0.05, and a 60-day forecast horizon. Each forecast provides a point estimate and 95% confidence interval (yhat, yhat_lower, yhat_upper).

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
---

## 🕑 Drift Detection Logic

Drift is detected using **dual rolling MAE windows**:

- **Short-term:** 7-day window (rapid detection)
- **Long-term:** 30-day window (gradual drift)

Drift is flagged if either window’s MAE exceeds **2.0× baseline** for **3+ consecutive days**. A 7-day cooldown prevents retrain oscillation. Baseline MAE is set from validation data.
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
│   ├── inventory.py
│   └── ...
│
├── dashboard/        # React frontend
├── api.py            # FastAPI backend
├── pipeline.py       # Core pipeline
├── requirements.txt
└── README.md
```

---


## ⚙️ Installation & Setup


### 1. Clone the Repository

```bash
git clone https://github.com/nidhik52/demand-forecasting-drift.git
cd demand-forecasting-drift
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

```bash
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---


## ▶️ Running the System


### 🔹 Step 1: Run the Pipeline

```bash
python pipeline.py --start 2025-07-01 --end 2025-07-10
```


This will:

- Train models
- Detect drift
- Retrain if needed
- Save models (with drift timestamp)
- Update metrics
- Generate inventory recommendations

---


### 🔹 Step 2: Start Backend (FastAPI)

```bash
uvicorn api:app --reload
```


Open the interactive API docs at:

http://127.0.0.1:8000/docs

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


Open the MLflow UI at:

http://127.0.0.1:5000

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
| `system_events.csv`             | Drift, retrain, and order logs      |
| `inventory_recommendations.csv` | Stock decisions                     |
| `models/`                       | Saved models (with drift timestamp) |

---


## 🧪 Drift Detection Logic


Drift is detected using:

     MAE > Threshold


If drift is detected:

- Event is logged
- Model is retrained
- New model is saved (with timestamp)
- MLflow run is created

---


## 📦 Inventory Logic


Inventory recommendations are based on:

- Forecasted demand
- Current stock
- Lead time

Prophet’s 95% confidence interval is used to estimate demand uncertainty:

     forecast_std = (yhat_upper - yhat_lower) / 4
     safety_stock = 1.65 × forecast_std × sqrt(lead_time)
     reorder_point = avg_daily_demand × lead_time + safety_stock
     order_quantity = avg_daily_demand × cycle_days (30)

Risk levels:

- 🟢 SAFE: Current stock ≥ reorder point
- 🟡 WARNING: Current stock < reorder point
- 🔴 CRITICAL: Current stock ≤ safety stock


Risk levels:

- 🟢 SAFE
- 🟡 WARNING
- 🔴 CRITICAL

---


## 📊 Dashboard Features


- SKU-wise demand visualization
- Drift points highlighted on graph
- Metrics trend (MAE over time)
- Inventory risk panel (SAFE, WARNING, CRITICAL)
- Order placement system (with order logging)
- Event timeline (drift, retrain, and order logs)

---


## 🔁 CI/CD Pipeline


Implemented using **GitHub Actions**:

- Automated pipeline execution on push and PR
- Output validation (checks for required files)
- API testing
- Frontend build verification

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


- Inventory order fulfillment: implement full order lifecycle (in-transit, restock after lead time, dynamic safety stock)
- Multivariate drift detection: use external signals (promotions, macroeconomic, etc.)
- Ensemble & online learning: combine Prophet, XGBoost, LSTM; enable incremental updates
- Real-time data integration: live POS feed via Kafka
- AutoML: automate retraining window/model selection (Bayesian optimization)
- A/B testing: deploy and compare multiple models in production to evaluate performance and drift response
- Cloud deployment: resolve dashboard CORS/env issues for full-stack deployment
- Add SageMaker integration for managed model training and deployment
- Advanced drift detection (KS test, ADWIN)
- Real-time monitoring with Prometheus/Grafana
- Enhance security and access control

---


## 👩‍💻 Author


**Nidhi Kambadkone**  
M.Tech Data Science

---


## 💡 Final Note


This project demonstrates how **machine learning systems evolve in production**, handling changing data patterns through **drift-aware continuous learning and MLOps practices**. Contributions and suggestions are welcome!

---

## 🐳 Docker: Run Anywhere

You can run the entire backend (API + dashboard) in a single lightweight Docker container, without installing Python, Node, or any dependencies on your machine.

### 1. Build the Docker Image

```bash
git clone https://github.com/nidhik52/demand-forecasting-drift.git
cd demand-forecasting-drift
# Build the image (replace <tag> as needed)
docker build -t demand-forecasting-drift:latest .
```

### 2. Run the Container

```bash
docker run -it --rm -p 8000:8000 demand-forecasting-drift:latest
```

- The FastAPI backend will be available at: http://localhost:8000
- The dashboard will be served at: http://localhost:8000/dashboard
- Interactive API docs: http://localhost:8000/docs

### 3. (Optional) Mount Your Own Data/Models

To use your own data/models, mount them as volumes:

```bash
docker run -it --rm -p 8000:8000 \
  -v $PWD/data:/app/data \
  -v $PWD/models:/app/models \
  demand-forecasting-drift:latest
```

### 4. Minimal Disk Usage
- Uses Python and Node slim images
- Only production frontend build is included
- No dev dependencies or build tools in final image
- Data/models are not baked in (mount as needed)

---

## 📊 Results

Key results (see paper for full details):

- Mean MAE: 8.03 units/day (all SKUs)
- Forecast accuracy: 91.8% (APPL-001, 20% tolerance)
- Drift events: 2.9% of predictions
- Post-drift retraining: up to 98.6% MAE reduction

---

## ⚠️ Known Limitations

- Inventory restock after order placement is not yet automated (panel shows pre-order stock)
- Synthetic dataset: external validity is limited
- Only Prophet evaluated (architecture supports others)
- Dashboard cloud deployment: CORS/env issues remain
- Univariate forecasting (no external covariates yet)

---

## 🌐 Dashboard Deployment Notes

The React dashboard works locally, but may face API connectivity issues on cloud platforms (Render, Vercel) due to CORS and environment variable configuration. Backend API deployment is operational; full stack cloud deployment is in progress.
