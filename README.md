# 🚀 Drift-Aware Continuous Learning for Retail Demand Forecasting

## 📌 Overview


This project implements a **drift-aware, end-to-end machine learning system** for retail demand forecasting with continuous learning and MLOps capabilities.


---

## 🗃️ Data & Preprocessing

The system uses the Kaggle *Retail Sales Data 2025* (Noir1112): **160,000 transaction records** across **72 SKUs**, January 2024 – January 2025. Unit quantities are synthesized from category heuristics and regions are reassigned at random. Data are aggregated to daily demand per SKU; missing dates are filled with zero demand. The series is extended to December 2025 as each SKU's historical mean × U[0.8, 1.2]. Four drift windows are then injected with `numpy.random.default_rng(42)`: 2024-03-05→03-17, 2024-09-18→09-30, 2024-10-15→10-27 and 2025-02-09→02-21 (13 days each), each covering 12 SKUs with a demand multiplier drawn from U[1.6, 2.4].

**Reproducibility note.** Seed 42 governs only the drift injection, not the region assignment, quantity synthesis or 2025 extension. Re-running preprocessing therefore gives a different `daily_demand.csv`. The committed `data/processed/daily_demand.csv` is the fixed input used for all reported results.

---

The system forecasts product demand, monitors model performance over time, detects **concept drift**, and automatically retrains models when performance degrades. It also generates **inventory replenishment recommendations** based on updated forecasts, ensuring optimal stock levels and reducing the risk of stockouts or overstocking.


In addition, the project integrates **modern MLOps practices** including experiment tracking, orchestration, CI/CD, and monitoring, making it production-ready and scalable.

---


## 🎯 Key Features


- 📈 Demand forecasting using historical sales data

- 🗃️ Synthetic data generation, preprocessing, and drift injection
- 🔍 Concept drift detection using prediction error (MAE)
- 🕑 Rolling 7-day MAE drift detection with consecutive-day confirmation and cooldown
- 🔁 Automated model retraining on drift
- 🛡️ Safety gate: a retrained candidate replaces the current model only if its MAPE on the 7 held-out days before the trigger is strictly lower (`gate_decisions.csv`)
- 💾 Model versioning with timestamped artifacts
- 📊 Metrics tracking over time
- 🧾 System event logging (drift, retrain, orders)
- 📦 Inventory replenishment recommendations
- 🌐 FastAPI backend for serving data
- ⚛️ React dashboard for visualization
- 🧪 MLflow integration for experiment tracking
- ⏱️ Airflow-compatible CLI (`pipeline.py --start --end`); no Airflow DAG is included in this repository
- 🔄 CI/CD using GitHub Actions
- 🛡️ Robust error handling and logging
- 🧩 Modular, extensible codebase

---

## 🤖 Forecasting

Forecasts are generated per SKU with **Prophet** (default additive model; negative forecasts clipped to zero).

- **Daily monitoring loop (`pipeline.py`).** Each SKU model is fitted on the history before the current day and forecasts that day. Daily, weekly and yearly seasonality are set to `"auto"`; with under two years of daily data, only weekly seasonality is active.
- **30-day export (`src/forecasting.py`).** Fits on the pipeline window with daily and weekly seasonality (yearly off) and writes `forecast_2025.csv`. Its in-window values are in-sample fits, so they are not used for evaluation.

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
Retraining + MAPE safety gate (if drift detected)
     ↓
Model Versioning + MLflow Logging
     ↓
Inventory Recommendation
     ↓
API + Dashboard + Monitoring
```

---

## 🕑 Drift Detection Logic

Drift is detected per SKU with a **7-day rolling MAE**:

- The baseline is the first full 7-day window mean, then frozen.
- Drift is flagged when rolling MAE / baseline **> 2.0** on **3 consecutive days**.
- A **7-day cooldown** limits repeated retraining.
- After an accepted retrain, the baseline is reset to the new model's error on the trigger day.

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
│   └── revision/     # outputs used in the revised paper
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
├── revision/         # JCMM revision experiments (see revision/README.md)
├── tests/            # pytest tests for the safety gate and revision code
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

Useful options:

| Option | Effect |
|---|---|
| `--history_start 2024-01-01` | Train on all data before each day |
| `--seed 42` | Reproducible simulated deliveries |
| `--output_dir <dir>` | Write outputs to another folder instead of `data/processed/` |
| `--inventory_file <csv>` | Opening stock |
| `--no_safety_gate` | Disable the gate |

To reproduce the paper's results, follow [`revision/README.md`](revision/README.md). Prophet 1.1.7 needs `requirements-prophet.txt`; set `USE_BUNDLED_CMDSTAN=1` to use its bundled CmdStan.


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


### 🔹 Step 5: Scheduling (optional)

No Airflow DAG is included in this repository. The pipeline is a CLI (`python pipeline.py --start ... --end ...`), so any scheduler (Airflow, cron, GitHub Actions) can call it.

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


See *Drift Detection Logic* above. When drift is confirmed and the cooldown allows retraining:

- A `DRIFT` event is logged.
- A candidate is fitted on data before the 7-day held-out window H, and compared with the current model by MAPE on H.
- **Accepted** (strictly lower MAPE): the candidate is deployed, saved with a timestamp, logged to MLflow and recorded as `RETRAIN_ACCEPTED`.
- **Rejected:** the current model is kept, the event is recorded as `RETRAIN_REJECTED`, and the cooldown starts.
- `python pipeline.py ... --no_safety_gate` restores the older replace-on-every-trigger behaviour.

---


## 📦 Inventory Logic


Daily per-SKU recommendations (`pipeline.py` → `inventory_recommendations.csv`):

     S_t  = stock after the day's demand (+ simulated delivery of 0/30/60/120 units if stock < 20)
     D_L  = lead_time × ŷ_t                 (ŷ_t = one-day-ahead forecast, clipped at 0)
     ROP  = D_L + safety_stock              (safety_stock = 10 units)
     Q    = max(0, ⌊ROP − S_t⌋)

Risk levels:

- 🟢 SAFE: Q = 0
- 🟡 WARNING: 0 < Q ≤ 100
- 🔴 CRITICAL: Q > 100

Opening stock, lead times (5/7/10 days) and stock profiles come from `src/generate_inventory.py` (seed 42). A separate single-date routine (`src/inventory.py`) sums the 30-day forecast over the lead time and adds a 20% buffer. It writes `inventory_recommendations_forecast.csv`.

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
| Airflow (Orchestration)      | 📌 No DAG included |
| CI/CD (GitHub Actions)       | ✅              |
| FastAPI (Serving)            | ✅              |
| React Dashboard              | ✅              |
| Docker                       | ✅              |
| Grafana / K8s / Kafka        | 📌 Future Work  |

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
- Pre-seeded `data/` is copied into the image; models are not (mount volumes to use your own)

---

## 📊 Results

Results reported in the revised JCMM manuscript are listed below. They come from commit `dd1a6fa`, with outputs in `data/revision/`; reproduction steps are in [`revision/README.md`](revision/README.md).

**Leakage-free one-day-ahead forecasts** (rolling origin, 2025-07-03 → 07-31, 72 SKUs, 2,088 SKU-days):

| Forecast | MAE (units/day) | MAPE | Within 20% |
|---|---|---|---|
| Prophet (refit daily) | 1.49 | 11.65% | 84.29% |
| Naive (y_{t−1}) | 1.83 | 13.92% | 75.34% |
| Seasonal naive (y_{t−7}) | 1.85 | 14.03% | 74.62% |
| Historical mean | 1.47 | 11.67% | 80.99% |
| Drift-aware pipeline (July run) | 1.51 | 11.82% | 83.29% |

**Drift detection** against the four injected windows (pipeline run 2024-02-01 → 2025-03-31):
- 21 of 48 SKU-window injections detected; median delay 6 days.
- 436 false-alarm alert-days.

**Safety gate:**
- Drift run: 139 of 231 candidates accepted (60.2%).
- July run: 2 of 2 accepted.

**Inventory:** 2,088 daily assessments in the July run: SAFE 484, WARNING 1,345, CRITICAL 259.

The data are semi-synthetic and the drift is self-injected, so these results do not establish performance on real retail demand. Files in `data/processed/` are the archived outputs of commit `f264989` (pre-revision). Their `forecast_2025.csv` July values are in-sample fits.

---

## ⚠️ Known Limitations

- Inventory restock after order placement is not yet automated (panel shows pre-order stock)
- Semi-synthetic dataset and self-injected drift: external validity is limited
- The drift detector raises many false alarms with the default settings (see Results)
- Preprocessing uses unseeded draws (only drift injection is seeded)
- Only Prophet evaluated (architecture supports others)
- Dashboard cloud deployment: CORS/env issues remain
- Univariate forecasting (no external covariates yet)

---

## 🌐 Dashboard Deployment Notes

The React dashboard works locally, but may face API connectivity issues on cloud platforms (Render, Vercel) due to CORS and environment variable configuration. Backend API deployment is operational; full stack cloud deployment is in progress.
