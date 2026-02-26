# Drift-Aware Continuous Learning Framework for Retail Demand Forecasting

> MTech Final Year Project | 2025-2026

## Overview

A drift-aware continuous learning framework that automatically detects concept drift in retail demand patterns and retrains forecasting models to maintain reliable inventory replenishment decisions.

## System Architecture

```
Raw Sales Data
      │
      ▼
┌─────────────┐
│ Data Layer  │  Preprocessing, aggregation, feature engineering
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Forecasting     │  Prophet — daily demand forecast (30-day horizon)
│ Layer           │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Monitoring &    │  Rolling MAE drift detection
│ Drift Detection │
└──────┬──────────┘
       │ drift detected?
       ▼
┌─────────────────┐
│ Auto Retraining │  Retrain on recent window → MLflow versioning
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Inventory       │  Safety stock, reorder point, replenishment alerts
│ Decisions       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Dashboard       │  Streamlit — forecasts, drift events, inventory
└─────────────────┘
```

## Project Structure

```
demand-forecasting-drift/
│
├── data/
│   ├── raw/                  # Original sales_100k.csv
│   ├── processed/            # final_demand_series.csv
│   └── drift_logs/           # Drift event records
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_forecasting.ipynb  # Prophet baseline
│   ├── 03_drift_detection.ipynb
│   └── 04_inventory.ipynb
│
├── src/
│   ├── data/                 # Data loading, preprocessing
│   ├── forecasting/          # Prophet model wrapper
│   ├── drift_detection/      # Rolling MAE drift detector
│   ├── retraining/           # Auto retraining logic
│   ├── inventory/            # Safety stock, reorder logic
│   └── monitoring/           # Evidently AI integration
│
├── api/                      # FastAPI backend
├── dashboard/                # Streamlit frontend
├── docker/                   # Dockerfile, docker-compose
├── .github/workflows/        # GitHub Actions CI/CD
├── reports/                  # Paper figures, results
├── requirements.txt
└── README.md
```

## Dataset

- **Source**: Synthetic retail sales dataset (100k transactions, 2025)
- **Extended**: Jan 2024 – Mar 2026 using statistical distribution preservation
- **Categories**: 6 macro-groups (Electronics & Tech, Health & Personal Care, Home & Lifestyle, Sports & Outdoors, Fashion & Accessories, Entertainment & Office)
- **Drift injected**:
  - Abrupt: Electronics & Tech +50% from Apr 2025
  - Gradual: Health & Personal Care +40% ramp Aug–Oct 2025

## Train / Test / Predict Split

| Period | Purpose |
|--------|---------|
| Jan 2024 – Oct 2025 | Training |
| Nov – Dec 2025 | Test / Drift simulation |
| Jan – Mar 2026 | Future prediction target |

## Tech Stack

| Component | Tool |
|-----------|------|
| Forecasting | Prophet |
| Drift Detection | Rolling MAE |
| Experiment Tracking | MLflow |
| Monitoring | Evidently AI |
| Backend API | FastAPI |
| Frontend | Streamlit |
| CI/CD | GitHub Actions |
| Containerization | Docker |

## Setup

```bash
# Clone repo and open in GitHub Codespaces (recommended)
# Or locally:
pip install -r requirements.txt

# Prepare dataset
python src/data/prepare_dataset.py

# Run forecasting baseline
jupyter notebook notebooks/02_forecasting.ipynb

# Start API
uvicorn api.main:app --reload --port 8000

# Start dashboard
streamlit run dashboard/app.py

# Start MLflow UI
mlflow ui --port 5000
```

## MLOps Features

- **Experiment tracking**: Every model training run logged in MLflow
- **Model versioning**: Automatic version bump on retrain
- **Drift monitoring**: Evidently AI reports on data drift
- **CI/CD pipeline**: GitHub Actions runs drift check daily
- **Containerization**: Docker for reproducible deployment

## Results

*To be updated as experiments are completed.*

## Future Work

The following extensions would bridge this prototype toward a production-grade MLOps system:

Points marked ✅ have been prototyped in this repository.

1. **Real Retail Data Integration** — Connect to a live POS/ERP feed via REST or Kafka for true online learning rather than simulated drift events, validating the framework on non-stationary real-world demand with genuine noise characteristics.

2. **Adaptive Drift Threshold** — Replace the fixed 2.0× MAE threshold with a statistically learned threshold (e.g., CUSUM or ADWIN) that adjusts dynamically to each category's natural demand volatility, reducing false positives during seasonal peaks without manual tuning.

3. ✅ **Pipeline Orchestration with Apache Airflow** — Migrate the walk-forward loop into scheduled Airflow DAGs for fault-tolerant, retry-able execution, enabling the drift-detection → retrain → deploy cycle to run autonomously on a daily schedule.

4. **Model Registry with A/B Testing** — Use the MLflow Model Registry to manage staged promotions (Staging → Production) with a shadow-mode A/B evaluation period before full deployment, further reducing the risk of deploying a model that degrades on unseen distribution shifts.

5. ✅ **Data Quality Gates** — Add pre-training data validation using Great Expectations (schema checks, outlier detection, missing-value thresholds) to prevent degraded retrains caused by upstream data pipeline issues — a common failure mode in production forecasting systems.

6. **Multi-Model Ensemble / Online Learning Fallback** — Investigate combining Prophet with an online learning model (River or scikit-multiflow) to handle abrupt drift faster than the current 45-day retrain window allows, with Prophet handling trend/seasonality and the online model adapting within days of a distribution shift.

7. ✅ **Containerised, Autoscaled Deployment** — Dockerise the FastAPI serving layer and deploy with Kubernetes horizontal pod autoscaling to handle variable inference load, moving from a single-process server to a production-grade, fault-tolerant serving infrastructure.

---

> **Reminder:** Once the core project is finalised, run the implemented Future Work extensions using the commands below.

### Running FW3 — Airflow Pipeline Orchestration

```bash
# Install Airflow (one-time)
pip install apache-airflow==2.9.0

# Point Airflow to the dags folder
export AIRFLOW_HOME=$(pwd)/.airflow
airflow db init
airflow dags list   # should show 'demand_forecasting_drift_pipeline'

# Start the scheduler + webserver (UI at http://localhost:8080)
airflow scheduler &
airflow webserver --port 8080

# Trigger a manual run
airflow dags trigger demand_forecasting_drift_pipeline
```

> DAG file: [dags/drift_pipeline_dag.py](dags/drift_pipeline_dag.py)

---

### Running FW5 — Data Quality Gates

```bash
# Run standalone validation against the processed dataset
python src/data/quality_checks.py data/processed/final_demand_series.csv

# Or import and call programmatically
python - <<'EOF'
import pandas as pd
from src.data.quality_checks import DemandDataValidator

df = pd.read_csv("data/processed/final_demand_series.csv")
validator = DemandDataValidator(df, use_great_expectations=False)
results = validator.run_all_checks()
for r in results:
    status = "PASS" if r.passed else "FAIL"
    print(f"[{status}] {r.check}: {r.message}")
EOF
```

> Validator file: [src/data/quality_checks.py](src/data/quality_checks.py)

---

### Running FW7 — Docker Deployment

```bash
# Build and start all services (API, Dashboard, MLflow, Airflow)
docker compose -f docker/docker-compose.yml up --build

# Services exposed:
#   FastAPI   → http://localhost:8000
#   Streamlit → http://localhost:8501
#   MLflow UI → http://localhost:5000
#   Airflow   → http://localhost:8080

# Stop and clean up
docker compose -f docker/docker-compose.yml down
```

> Docker files: [docker/Dockerfile](docker/Dockerfile), [docker/docker-compose.yml](docker/docker-compose.yml)

---

## Paper

**Title**: Drift-Aware Continuous Learning Framework for Retail Demand Forecasting and Inventory Replenishment

**Sections**: Introduction, Literature Review (25 papers), Methodology, System Architecture, Experiments & Results, Discussion, Conclusion
