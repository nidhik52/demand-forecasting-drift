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

## Paper

**Title**: Drift-Aware Continuous Learning Framework for Retail Demand Forecasting and Inventory Replenishment

**Sections**: Introduction, Literature Review (25 papers), Methodology, System Architecture, Experiments & Results, Discussion, Conclusion
