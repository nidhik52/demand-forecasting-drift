# Drift-Aware Continuous Learning for Retail Demand Forecasting

## Project Overview

This project implements an end-to-end machine learning system for retail demand forecasting with automated concept drift detection and continuous learning.

The system forecasts product demand using historical sales data, continuously monitors incoming sales behavior, and retrains models when demand patterns change significantly.

It also generates inventory replenishment recommendations from forecasted demand.

## Key Features

- Demand forecasting using Prophet
- Simulated real-time data streaming
- Concept drift detection
- Automated model retraining
- Inventory replenishment recommendations
- System event logging
- Prediction error tracking (metrics)
- Model versioning

## System Architecture

```text
Historical Sales Data
        |
        v
Data Preprocessing
        |
        v
Demand Forecasting (Prophet)
        |
        v
Streaming Simulation
        |
        v
Drift Detection
        |
        v
Model Retraining
        |
        v
Inventory Recommendation
        |
        v
Metrics and Event Logging
```

## Project Structure

```text
demand-forecasting-drift/
├── data/
│   └── processed/
│       ├── daily_demand.csv
│       ├── forecast_2025.csv
│       ├── metrics.csv
│       ├── system_events.csv
│       ├── inventory_recommendations.csv
│       └── current_stream.csv
├── models/
│   ├── prophet_initial_model.pkl
│   └── prophet_retrained_YYYY-MM-DD.pkl
├── src/
│   ├── preprocessing.py
│   ├── forecasting.py
│   ├── streaming.py
│   ├── drift_detection.py
│   ├── retraining.py
│   ├── inventory.py
│   └── event_logger.py
├── pipeline.py
└── README.md
```

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
```

### 2. Activate environment

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install pandas prophet scikit-learn tqdm joblib
```

## Docker Deployment (Low Disk Usage)

This repository supports a single Docker image that serves both backend and frontend.

### 1. Build and run

```bash
docker compose up -d --build
```

App endpoint:

- http://localhost:8000

### 2. Auto rebuild on local file changes

Use Docker Compose watch to rebuild and restart when backend or dashboard files change:

```bash
docker compose watch
```

### 3. Auto redeploy on remote git changes (Docker-only)

If you deploy on a server and want Docker redeploy without CI/CD changes, run:

```bash
chmod +x scripts/auto-redeploy.sh
./scripts/auto-redeploy.sh
```

This script:

- polls origin/main for new commits
- runs git pull
- rebuilds and restarts with docker compose up -d --build
- prunes old images to control disk usage

Optional environment variables:

- REPO_DIR: repository path to watch
- BRANCH: branch name (default main)
- INTERVAL_SECONDS: poll interval (default 60)

Run it as a background service on Linux (systemd):

```bash
chmod +x scripts/install-auto-redeploy-service.sh
./scripts/install-auto-redeploy-service.sh
```

Service file source:

- scripts/retail-auto-redeploy.service

### 4. Disk cleanup recommendations

Run periodically on the deployment host:

```bash
docker image prune -af
docker builder prune -af
docker container prune -f
```

Install a weekly cleanup cron (Sunday 03:00):

```bash
chmod +x scripts/install-weekly-prune-cron.sh
./scripts/install-weekly-prune-cron.sh
```

## First-Time Setup

Run these once to prepare the system.

### 1. Preprocess dataset

```bash
python src/preprocessing.py
```

Creates:

- data/processed/daily_demand.csv

### 2. Generate baseline forecast

```bash
python src/forecasting.py
```

Outputs:

- data/processed/forecast_2025.csv
- models/prophet_initial_model.pkl

### 3. Generate initial inventory recommendations

```bash
python src/inventory.py
```

Outputs:

- data/processed/inventory_recommendations.csv

## Running the Streaming Pipeline

To simulate real-time demand monitoring:

```bash
python pipeline.py
```

The pipeline does the following:

- Streams simulated sales data
- Compares predicted demand with actual demand
- Detects concept drift
- Retrains forecasting model when drift is detected
- Saves new model versions
- Updates inventory recommendations
- Logs system events
- Records prediction errors

## Generated Output Files

After running the pipeline, these files are created or updated:

- data/processed/current_stream.csv
- data/processed/metrics.csv
- data/processed/system_events.csv
- data/processed/inventory_recommendations.csv

## Model Versioning

Models are stored in the models directory.

Example:

```text
models/
├── prophet_initial_model.pkl
├── prophet_retrained_2025-08-01.pkl
└── prophet_retrained_2025-09-10.pkl
```

Each retrained model includes the drift date in its filename to track model evolution.

## Concept Drift Detection

Concept drift occurs when the relationship between input data and demand changes over time.

The system compares:

- Actual demand
- Predicted demand

If error exceeds a configured threshold across monitored SKUs, the system:

- Logs a drift event
- Retrains the forecasting model
- Saves a new model version
- Updates inventory recommendations

## Monitoring Model Performance

Prediction metrics are stored in:

- data/processed/metrics.csv

Example:

```text
Date,SKU,Actual_Demand,Predicted_Demand,Error
2025-07-01,ELEC-001,12,11.8,0.2
```

These metrics support analysis of:

- Prediction accuracy
- Drift events
- Demand fluctuations

## Inventory Recommendation Logic

Inventory decisions are based on:

- Average forecast demand
- Lead time
- Safety stock
- Current inventory

Example action:

- Order 25 units by 2025-08-10

This helps avoid stockouts while reducing overstock risk.

## Adjusting Forecast Horizon

You can modify the horizon in src/forecasting.py.

Current pattern:

```python
future = model.make_future_dataframe(periods=365)
```

Common alternatives:

- periods=30 for next 30 days
- periods=90 for next 3 months
- periods=365 for next 1 year

## System Event Logging

All system events are recorded in:

- data/processed/system_events.csv

Example events:

```text
2025-08-01 Drift detected
2025-08-01 Model retraining started
2025-08-01 Model retraining completed
```

This provides traceability across the model lifecycle.

## Future Improvements

- Real-time streaming with Kafka
- Advanced drift detection algorithms
- Real-time monitoring dashboards
- Containerization with Docker
- Kubernetes deployment

## Author

Nidhi Kambadkone
M.Tech Data Science, 1MS24SDS08

Drift-Aware Continuous Learning Framework for Retail Demand Forecasting and Inventory Replenishment
