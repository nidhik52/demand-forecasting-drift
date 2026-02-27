# AI Session Context — demand-forecasting-drift

> **Last updated:** 2026-02-27 (session 4 — MLflow schema/artifacts/registry polished; Kafka real-time simulation planned)  
> **Purpose:** Persistent context for Claude AI (or any LLM assistant) picking up this project mid-stream. Read this before starting any new session.

---

## Project Overview

Retail demand-forecasting system for an Indian e-commerce dataset (`data/raw/sales_100k.csv`). The pipeline:

1. **Ingests** raw sales data → `data/processed/final_demand_series.csv` (category-level daily demand, rupees)
2. **Forecasts** 2026 demand per category using Prophet
3. **Detects drift** by comparing rolling MAE against a baseline window
4. **Retrains** models when drift is detected
5. **Computes inventory** decisions (reorder points, safety stock) using the forecasts

Six product categories (all tracked): **Electronics & Tech**, **Entertainment & Office**, **Fashion & Accessories**, **Health & Personal Care**, **Home & Lifestyle**, **Sports & Outdoors**.

---

## Repository Layout (key files only)

```
src/
  data/
    prepare_dataset.py          # single-category prep (legacy)
    master_data_pipeline.py     # runs all categories → final_demand_series.csv
    quality_checks.py           # Great Expectations wrapper (FW5)
  forecasting/
    prophet_model.py            # ProphetForecaster + ForecasterRegistry classes
  drift_detection/
    run_drift_check.py          # headless CI/CD script; exits 0=clean 1=drift 2=missing
    drift_detector.py           # core drift logic
  retraining/
    retrain_pipeline.py         # MLflow-tracked retraining
  inventory/
    replenishment.py            # InventoryCalculator class
  monitoring/
    (empty — placeholder)

notebooks/
  01_eda.ipynb                  # exploratory data analysis
  02_model_comparison.ipynb     # Prophet vs SARIMA vs XGBoost
  03_drift_detection.ipynb      # drift simulation over 61 days
  04_retraining.ipynb           # retrain on drifted data, compare MAE
  05_inventory.ipynb            # NOT YET CREATED — next priority
  06_kafka_simulation.ipynb     # NOT YET CREATED — real-time Kafka simulation

api/                            # EMPTY — second priority
dashboard/                      # EMPTY — third priority

kafka/                          # NOT YET CREATED — real-time streaming
  producer.py                   # Kafka producer: streams sales_100k.csv row-by-row
  consumer.py                   # Kafka consumer: triggers drift check + retrain
  docker-compose.kafka.yml      # Zookeeper + Kafka + Schema Registry

dags/
  drift_pipeline_dag.py         # Airflow DAG (FW3 — daily, 7 tasks)

docker/
  Dockerfile                    # multi-stage, non-root user
  docker-compose.yml            # 4 services: api:8000 streamlit:8501 mlflow:5000 airflow:8080

.github/workflows/
  drift_pipeline.yml            # CI/CD — runs drift check on push + daily at 02:00 UTC

misc/
  AI_Handoff_Package (3).pdf    # original project spec PDF
  job-logs.txt                  # CI logs from the failing cmdstandpy run (for reference)
  README.md                     # this file
```

---

## Python Environment

- **Interpreter:** `.venv` at `/workspaces/demand-forecasting-drift/.venv`, Python 3.12.1
- **Activate:** `source .venv/bin/activate`
- **Key constraint:** Prophet 1.1.5 is **incompatible with cmdstanpy ≥ 1.3.0** — `stan_backend` attribute was removed. `requirements.txt` pins `cmdstanpy==1.2.4`.

---

## Known Type-Safety Rules (Pylance strict mode)

These patterns MUST be followed in all source files:

| Wrong | Correct | Reason |
|---|---|---|
| `yearly_seasonality=True/False` | `yearly_seasonality='auto'` | Prophet stubs type `str`, not `bool` |
| `series.values - other` | `series.to_numpy(dtype=float) - other` | Avoids numpy dtype ambiguity |
| `mlflow: Any = None` in except | Already done in retrain_pipeline.py | "possibly unbound" suppression |
| `Prophet: Any = None` in except | Already done everywhere | Same |
| `mlflow.set_tracking_uri(f'file://{path}')` | Use `os.path.join(sys.path[0], 'mlruns')` as path in notebooks | Prevents writing to `notebooks/mlruns/` |

---

## Notebook Findings (verified from actual run output)

### 03_drift_detection.ipynb
- Electronics first flagged: **Day 1**, retraining triggered: **Day 3** (Nov 3)
- Health first flagged: **Day 3**, retraining triggered: **Day 5** (Nov 5)
- Health total drift days: **52 / 61** (drift started Aug 1, already severe by Nov 1)

### 04_retraining.ipynb
- **Electronics:** pre-MAE Rs. 8,921 → post-MAE Rs. 6,423 (improvement: **+28.0%**)
- **Health:** pre-MAE Rs. 6,758 → post-MAE Rs. 3,816 (improvement: **+43.5%**)

### End-to-end CI pipeline test (run_drift_check.py --days 61 --retrain, 2026-02-27)

Baseline MAE computed on clean val window (Oct 2–31, 2025):

| Category | Val Baseline MAE | Post-Retrain MAE | Decision |
|---|---|---|---|
| Electronics & Tech | Rs. 2,051 | Rs. 5,918 | ✅ ACCEPTED (+46.4% vs pre) |
| Health & Personal Care | Rs. 4,466 | Rs. 3,935 | ✅ ACCEPTED (+49.9% vs pre) |
| Fashion & Accessories | Rs. 1,592 | Rs. 2,374 | ✅ ACCEPTED (+48.3% vs pre) |
| Home & Lifestyle | Rs. 2,615 | Rs. 5,655 | ✅ ACCEPTED (+14.5% vs pre) |
| Entertainment & Office | Rs. 4,180 | Rs. 3,990 | ❌ REJECTED (retrain worse) |
| Sports & Outdoors | Rs. 1,485 | Rs. 2,866 | ❌ REJECTED (retrain worse) |

4 models saved to `models/`, 2 rejections correctly kept old model.

> Note: Val baseline MAEs above are clean-period (Oct) MAEs. The `pre_retrain_mae` logged to MLflow is computed over the full 61-day eval window and will be higher because it covers the drifted period.

---

## CI/CD Status

**Workflow:** `.github/workflows/drift_pipeline.yml`  
**Schedule:** Daily at **02:00 UTC** (cron `0 2 * * *`)  
**Triggers on push:** `src/**`, `data/processed/**`, `requirements.txt`, `.github/workflows/**`

### Fix history (why things are the way they are)

| Commit | Fix | Why |
|---|---|---|
| `fix: install cmdstan` | Added `cmdstanpy.install_cmdstan()` step | Prophet needs CmdStan at runtime; GitHub runners don't have it |
| `fix: cmdstan cache` | Added `actions/cache@v4` for `~/.cmdstan` | Install takes ~3 min; cache skips it on subsequent runs |
| `fix: data generation step` | Added `python master_data_pipeline.py` step | `data/processed/` is gitignored; CI has no CSVs otherwise |
| `!data/raw/sales_100k.csv` in `.gitignore` | Exception to the data gitignore | The seed raw file must be committed for the pipeline step to work |
| `cmdstanpy==1.2.4` in `requirements.txt` | Pinned cmdstanpy | 1.3.0 removed `stan_backend` attr; Prophet 1.1.5 crashes with `AttributeError` |
| `reports/drift_report*.json` in `.gitignore` | Untrack generated reports | Stale committed report was masking real CI failures in summary step |
| `../reports/figures/` in all notebooks | Fixed savefig paths | Notebooks CWD is `notebooks/` at runtime — bare `reports/` paths wrote to `notebooks/reports/` instead of repo root |

**Current CI status:** ✅ Passing cleanly as of 2026-02-27.

### Model Persistence (added session 3)

Retrained models are saved as `models/{slug}.pkl` using `joblib` so they survive between CI runs.

| File | Category |
|---|---|
| `models/electronics_tech.pkl` | Electronics & Tech |
| `models/health_personal_care.pkl` | Health & Personal Care |
| `models/fashion_accessories.pkl` | Fashion & Accessories |
| `models/home_lifestyle.pkl` | Home & Lifestyle |

`Entertainment & Office` and `Sports & Outdoors` were rejected (retrain performed worse) — their pkl files remain from the previous accepted version if one existed.

GitHub Actions caches `models/` using `actions/cache@v4` with `restore-keys: models-${{ runner.os }}-` so each CI run inherits the previous run's accepted models.

---

## Future Work — Full List & Status

All 7 future work points from the project spec, with implementation status:

| # | Title | Status | Files |
|---|---|---|---|
| # | Title | Status | Files |
|---|---|---|---|
| FW1 | Real Retail Data Integration | ⏭️ Skip | Simulated data is academically valid; live POS feed is infra work, not ML |
| FW2 | Adaptive Drift Threshold (CUSUM/ADWIN) | ⏭️ Skip | 1.5× fixed threshold works correctly and is fully explainable in a viva |
| FW3 | Pipeline Orchestration with Apache Airflow | ✅ Prototyped | `dags/drift_pipeline_dag.py` |
| FW4 | Model Registry with A/B Testing | ✅ Done (registry only) | `mlruns/` + `retrain_pipeline.py` |
| FW5 | Data Quality Gates (Great Expectations) | ✅ Done | `src/data/quality_checks.py` — 8/8 checks passing |
| FW6 | Multi-Model Ensemble / Online Learning Fallback | ⏭️ Skip | Adds complexity without strengthening the MLOps architecture story |
| FW7 | Containerised, Autoscaled Deployment | ✅ Prototyped | `docker/Dockerfile`, `docker/docker-compose.yml` — 4 services |
| FW8 | Real-Time Kafka Simulation | 🔄 In Progress | `kafka/producer.py`, `kafka/consumer.py`, `kafka/docker-compose.kafka.yml` |

### FW1 — Real Retail Data Integration
Connect to a live POS/ERP feed via REST or Kafka for true online learning rather than simulated drift events, validating the framework on non-stationary real-world demand with genuine noise characteristics.

### FW2 — Adaptive Drift Threshold (SKIP)
Replace the fixed 1.5× MAE threshold with a statistically learned threshold (e.g., CUSUM or ADWIN) that adjusts dynamically to each category's natural demand volatility, reducing false positives during seasonal peaks without manual tuning.

**Why we skip it:** The 1.5× threshold was validated empirically on this dataset — Electronics fires on Day 1, Health on Day 3. CUSUM/ADWIN would be a contribution to a research paper, not a viva deliverable.

### FW3 ✅ — Pipeline Orchestration with Apache Airflow
Migrate the walk-forward loop into scheduled Airflow DAGs for fault-tolerant, retry-able execution, enabling the drift-detection → retrain → deploy cycle to run autonomously on a daily schedule.

**Run it:**
```bash
pip install apache-airflow==2.9.0
export AIRFLOW_HOME=$(pwd)/.airflow
airflow db init
airflow dags list   # should show 'demand_forecasting_drift_pipeline'
airflow scheduler &
airflow webserver --port 8080
airflow dags trigger demand_forecasting_drift_pipeline
```

### FW4 ✅ — Model Registry (registry only, no A/B testing)
**What is implemented:**
- **Every** drift-triggered retrain now logs the `prophet_model` artifact (MLmodel, conda.yaml, signature, etc.) — even rejected ones — so every run has a browsable artifact tree in the MLflow UI
- **Accepted** retrains are additionally registered under `demand_{slug}` (e.g., `demand_health_personal_care`) and immediately aliased as `"production"`
- **Rejected** retrains have the artifact but are NOT registered — old production model stays unchanged
- Model versioning is automatic — each accepted retrain creates `v1`, `v2`, etc.
- **Signature** is inferred via `mlflow.models.signature.infer_signature()` from a live Prophet prediction, so `ds` resolves as `datetime` (not `string`) matching Prophet's actual output type
- **Dataset** is logged via `mlflow.log_input()` pointing to `data/processed/final_demand_series.csv` — populates the Dataset column in the MLflow UI

**What is NOT done (intentionally skipped):** Shadow-mode A/B evaluation serving both models in parallel before promotion. Out of scope for this project.

**Model schema (visible in MLflow UI → Models → version → Schema section):**

| Direction | Column | Type |
|---|---|---|
| Input | `ds` | datetime |
| Output | `yhat` | double |
| Output | `yhat_lower` | double |
| Output | `yhat_upper` | double |

**Recent MLflow fixes (session 4):**

| Fix | Commit | Effect |
|---|---|---|
| `infer_signature` replaces manual `Schema/ColSpec` | `dfb8053` | `ds` now correctly inferred as `datetime`; type mismatch that silently dropped artifacts is gone |
| Log artifact for all retrains, not just accepted | `5948166` | All runs now show `prophet_model/` tree in UI |
| `mlflow.log_input()` dataset tracking | previous session | Dataset column populated for all retrain runs |

**Query the registry:**
```python
import mlflow, os
mlflow.set_tracking_uri(f'file://{os.path.abspath("mlruns")}')
client = mlflow.MlflowClient()
for rm in client.search_registered_models():
    v = client.get_model_version_by_alias(rm.name, 'production')
    print(f'{rm.name}  v{v.version}  run_id={v.run_id[:8]}')

# Load production model directly:
model = mlflow.pyfunc.load_model('models:/demand_health_personal_care@production')
```

### FW5 ✅ — Data Quality Gates (Great Expectations)
Add pre-training data validation (schema checks, outlier detection, missing-value thresholds) to prevent degraded retrains caused by upstream data pipeline issues.

**Run it:**
```bash
python src/data/quality_checks.py data/processed/final_demand_series.csv
```

### FW8 🔄 — Real-Time Kafka Simulation (IN PROGRESS)
Stream historical `sales_100k.csv` rows through Kafka at configurable speed to simulate live POS transactions arriving in real time. The consumer applies drift detection on each batch and triggers Prophet retraining when drift is detected — demonstrating the full MLOps loop end-to-end on a live event stream.

**Architecture:**
```
[producer.py]
  reads sales_100k.csv row by row
  → publishes to Kafka topic 'sales_events'
  → configurable replay speed (e.g. 1 row/sec = 1 day/sec of simulated time)

[consumer.py]
  subscribes to 'sales_events'
  → aggregates daily demand per category in a rolling buffer
  → on each new day: calls DriftDetectorRegistry.check()
  → if drift > threshold: calls RetrainPipeline.retrain()
  → logs retrain to MLflow in real time

[MLflow UI]
  refreshes as consumer runs — shows new runs appearing live
```

**Docker stack (kafka/docker-compose.kafka.yml):**
- Zookeeper :2181
- Kafka broker :9092
- Kafka UI (Redpanda Console) :8080 — browse topics/messages visually

**Run it:**
```bash
docker compose -f kafka/docker-compose.kafka.yml up -d
python kafka/producer.py --speed 100   # 100× faster than real time
python kafka/consumer.py               # watch drift detection + retrain in terminal
```

---

### FW6 — Multi-Model Ensemble / Online Learning Fallback (SKIP)
Combine Prophet with an online learning model (River or scikit-multiflow) to handle abrupt drift faster than the current 45-day retrain window allows, with Prophet handling trend/seasonality and the online model adapting within days of a shift.

**Why we skip it:** Prophet already wins the model comparison (02_model_comparison.ipynb). Adding an ensemble doesn't strengthen the MLOps deployment story.

### FW7 ✅ — Containerised Deployment
Dockerise the FastAPI serving layer with multi-stage build and non-root user. Four services: API (8000), Streamlit dashboard (8501), MLflow (5000), Airflow (8080).

**Run it:**
```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## What Still Needs Building (in recommended order)

### ✅ COMPLETED
- `notebooks/01_eda.ipynb` — exploratory data analysis (10/10 cells executed)
- `notebooks/02_model_comparison.ipynb` — Prophet vs SARIMA vs XGBoost (all cells executed, Fig 10 fixed)
- `notebooks/03_drift_detection.ipynb` — drift simulation over 61 days with DriftDetectorRegistry
- `notebooks/04_retraining.ipynb` — retrain on drifted data, MLflow comparison
- `src/data/master_data_pipeline.py` — full data pipeline
- `src/data/quality_checks.py` — Great Expectations (FW5) — 8/8 checks passing
- `src/forecasting/prophet_model.py` — ProphetForecaster + ForecasterRegistry
- `src/drift_detection/drift_detector.py` — DriftDetector + DriftDetectorRegistry (dual-window, 1.5× threshold)
- `src/drift_detection/run_drift_check.py` — headless CI/CD script (uses registry, exits 0/1/2)
- `src/retraining/retrain_pipeline.py` — RetrainPipeline with MLflow tracking + joblib model saving + MLflow Model Registry (FW4)
- `src/inventory/replenishment.py` — InventoryCalculator + InventoryDecision
- `dags/drift_pipeline_dag.py` — Airflow DAG (FW3)
- `docker/Dockerfile` + `docker/docker-compose.yml` — 4-service containerisation (FW7)
- `.github/workflows/drift_pipeline.yml` — CI/CD with model caching
- `models/*.pkl` — 4 accepted models saved locally
- `reports/figures/` — all 20 figures committed to repo root (path fix applied)

---

## MLOps Deployment Story — Are We Good?

The full MLOps loop is implemented:

```
[Kafka producer / GitHub Actions / Airflow schedule]
         ↓
[run_drift_check.py  OR  kafka/consumer.py]
  → loads cached Prophet model (models/*.pkl)
  → walks forward 61 days (batch) OR processes rolling event stream (Kafka)
  → DriftDetectorRegistry flags categories via dual-window MAE
         ↓
[retrain_pipeline.py]
  → retrains Prophet on last 45 days of actual demand
  → A/B gates: only accepts if post_mae < pre_mae
  → logs EVERY retrain run to MLflow (artifact + schema + dataset + metrics)
  → registers accepted model to MLflow Model Registry @ production alias
  → saves accepted model back to models/*.pkl
         ↓
[api/main.py]  ← STILL MISSING
  → serves current model predictions via FastAPI
  → Docker container (port 8000)
         ↓
[dashboard/app.py]  ← STILL MISSING
  → Streamlit UI showing forecasts, drift alerts, inventory
```

**What you can say in a viva today:**
- CI/CD: ✅ GitHub Actions runs drift check daily, triggers retrain automatically
- Experiment tracking: ✅ MLflow logs every retrain — full artifact tree, schema via `infer_signature`, dataset tracking
- Model Registry: ✅ Every accepted retrain registered under `demand_{slug}@production`
- Orchestration: ✅ Airflow DAG prototyped
- Containerisation: ✅ Docker compose with 4 services
- Data quality: ✅ Great Expectations gates
- Model persistence: ✅ joblib + GitHub Actions cache
- Real-time simulation: 🔄 Kafka producer/consumer (in progress)
- Serving layer: ❌ api/main.py not built yet (Docker container has no app to serve)

**Remaining to complete the deployment story:**

### 1. `notebooks/05_inventory.ipynb` ← **DO NEXT (after full test pass)**
- Show inventory decisions (reorder point, safety stock, order quantity) before and after retraining
- Use `InventoryCalculator(service_level=0.95, lead_time_days=7, cycle_days=30)` from `src/inventory/replenishment.py`
- Key metric: rupee value of avoidable stockout/overstock
- Focus on Electronics & Tech (+46.4% MAE improvement) and Health & Personal Care (+49.9%)

### 2. `api/main.py` — FastAPI backend ← **CRITICAL FOR DEPLOYMENT**
This is what the Docker container actually serves. Without it, `docker compose up` starts containers with no running application.

Endpoints to implement:
- `GET /health` — liveness check
- `POST /forecast` — takes `{category, horizon_days}`, returns forecast as JSON
- `GET /drift` — reads latest `reports/drift_report*.json`, returns drift status per category
- `POST /inventory` — takes `{category, service_level, lead_time_days}`, returns reorder recommendations
- `GET /categories` — list available categories

### 3. `dashboard/app.py` — Streamlit dashboard ← NICE TO HAVE
- Day-by-day 2026 forecast playback slider
- Drift alert banner when drift is detected
- Inventory recommendation table
- Charts: forecast vs actual, MAE over time, drift ratio per category

### 4. `kafka/` — Real-Time Kafka Simulation ← AFTER API
See FW8 section above for full architecture and run instructions.

---

## `ProphetForecaster` API (src/forecasting/prophet_model.py)

```python
from src.forecasting.prophet_model import ProphetForecaster, ForecasterRegistry

# Single model
fc = ProphetForecaster(category="Electronics", yearly_seasonality='auto')
fc.fit(df)                               # df must have 'ds' and 'y' columns
forecast = fc.predict(horizon=30)         # returns DataFrame with ds, yhat, yhat_lower, yhat_upper
mae, rmse = fc.evaluate(test_df)

# Multi-category registry
registry = ForecasterRegistry()
registry.fit_all(data_dict)              # {category: df}
forecasts = registry.predict_all(horizon=30)
```

## `InventoryCalculator` API (src/inventory/replenishment.py)

```python
from src.inventory.replenishment import InventoryCalculator

calc = InventoryCalculator(service_level=0.95, lead_time_days=7, cycle_days=30)
result = calc.compute(category="Electronics & Tech", forecast_df=forecast_df)
# result is an InventoryDecision dataclass:
# result.reorder_point, result.safety_stock, result.order_quantity,
# result.avg_daily_demand, result.forecast_std, result.to_dict()
```

---

## MLflow Tracking

- Runs are stored in `mlruns/` at repo root (not `notebooks/mlruns/`)
- In notebooks, set: `sys.path[0] = os.path.abspath('..')` then `mlflow.set_tracking_uri(f'file://{os.path.join(sys.path[0], "mlruns")}')`
- View UI: `mlflow ui --backend-store-uri mlruns/`
