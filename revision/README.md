# Revision experiments (JCMM_1268)

This folder and `data/revision/` hold the code and outputs used for the results
in the revised manuscript. The archived files in `data/processed/` are left
unchanged; they are the outputs of commit `f264989`.

## Environment used

| Item | Version |
|---|---|
| Python | 3.11.17 |
| prophet | 1.1.7 (bundled CmdStan 2.33.1) |
| cmdstanpy | 1.3.0 |
| pandas / numpy | 2.2.3 / 1.26.4 |
| mlflow / sqlalchemy | 2.15.1 / 2.0.29 |
| OS / hardware | Ubuntu 24.04, Linux x86_64, 4 vCPU Intel Xeon 2.1 GHz, 15 GB RAM |

Install: `pip install -r requirements.txt -r requirements-prophet.txt tqdm pytest`.

cmdstanpy 1.3.0 rejects Prophet 1.1.7's bundled CmdStan because it has no `makefile`. Create an empty file to satisfy the check:
`touch <site-packages>/prophet/stan_model/cmdstan-2.33.1/makefile`.
Then set `USE_BUNDLED_CMDSTAN=1` so that `pipeline.py` does not download and compile CmdStan into `~/.cmdstan`.

## Input

`data/processed/daily_demand.csv` (SHA-256 `06281acf21df2d3cfab4e2ef195d065972f947594f6e9d18a041f43cbf71bbf8`).
It is identical in every commit of the repository.

Preprocessing is **not** re-run, for this reason:
- Seed 42 governs only the drift injection.
- Region assignment, quantity synthesis and the 2025 extension use unseeded draws.

## Commands (run from the repository root)

```bash
export USE_BUNDLED_CMDSTAN=1

# 0. Tests
pytest -q tests/

# 1. Leakage-free rolling-origin forecast evaluation (Tables 6-8, Figure 2)
python revision/rolling_origin_eval.py --eval_start 2025-07-03 --eval_end 2025-07-31 \
    --history_start 2024-01-01 --workers 4 --out data/revision/rolling_origin_jul2025

# 2. Seeded opening stock (generate_inventory.py uses np.random.seed(42))
python -c "from src.generate_inventory import generate_inventory; generate_inventory(output_path='data/revision/inventory_master_seed42.csv')"

# 3. Drift-detection and safety-gate run covering the four injection windows
#    (run from an empty scratch directory: MLflow and model files are written to the working directory)
DATABASE_URL=sqlite:///$PWD/scratch_A.db python pipeline.py --start 2024-02-01 --end 2025-03-31 \
    --history_start 2024-01-01 --seed 42 --drift_threshold 2.0 --cooldown_days 7 --holdout_days 7 \
    --inventory_file data/revision/inventory_master_seed42.csv \
    --output_dir data/revision/pipeline_drift_2024-02-01_2025-03-31
python revision/drift_window_eval.py --run_dir data/revision/pipeline_drift_2024-02-01_2025-03-31

# 4. Operational July run (Tables 9-10, Figure 3, 2,088 inventory assessments)
DATABASE_URL=sqlite:///$PWD/scratch_B.db python pipeline.py --start 2025-07-03 --end 2025-07-31 \
    --history_start 2024-01-01 --seed 42 --drift_threshold 2.0 --cooldown_days 7 --holdout_days 7 \
    --inventory_file data/revision/inventory_master_seed42.csv \
    --output_dir data/revision/pipeline_july_2025-07-03_2025-07-31

# 5. Figures 2 and 3
python revision/make_figures.py
```

`DATABASE_URL` must point to a new, empty SQLite file. The committed `inventory.db` holds state from earlier runs, and `pipeline.py` initialises stock from `--inventory_file` only when the database is empty.

## Definitions

* **Forecast evaluation.** For each SKU and forecast date d in 2025-07-03..2025-07-31:
  * Prophet (`pipeline.train_prophet`) is fitted on that SKU's observations from 2024-01-01 to d−1 and predicts d.
  * The baselines are scored on the same (SKU, d) pairs: naive y(d−1), seasonal-naive y(d−7), and the historical mean over the same training data.
  * MAPE and Tol20 use nonzero actual demand.
* **Drift evaluation.** The definitions are fixed in `preregistered_drift_eval.json`, which was written before any detector run.
* **Safety gate.** Implemented in `pipeline.py::mape_safety_gate`:
  * H is the 7 observed days before the trigger.
  * The candidate is fitted on data before H.
  * Both models are scored by MAPE on the nonzero days of H.
  * The candidate is accepted only if its MAPE is strictly lower, and the evaluated candidate is deployed.
  * Each decision is written to `gate_decisions.csv`.

## Reproducibility checks performed

* Rolling-origin evaluation re-run: `predictions.csv` identical.
* July pipeline run re-run with seed 42: all eight output files byte-identical.
