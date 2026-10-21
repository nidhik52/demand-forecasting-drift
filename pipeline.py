import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import pickle
import random
import os
import warnings
from collections import deque
from tqdm import tqdm
from datetime import timedelta, datetime
from typing import Dict, Optional, Any, List, cast

warnings.filterwarnings("ignore")

import mlflow

MODEL_TYPE = os.getenv("MODEL_TYPE", "prophet")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

from sqlalchemy.orm import Session
from src.db import SessionLocal, Inventory, Order

# ----------------------------
# CmdStan setup
# ----------------------------
_CMDSTAN_SETUP_DONE = False

def setup_cmdstan_once():
    global _CMDSTAN_SETUP_DONE
    if _CMDSTAN_SETUP_DONE:
        return
    # REVISION CHANGE: opt-out that keeps Prophet's bundled CmdStan instead of
    # downloading and compiling CmdStan 2.33.1 into ~/.cmdstan.
    if os.getenv("USE_BUNDLED_CMDSTAN") == "1":
        _CMDSTAN_SETUP_DONE = True
        return
    try:
        import cmdstanpy
        os.environ.setdefault("CXXFLAGS", "-O0 -g0 -std=c++14")
        os.environ.setdefault("MAKEFLAGS", "-j2")
        cmdstan_dir = Path.home() / ".cmdstan"
        cmdstan_path = cmdstan_dir / "cmdstan-2.33.1"
        if not cmdstan_path.exists():
            cmdstanpy.install_cmdstan(version="2.33.1", dir=str(cmdstan_dir), overwrite=False)
        cmdstanpy.set_cmdstan_path(str(cmdstan_path))
        _CMDSTAN_SETUP_DONE = True
    except Exception as e:
        raise RuntimeError(f"CmdStan setup failed: {e}")

# CmdStan is set up lazily inside run_pipeline() to avoid import-time crashes

# ----------------------------
# Model functions (unchanged)
# ----------------------------
def train_baseline(df: pd.DataFrame) -> float:
    return float(df["Demand"].mean())

def predict_baseline(model: float, df: pd.DataFrame) -> np.ndarray:
    return np.full(len(df), model)

def train_prophet(df: pd.DataFrame):
    if Prophet is None:
        raise RuntimeError("Prophet not available.")
    prophet_df = df.rename(columns={"Date": "ds", "Demand": "y"})[["ds", "y"]]
    model = Prophet(
        daily_seasonality="auto",
        weekly_seasonality="auto",
        yearly_seasonality="auto",
        stan_backend="CMDSTANPY"
    )
    model.fit(prophet_df)
    return model

def predict_prophet(model, df: pd.DataFrame) -> np.ndarray:
    future = pd.DataFrame({"ds": pd.to_datetime(df["Date"])})
    forecast = model.predict(future)
    pred = forecast["yhat"].astype(float).to_numpy()
    return np.clip(pred, 0.0, None)

def calculate_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))

def calculate_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    nonzero = actual != 0
    if not np.any(nonzero):
        return 0.0
    return float(np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100)

def calculate_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))

def data_quality_check(df: pd.DataFrame, sku: str, date: pd.Timestamp) -> Dict:
    issues = []
    if df["Demand"].isnull().any():
        issues.append("missing_demand")
    if (df["Demand"] < 0).any():
        issues.append("negative_demand")
    mean = df["Demand"].mean()
    std = df["Demand"].std()
    if std > 0 and (df["Demand"] > mean + 3 * std).any():
        issues.append("outlier_demand")
    return {"sku": sku, "date": date, "issues": issues, "pass": len(issues) == 0}

# ----------------------------
# Data loading
# ----------------------------
def load_data() -> pd.DataFrame:
    from src.config import DAILY_DEMAND_FILE
    df = pd.read_csv(DAILY_DEMAND_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def ensure_data_range(start: str, end: str) -> None:
    from src.config import DAILY_DEMAND_FILE
    from src.preprocessing import load_data as load_raw_data, preprocess_data, save_processed_data
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if not DAILY_DEMAND_FILE.exists():
        raw_df = load_raw_data()
        demand = preprocess_data(raw_df)
        save_processed_data(demand)
    else:
        existing = pd.read_csv(DAILY_DEMAND_FILE)
        existing["Date"] = pd.to_datetime(existing["Date"])
        min_date, max_date = existing["Date"].min(), existing["Date"].max()
        if start_dt < min_date or end_dt > max_date:
            raw_df = load_raw_data()
            demand = preprocess_data(raw_df)
            save_processed_data(demand)

# ----------------------------
# Event logger
# ----------------------------
try:
    from src.event_logger import log_event as _log_event
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

def log_event(event_type: str, message: str, sim_date: pd.Timestamp):
    """
    Log event using the SIMULATION date (not datetime.now()).
    Adds a realistic hour offset so timestamps aren't 00:00:00.
    """
    hour   = random.randint(8, 18)
    minute = random.randint(0, 59)
    event_time = sim_date + timedelta(hours=hour, minutes=minute)

    if LOGGER_AVAILABLE:
        _log_event(event_type, message, event_time)
    else:
        print(f"[{event_time}] {event_type} → {message}")

# ----------------------------
# Rolling MAE drift detector
# FIX: replaces random.random() with real MAE-ratio logic
# ----------------------------
class RollingDriftDetector:
    """
    Per-SKU rolling MAE drift detector.
    Flags drift when recent MAE > threshold * baseline MAE.
    Cooldown prevents re-triggering immediately after retrain.
    """
    def __init__(self, threshold: float = 2.0, window: int = 7, min_days: int = 3, cooldown_days: int = 7):
        self.threshold    = threshold
        self.window       = window
        self.min_days     = min_days
        self.cooldown_days = cooldown_days

        self._errors:       Dict[str, deque] = {}
        self._baselines:    Dict[str, float] = {}
        self._consec:       Dict[str, int]   = {}
        self._last_retrain: Dict[str, Optional[pd.Timestamp]] = {}

    def update(self, sku: str, actual: float, predicted: float,
               current_date: pd.Timestamp) -> Dict:
        """
        Feed one day's error. Returns drift status dict.
        """
        if sku not in self._errors:
            self._errors[sku]       = deque(maxlen=self.window)
            self._baselines[sku]    = None
            self._consec[sku]       = 0
            self._last_retrain[sku] = None

        err = abs(actual - predicted)
        self._errors[sku].append(err)

        # Need at least window days before we have a baseline
        if len(self._errors[sku]) < self.window:
            return {"drift": False, "retrain": False, "ratio": 0.0, "rolling_mae": err}

        rolling_mae = float(np.mean(self._errors[sku]))

        # Set baseline once from first full window, then freeze it
        # (reset when retraining happens)
        if self._baselines[sku] is None:
            self._baselines[sku] = rolling_mae
            return {"drift": False, "retrain": False, "ratio": 1.0, "rolling_mae": rolling_mae}

        baseline = self._baselines[sku]
        ratio    = rolling_mae / baseline if baseline > 0 else 1.0

        flagged = ratio > self.threshold
        if flagged:
            self._consec[sku] += 1
        else:
            self._consec[sku] = 0

        drift_detected = self._consec[sku] >= self.min_days

        # Cooldown check
        in_cooldown = False
        if self._last_retrain[sku] is not None:
            days_since = (current_date - self._last_retrain[sku]).days
            in_cooldown = days_since < self.cooldown_days

        retrain_trigger = drift_detected and not in_cooldown

        return {
            "drift":       drift_detected,
            "retrain":     retrain_trigger,
            "ratio":       round(ratio, 3),
            "rolling_mae": round(rolling_mae, 3),
            "in_cooldown": in_cooldown,
        }

    def record_retrain(self, sku: str, current_date: pd.Timestamp, new_baseline: Optional[float] = None):
        """Call after successful retrain. Resets cooldown and baseline."""
        self._last_retrain[sku] = current_date
        self._consec[sku]       = 0
        self._errors[sku].clear()
        if new_baseline is not None:
            self._baselines[sku] = new_baseline
        else:
            self._baselines[sku] = None  # will recompute after window fills

    # REVISION CHANGE: a rejected candidate starts the cooldown but keeps the
    # incumbent's baseline and error queue, so persistent drift is re-evaluated
    # after the cooldown instead of on every following day.
    def record_rejection(self, sku: str, current_date: pd.Timestamp):
        self._last_retrain[sku] = current_date
        self._consec[sku]       = 0

# ----------------------------
# REVISION CHANGE: held-out MAPE safety gate (manuscript Eq. 4, Algorithm 3).
# H = the last `holdout_days` observed days before the trigger date; the
# candidate is fitted only on data before H and compared with the incumbent on
# the nonzero-demand days of H. Replacement happens only if the candidate's
# MAPE is strictly lower; the evaluated candidate is the one deployed.
# ----------------------------
def mape_safety_gate(train_df: pd.DataFrame, incumbent, incumbent_train_end: pd.Timestamp,
                     fit_fn, predict_fn, holdout_days: int = 7) -> Dict[str, Any]:
    train_df = train_df.sort_values("Date")
    holdout  = train_df.iloc[-holdout_days:]
    fit_part = train_df.iloc[:-holdout_days]
    decision: Dict[str, Any] = {
        "accepted": False, "candidate": None, "mape_candidate": None, "mape_incumbent": None,
        "n_holdout": len(holdout), "n_holdout_nonzero": 0,
        "holdout_start": holdout["Date"].min() if len(holdout) else None,
        "holdout_end": holdout["Date"].max() if len(holdout) else None,
        "candidate_train_end": fit_part["Date"].max() if len(fit_part) else None,
        "incumbent_train_end": incumbent_train_end, "reason": "",
    }
    if len(holdout) < holdout_days or len(fit_part) < 3:
        decision["reason"] = "insufficient_history"
        return decision
    # Leakage guard: neither model may have been fitted on any day of H.
    if incumbent_train_end is not None and incumbent_train_end >= decision["holdout_start"]:
        decision["reason"] = "incumbent_overlaps_holdout"
        return decision
    actual  = holdout["Demand"].to_numpy(dtype=float)
    nonzero = actual > 0
    decision["n_holdout_nonzero"] = int(nonzero.sum())
    if not nonzero.any():
        # Zero denominator |H+| = 0: no valid comparison, so the incumbent is kept.
        decision["reason"] = "no_nonzero_holdout_days"
        return decision
    candidate = fit_fn(fit_part)
    mape_c = calculate_mape(actual[nonzero], predict_fn(candidate, holdout)[nonzero])
    mape_i = calculate_mape(actual[nonzero], predict_fn(incumbent, holdout)[nonzero])
    decision.update(candidate=candidate, mape_candidate=mape_c, mape_incumbent=mape_i,
                    accepted=bool(mape_c < mape_i),
                    reason="candidate_lower_mape" if mape_c < mape_i else "candidate_not_lower")
    return decision

# ----------------------------
# Inventory helpers (unchanged)
# ----------------------------
def process_in_transit_orders(current_date: pd.Timestamp, session: Session):
    orders = session.query(Order).filter(Order.restock_date <= current_date, Order.received == 0).all()
    for order in orders:
        inv = session.query(Inventory).filter_by(sku=order.sku).first()
        if inv:
            inv.current_stock += order.order_qty   # type: ignore
            inv.in_transit    -= order.order_qty   # type: ignore
            order.received     = 1                 # type: ignore
    session.commit()

def update_inventory_after_demand(sku: str, demand_quantity: int, session: Session) -> int:
    inv = session.query(Inventory).filter_by(sku=sku).first()
    if not inv:
        inv = Inventory(sku=sku, current_stock=100, lead_time_days=7, safety_stock=10)
        session.add(inv)
        session.commit()
        session.refresh(inv)
    def safe_int(val, default=0):
        if isinstance(val, bytes):
            try:
                return int.from_bytes(val, 'little')
            except Exception:
                return default
        try:
            return int(float(val))
        except Exception:
            return default
    current_stock = safe_int(inv.current_stock, 0)
    new_stock     = max(0, current_stock - demand_quantity)
    # --- Simple replenishment: if stock falls below 20, restock to a random level (simulate delivery) ---
    if new_stock < 20:
        restock = np.random.choice([0, 30, 60, 120], p=[0.2, 0.4, 0.3, 0.1])
        new_stock += restock
    # REVISION CHANGE: store a Python int; numpy.int64 was written to SQLite as
    # an 8-byte blob, which corrupted Current_Stock in inventory_master.csv.
    new_stock = int(new_stock)
    inv.current_stock = new_stock          # type: ignore
    inv.last_updated  = datetime.utcnow() # type: ignore
    session.commit()
    return new_stock

def get_inventory_data(sku: str, session: Session) -> Dict:
    inv = session.query(Inventory).filter_by(sku=sku).first()
    def safe_int(val, default=0):
        if isinstance(val, bytes):
            try:
                return int.from_bytes(val, 'little')
            except Exception:
                return default
        try:
            return int(float(val))
        except Exception:
            return default
    if not inv:
        return {"current_stock": 100, "lead_time_days": 7, "safety_stock": 10, "in_transit": 0}
    return {
        "current_stock": safe_int(inv.current_stock),    # type: ignore
        "lead_time_days": safe_int(inv.lead_time_days, 7),  # type: ignore
        "safety_stock": safe_int(inv.safety_stock, 10),      # type: ignore
        "in_transit": safe_int(inv.in_transit, 0)           # type: ignore
    }

# ----------------------------
# Main pipeline
# FIX: drift uses RollingDriftDetector not random
# FIX: event timestamps use sim date not datetime.now()
# FIX: MLflow uses log_model correctly for Prophet
# ----------------------------
def run_pipeline(start: str, end: str, run_id: str = "manual",
                 drift_threshold: float = 2.0, cooldown_days: int = 7,
                 history_start: Optional[str] = None, seed: Optional[int] = None,
                 output_dir: Optional[str] = None, inventory_file: Optional[str] = None,
                 safety_gate: bool = True, holdout_days: int = 7) -> None:
    print(f"\n🚀 Pipeline | {start} → {end} | Threshold={drift_threshold}x | Cooldown={cooldown_days}d")
    # REVISION CHANGE: optional seeding of the pipeline's own random draws
    # (simulated restocks, event/model timestamps). Unseeded when seed is None.
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    # REVISION CHANGE: optional separate output directory so that revision runs
    # never overwrite or delete the archived files in data/processed/.
    out_dir = Path(output_dir) if output_dir else Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_dir:
        import src.event_logger as _event_logger
        _event_logger.EVENT_LOG_FILE = out_dir / "system_events.csv"
    gate_log = []
    # FIX 7: set up CmdStan here, not at import time — avoids crash on import
    if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
        try:
            setup_cmdstan_once()
        except Exception as e:
            print(f"⚠️  CmdStan setup failed: {e}. Falling back to baseline model.")
            import os
            os.environ["MODEL_TYPE"] = "baseline"
    ensure_data_range(start, end)
    df = load_data()
    # REVISION CHANGE: by default models only ever see dates from `start`
    # (2 days of history on the first day). With history_start, models train on
    # all observations before each simulated date, while the daily loop still
    # covers only [start, end].
    df = df[(df["Date"] >= (history_start or start)) & (df["Date"] <= end)]
    loop_dates = sorted(df.loc[df["Date"] >= start, "Date"].unique())

    detector = RollingDriftDetector(
        threshold=drift_threshold,
        window=7,
        min_days=3,
        cooldown_days=cooldown_days,
    )

    sku_model   = {}
    sku_train_end: Dict[str, pd.Timestamp] = {}  # REVISION CHANGE: last training date per model
    results     = []
    inv_results = []
    quality_log = []

    # Always clear event log at start so timestamps match this run's sim dates
    from src.config import EVENT_LOG_FILE
    # Also clear the hardcoded path in event_logger (same file, different reference)
    hardcoded_log = Path("data/processed/system_events.csv")
    # REVISION CHANGE: with output_dir only that run's own log is cleared.
    log_paths = [out_dir / "system_events.csv"] if output_dir else [EVENT_LOG_FILE, hardcoded_log]
    for log_path in set(log_paths):
        if log_path.exists():
            log_path.unlink()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment_name = "demand_forecasting_drift"
    artifact_location = str(Path("./mlruns").absolute())
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    else:
        # If artifact location is not as desired, warn but do not delete (DB mode does not allow delete by name)
        if not exp.artifact_location.startswith("file://" + str(Path("./mlruns").absolute())) and not exp.artifact_location.endswith("/mlruns"):
            print(f"WARNING: MLflow experiment '{experiment_name}' exists with artifact location {exp.artifact_location}.\nArtifacts may not be stored in the desired location. To change, delete the experiment manually from the MLflow UI or DB.")
    mlflow.set_experiment(experiment_name)


    session = SessionLocal()
    # --- Initialize DB from inventory_master.csv if DB is empty ---
    if session.query(Inventory).count() == 0:
        from src.config import INVENTORY_FILE
        def safe_int(val, default=0):
            if isinstance(val, bytes):
                try:
                    return int.from_bytes(val, 'little')
                except Exception:
                    return default
            try:
                return int(float(val))
            except Exception:
                return default
        init_file = Path(inventory_file) if inventory_file else INVENTORY_FILE  # REVISION CHANGE
        if init_file.exists():
            inv_df = pd.read_csv(init_file)
            for _, row in inv_df.iterrows():
                inv = Inventory(
                    sku=row["SKU"],
                    current_stock=safe_int(row["Current_Stock"]),
                    in_transit=safe_int(row.get("In_Transit", 0)),
                    lead_time_days=safe_int(row.get("Lead_Time_Days", 7)),
                    safety_stock=safe_int(row.get("Safety_Stock", 10)),
                )
                session.add(inv)
            session.commit()
            print("Initialized inventory DB from inventory_master.csv")

    for current_date in tqdm(loop_dates, desc="Processing days"):
        current_date = pd.Timestamp(current_date)
        process_in_transit_orders(current_date, session)

        day_df = df[df["Date"] == current_date]
        for sku in day_df["SKU"].unique():
            hist = df[(df["SKU"] == sku) & (df["Date"] <= current_date)].sort_values("Date")
            if len(hist) < 3:
                continue

            train_df = hist.iloc[:-1]
            test_df  = hist.iloc[-1:]

            quality = data_quality_check(train_df, sku, current_date)
            quality_log.append(quality)

            # Train initial model
            if sku not in sku_model:
                if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                    model = train_prophet(train_df)
                else:
                    model = train_baseline(train_df)
                sku_model[sku] = model
                sku_train_end[sku] = train_df["Date"].max()

            model = sku_model[sku]
            if MODEL_TYPE == "prophet" and PROPHET_AVAILABLE:
                preds = predict_prophet(model, test_df)
            else:
                preds = predict_baseline(model, test_df)

            actual = test_df["Demand"].values
            mae    = calculate_mae(actual, preds)
            mape   = calculate_mape(actual, preds)
            rmse   = calculate_rmse(actual, preds)

            # FIX: use rolling MAE ratio, not random
            drift_status     = detector.update(sku, float(actual[0]), float(preds[0]), current_date)
            drift_detected   = drift_status["drift"]
            retrain_happened = False

            if drift_detected:
                # FIX: event timestamp = simulation date (not now())
                log_event("DRIFT",
                          f"{sku} drift detected (ratio={drift_status['ratio']:.2f}x, rolling_MAE={drift_status['rolling_mae']:.2f})",
                          current_date)

                use_prophet = MODEL_TYPE == "prophet" and PROPHET_AVAILABLE
                fit_fn      = train_prophet if use_prophet else train_baseline
                predict_fn  = predict_prophet if use_prophet else predict_baseline
                gate = None
                if drift_status["retrain"] and safety_gate:
                    # REVISION CHANGE: candidate must pass the held-out MAPE gate.
                    gate = mape_safety_gate(train_df, model, sku_train_end.get(sku),
                                            fit_fn, predict_fn, holdout_days)
                    gate_log.append({
                        "Trigger_Date": current_date, "SKU": sku,
                        "Decision": "ACCEPTED" if gate["accepted"] else "REJECTED",
                        "Reason": gate["reason"],
                        "MAPE_Candidate": gate["mape_candidate"], "MAPE_Incumbent": gate["mape_incumbent"],
                        "Holdout_Start": gate["holdout_start"], "Holdout_End": gate["holdout_end"],
                        "N_Holdout": gate["n_holdout"], "N_Holdout_Nonzero": gate["n_holdout_nonzero"],
                        "Candidate_Train_End": gate["candidate_train_end"],
                        "Incumbent_Train_End": gate["incumbent_train_end"],
                        "Drift_Ratio": drift_status["ratio"],
                    })
                    if not gate["accepted"]:
                        detector.record_rejection(sku, current_date)
                        _mc = "NA" if gate["mape_candidate"] is None else f"{gate['mape_candidate']:.2f}"
                        _mi = "NA" if gate["mape_incumbent"] is None else f"{gate['mape_incumbent']:.2f}"
                        log_event("RETRAIN_REJECTED",
                                  f"{sku} candidate rejected (MAPE_candidate={_mc} MAPE_incumbent={_mi}, {gate['reason']})",
                                  current_date)

                if drift_status["retrain"] and (gate is None or gate["accepted"]):
                    print(f"  🔄 {sku}: RETRAIN on {current_date.date()} (ratio={drift_status['ratio']:.2f}x)")
                    if gate is not None:
                        new_model = gate["candidate"]  # REVISION CHANGE: deploy the evaluated candidate
                        sku_train_end[sku] = gate["candidate_train_end"]
                    else:
                        new_model = fit_fn(train_df)
                        sku_train_end[sku] = train_df["Date"].max()

                    sku_model[sku] = new_model
                    retrain_happened = True

                    # FIX: save model with sim date timestamp + random hour:minute
                    # current_date is a date-only Timestamp (time = 00:00:00).
                    # Adding a random offset mirrors what log_event() does, so
                    # model filenames reflect a realistic intra-day time rather
                    # than always ending in _000000.

                    # Use a relative, writable path for model artifacts (CI/CD safe)
                    model_dir = Path("./models") / ("prophet" if MODEL_TYPE == "prophet" else "baseline")
                    model_dir.mkdir(parents=True, exist_ok=True)
                    _save_hour   = random.randint(8, 18)
                    _save_minute = random.randint(0, 59)
                    _save_ts     = current_date + pd.Timedelta(hours=_save_hour, minutes=_save_minute)
                    ts_str = _save_ts.strftime("%Y%m%d_%H%M%S")
                    model_path = model_dir / f"{sku}_{ts_str}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(new_model, f)

                    # Log artifact using a relative path
                    with mlflow.start_run(run_name=f"{sku}_{current_date.strftime('%Y%m%d')}"):
                        mlflow.log_param("sku", sku)
                        mlflow.log_param("model_type", MODEL_TYPE)
                        mlflow.log_param("retrain_date", str(current_date.date()))
                        mlflow.log_metric("mae", mae)
                        mlflow.log_metric("mape", mape)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("drift_ratio", drift_status["ratio"])
                        mlflow.log_artifact(str(model_path))
                        # FIX 6: removed broken mlflow.pyfunc.log_model call
                        # Prophet is not a pyfunc-compatible model; log_artifact above is sufficient

                    # New baseline = post-retrain MAE on same day
                    new_preds   = predict_prophet(new_model, test_df) if MODEL_TYPE == "prophet" else predict_baseline(new_model, test_df)
                    new_mae     = calculate_mae(actual, new_preds)
                    detector.record_retrain(sku, current_date, new_baseline=new_mae)

                    if gate is not None:
                        log_event("RETRAIN_ACCEPTED",
                                  f"{sku} candidate accepted (MAPE_candidate={gate['mape_candidate']:.2f} "
                                  f"MAPE_incumbent={gate['mape_incumbent']:.2f}; pre_MAE={mae:.2f} post_MAE={new_mae:.2f})",
                                  current_date)
                    else:
                        log_event("RETRAIN",
                                  f"{sku} retrained (pre_MAE={mae:.2f} post_MAE={new_mae:.2f})",
                                  current_date)
                elif not drift_status["retrain"]:  # REVISION CHANGE: rejections are not cooldowns
                    print(f"  ⏭️ {sku}: drift in cooldown on {current_date.date()}")
                    log_event("COOLDOWN",
                              f"{sku} drift detected but in cooldown period",
                              current_date)
            else:
                pass  # no log for stable (avoids log explosion)

            demand_today = int(test_df["Demand"].iloc[0])
            new_stock    = update_inventory_after_demand(sku, demand_today, session)
            inv_data     = get_inventory_data(sku, session)
            lead_time    = inv_data["lead_time_days"]
            safety_stock = inv_data["safety_stock"]
            forecast_demand = float(preds[0])

            reorder_point = forecast_demand * lead_time + safety_stock
            reorder_qty   = max(0, int(reorder_point - new_stock))
            risk = "CRITICAL" if reorder_qty > 100 else "WARNING" if reorder_qty > 0 else "SAFE"

            inv_results.append({
                "Date": current_date,
                "SKU": sku,
                "Current_Stock": new_stock,
                "In_Transit": inv_data["in_transit"],
                "Recommended_Order_Qty": reorder_qty,
                "Risk_Level": risk,
                "Lead_Time_Days": lead_time,
                "Safety_Stock": safety_stock,
            })

            results.append({
                "Date":      test_df["Date"].iloc[0],
                "SKU":       sku,
                "Actual":    demand_today,
                "Predicted": round(forecast_demand, 4),
                "MAE":       round(mae, 4),
                "MAPE":      round(mape, 4),
                "RMSE":      round(rmse, 4),
                "Drift":     int(drift_detected),
                "Retrained": int(retrain_happened),
            })


    # --- Sync inventory_master.csv with DB ---
    from src.config import INVENTORY_FILE
    inventory_db = session.query(Inventory).all()
    inv_rows = []
    for inv in inventory_db:
        inv_rows.append({
            "SKU": inv.sku,
            "Current_Stock": inv.current_stock,
            "In_Transit": inv.in_transit,
            "Lead_Time_Days": inv.lead_time_days,
            "Safety_Stock": inv.safety_stock,
            "Stock_As_Of_Date": inv.last_updated.strftime("%Y-%m-%d") if inv.last_updated else ""
        })
    inv_df = pd.DataFrame(inv_rows)
    # REVISION CHANGE: with output_dir, the export and the two end-of-run
    # artifacts below are written to that directory instead of data/processed/.
    inv_export = out_dir / "inventory_master.csv" if output_dir else INVENTORY_FILE
    inv_export.parent.mkdir(parents=True, exist_ok=True)
    inv_df.to_csv(inv_export, index=False)
    print(f"Inventory DB exported to {inv_export}")
    if output_dir:
        import src.forecasting as _forecasting
        import src.inventory as _inventory
        _forecasting.FORECAST_FILE = out_dir / "forecast_2025.csv"
        _forecasting.MODELS_DIR = Path("./models_forecasting")
        _inventory.FORECAST_FILE = out_dir / "forecast_2025.csv"
        _inventory.INVENTORY_FILE = inv_export

    # FIX 8: Generate forecast_2025.csv FIRST so inventory recommendations use fresh data
    try:
        from src.forecasting import run_forecasting as _run_forecasting
        print("\n🔮 Generating forecast_2025.csv...")
        _run_forecasting(df, silent=True)
        print("✅ forecast_2025.csv generated.")
    except Exception as e:
        print(f"⚠️  forecast_2025.csv generation failed: {e}. Inventory may use stale data.")

    # --- Regenerate inventory_recommendations.csv ---
    try:
        from src.inventory import load_data as load_inv_data, generate_inventory_recommendations, save_inventory
        forecast, inventory = load_inv_data()
        # Use the last date in the forecast as the reference date for recommendations
        if not forecast.empty:
            date_col = 'Date' if 'Date' in forecast.columns else ('date' if 'date' in forecast.columns else None)
            if date_col:
                current_date = pd.to_datetime(forecast[date_col]).max()
            else:
                current_date = pd.Timestamp.now()
        else:
            current_date = pd.Timestamp.now()
        recs_df = generate_inventory_recommendations(forecast, inventory, current_date)
        # FIX 3: save to a separate forecast-based file — don't overwrite the
        # per-day inv_results CSV which is the primary source for the dashboard
        from src.config import PROCESSED_DIR
        forecast_recs_path = (out_dir if output_dir else PROCESSED_DIR) / "inventory_recommendations_forecast.csv"
        recs_df.to_csv(forecast_recs_path, index=False)
        print(f"Inventory forecast recommendations saved to inventory_recommendations_forecast.csv")
        print(f"Primary inventory_recommendations.csv (per-day) preserved.")
    except Exception as e:
        print(f"Failed to regenerate inventory recommendations: {e}")

    session.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(inv_results).to_csv(out_dir / "inventory_recommendations.csv", index=False)
    pd.DataFrame(quality_log).to_csv(out_dir / "data_quality.csv", index=False)
    if safety_gate:
        # REVISION CHANGE: one row per gate decision (accepted and rejected).
        pd.DataFrame(gate_log, columns=[
            "Trigger_Date", "SKU", "Decision", "Reason", "MAPE_Candidate", "MAPE_Incumbent",
            "Holdout_Start", "Holdout_End", "N_Holdout", "N_Holdout_Nonzero",
            "Candidate_Train_End", "Incumbent_Train_End", "Drift_Ratio",
        ]).to_csv(out_dir / "gate_decisions.csv", index=False)

    # FIX: always write system_events.csv even if empty (CI/CD validate step needs it)
    event_log_path = out_dir / "system_events.csv"
    if not event_log_path.exists():
        pd.DataFrame(columns=["timestamp", "event_type", "message"]).to_csv(event_log_path, index=False)
        print("  ℹ  No drift events — empty system_events.csv written")

    print(f"\n✅ Pipeline done. {len(results)} predictions written.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",          required=True)
    parser.add_argument("--end",            required=True)
    parser.add_argument("--run_id",         default="manual")
    parser.add_argument("--drift_threshold",type=float, default=2.0)
    parser.add_argument("--cooldown_days",  type=int, default=7)
    # REVISION CHANGE: options added for the JCMM_1268 revision experiments
    parser.add_argument("--history_start",  default=None, help="first date models may train on (default: --start)")
    parser.add_argument("--seed",           type=int, default=None)
    parser.add_argument("--output_dir",     default=None, help="write all outputs here instead of data/processed")
    parser.add_argument("--inventory_file", default=None, help="initial inventory CSV (default: config INVENTORY_FILE)")
    parser.add_argument("--no_safety_gate", action="store_true", help="replace the model at every trigger (pre-revision behaviour)")
    parser.add_argument("--holdout_days",   type=int, default=7)
    args = parser.parse_args()
    run_pipeline(args.start, args.end, args.run_id, args.drift_threshold, args.cooldown_days,
                 history_start=args.history_start, seed=args.seed, output_dir=args.output_dir,
                 inventory_file=args.inventory_file, safety_gate=not args.no_safety_gate,
                 holdout_days=args.holdout_days)