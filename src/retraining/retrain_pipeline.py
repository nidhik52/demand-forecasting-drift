"""
Retraining Pipeline — Auto-Retrain Prophet on Drift Detection
=============================================================
Project : Drift-Aware Continuous Learning Framework
File    : src/retraining/retrain_pipeline.py

HOW IT WORKS (plain language)
------------------------------
When the DriftDetector fires for a category, this module:

  1. Collects the last 45 days of ACTUAL demand for that category
     (not the model's predictions — the real values)
     Why 45 days? The default window_days=90 but run_drift_check.py
     passes window_days=45 to keep the training data recent and tight.

  2. Retrains a fresh Prophet model on those days minus a 14-day holdout
     The 14-day holdout is never seen during training — it's the test set.

  3. Evaluates the new model on the 14-day holdout
     to confirm it's better than the old model on the same window

  4. If new model is better: replaces old model, saves to models/{slug}.pkl
     If new model is worse: keeps old model (safety gate)

  5. Logs everything to MLflow:
     - experiment: "demand_forecasting_drift"
     - tags    : category, retrain_date, trigger_reason, model_accepted
     - params  : window_days, holdout_days, train_days, seasonality, changepoint_scale
     - metrics : pre_retrain_mae, post_retrain_mae, mae_improvement, mae_improvement_pct
     - dataset : final_demand_series (source: data/processed/final_demand_series.csv)
     - artifact: Prophet model (accepted retrains only) with ModelSignature
     If accepted, also registers to MLflow Model Registry as demand_{slug}@production.

WHY 45-DAY WINDOW (when called from CI)
-----------------------------------------
Using all training data (Jan 2024 onward) would teach the model
the OLD demand pattern — exactly what we're trying to escape.
45 days of recent data reflects the new pattern. The class default
is window_days=90 but run_drift_check.py passes window_days=45 for
tighter adaptation to abrupt drift events.

MLFLOW TRACKING (exact keys)
-----------------------------
Every retrain creates a new MLflow run under experiment "demand_forecasting_drift":
  tags    : category, retrain_date, trigger_reason, model_type, model_accepted
  params  : window_days, holdout_days, train_days, seasonality, changepoint_scale
  metrics : pre_retrain_mae, post_retrain_mae, mae_improvement, mae_improvement_pct
  dataset : final_demand_series  (source: data/processed/final_demand_series.csv)
  artifact: prophet_model/  (only for accepted retrains, includes ModelSignature)
  registry: demand_{slug}@production  (only for accepted retrains)
"""

import pandas as pd
import numpy as np
import warnings
import logging
import os
import re
import joblib
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Optional, Dict, Tuple


def _category_slug(category: str) -> str:
    """Convert 'Electronics & Tech' -> 'electronics_tech' for use as filename."""
    return re.sub(r'[^a-z0-9]+', '_', category.lower()).strip('_')

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

try:
    import mlflow
    import mlflow.pyfunc
    from mlflow.types.schema import Schema, ColSpec
    from mlflow.models.signature import ModelSignature
    MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow: Any = None
    MLFLOW_AVAILABLE = False
    print("⚠️  MLflow not installed — run: pip install mlflow")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    Prophet: Any = None
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not installed — run: pip install prophet")


# ─────────────────────────────────────────────────────────
# Data class — result of one retrain event
# ─────────────────────────────────────────────────────────

@dataclass
class RetrainResult:
    category:           str
    retrain_date:       str
    trigger_reason:     str          # 'drift_detected' or 'manual'
    window_days:        int          # how many days of data used
    pre_retrain_mae:    float        # old model's recent MAE
    post_retrain_mae:   float        # new model's MAE on holdout
    mae_improvement:    float        # pre - post (positive = better)
    mae_improvement_pct: float       # % improvement
    model_accepted:     bool         # True if new model is better
    mlflow_run_id:      Optional[str]
    new_baseline_mae:   float        # MAE to pass to DriftDetector.reset()

    def to_dict(self):
        return {
            'category'           : self.category,
            'retrain_date'       : self.retrain_date,
            'trigger_reason'     : self.trigger_reason,
            'window_days'        : self.window_days,
            'pre_retrain_mae'    : round(self.pre_retrain_mae, 1),
            'post_retrain_mae'   : round(self.post_retrain_mae, 1),
            'mae_improvement'    : round(self.mae_improvement, 1),
            'mae_improvement_pct': round(self.mae_improvement_pct, 2),
            'model_accepted'     : self.model_accepted,
            'mlflow_run_id'      : self.mlflow_run_id,
            'new_baseline_mae'   : round(self.new_baseline_mae, 1),
        }

    def __repr__(self):
        status = "✅ ACCEPTED" if self.model_accepted else "❌ REJECTED"
        return (
            f"RetrainResult({self.category}, {self.retrain_date}, {status}, "
            f"MAE: {self.pre_retrain_mae:.0f} → {self.post_retrain_mae:.0f} "
            f"({self.mae_improvement_pct:+.1f}%))"
        )


# ─────────────────────────────────────────────────────────
# RetrainPipeline — core class
# ─────────────────────────────────────────────────────────

class RetrainPipeline:
    """
    Manages Prophet retraining for all product categories.

    Parameters
    ----------
    demand_data     : pd.DataFrame with columns [ds, category, y]
                      The full demand time series
    window_days     : int — how many recent days to train on (default 90)
    holdout_days    : int — how many days to hold out for evaluation (default 14)
    mlflow_tracking : bool — whether to log to MLflow (default True)
    experiment_name : str — MLflow experiment name
    """

    def __init__(
        self,
        demand_data:     pd.DataFrame,
        window_days:     int  = 90,
        holdout_days:    int  = 14,
        mlflow_tracking: bool = True,
        experiment_name: str  = 'demand_forecasting_drift',
        model_dir:       str  = 'models',
    ):
        self.demand_data     = demand_data.copy()
        self.demand_data['ds'] = pd.to_datetime(self.demand_data['ds'])
        self.window_days     = window_days
        self.holdout_days    = holdout_days
        self.mlflow_tracking = mlflow_tracking and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.model_dir       = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self._retrain_log: list[RetrainResult] = []

        # Setup MLflow
        if self.mlflow_tracking:
            mlflow.set_experiment(experiment_name)
            print(f"  MLflow experiment: '{experiment_name}'")
        else:
            print("  MLflow tracking disabled")

    # ── Main method — call when drift is detected ─────────

    def retrain(
        self,
        category:       str,
        retrain_date:   str,
        current_model:  Any,
        pre_retrain_mae: float,
        trigger_reason: str = 'drift_detected',
    ) -> Tuple[RetrainResult, Any]:
        """
        Retrain Prophet for one category on the most recent window_days.

        Parameters
        ----------
        category        : product category name
        retrain_date    : date string 'YYYY-MM-DD' — the day drift was detected
        current_model   : the existing Prophet model (kept as fallback)
        pre_retrain_mae : the current rolling MAE that triggered the retrain
        trigger_reason  : 'drift_detected' or 'manual'

        Returns
        -------
        (RetrainResult, new_prophet_model_or_None)
        If model_accepted=True, use the new model
        If model_accepted=False, keep the current model
        """
        print(f"\n  ⚙️  Retraining {category} on {retrain_date}...")

        retrain_dt = pd.Timestamp(retrain_date)

        # ── 1. Collect retraining data
        window_start = retrain_dt - timedelta(days=self.window_days)
        holdout_start = retrain_dt - timedelta(days=self.holdout_days)

        cat_data = (
            self.demand_data[self.demand_data['category'] == category]
            .sort_values('ds')
        )

        # Training slice: window_start to (holdout_start - 1 day)
        train_data = cat_data[
            (cat_data['ds'] >= window_start) &
            (cat_data['ds'] < holdout_start)
        ][['ds', 'y']].reset_index(drop=True)

        # Holdout slice: holdout_start to retrain_date
        holdout_data = cat_data[
            (cat_data['ds'] >= holdout_start) &
            (cat_data['ds'] <= retrain_dt)
        ][['ds', 'y']].reset_index(drop=True)

        if len(train_data) < 30:
            print(f"  ⚠️  Not enough training data ({len(train_data)} days). Skipping.")
            result = RetrainResult(
                category=category, retrain_date=retrain_date,
                trigger_reason=trigger_reason, window_days=self.window_days,
                pre_retrain_mae=pre_retrain_mae, post_retrain_mae=pre_retrain_mae,
                mae_improvement=0, mae_improvement_pct=0,
                model_accepted=False, mlflow_run_id=None,
                new_baseline_mae=pre_retrain_mae,
            )
            self._retrain_log.append(result)
            return result, None

        # ── 2. Train new Prophet model
        new_model = Prophet(
            yearly_seasonality      = 'auto',
            weekly_seasonality      = 'auto',
            daily_seasonality       = 'auto',
            seasonality_mode        = 'multiplicative',
            changepoint_prior_scale = 0.1,   # slightly more flexible for adaptation
            interval_width          = 0.95,
        )
        new_model.fit(train_data)

        # ── 3. Evaluate on holdout
        future_holdout = pd.DataFrame({'ds': holdout_data['ds']})
        forecast_holdout = new_model.predict(future_holdout)
        post_mae = float(np.mean(
            np.abs(holdout_data['y'].to_numpy(dtype=float) - forecast_holdout['yhat'].to_numpy(dtype=float))
        ))

        # ── 4. Also evaluate OLD model on same holdout (for comparison)
        if current_model is not None:
            old_forecast = current_model.predict(future_holdout)
            old_holdout_mae = float(np.mean(
                np.abs(holdout_data['y'].to_numpy(dtype=float) - old_forecast['yhat'].to_numpy(dtype=float))
            ))
        else:
            old_holdout_mae = pre_retrain_mae

        improvement     = old_holdout_mae - post_mae
        improvement_pct = (improvement / old_holdout_mae) * 100 if old_holdout_mae > 0 else 0
        model_accepted  = post_mae < old_holdout_mae  # accept only if genuinely better

        new_baseline = post_mae if model_accepted else pre_retrain_mae

        # ── 5. Log to MLflow
        run_id = None
        if self.mlflow_tracking:
            run_id = self._log_to_mlflow(
                category        = category,
                retrain_date    = retrain_date,
                trigger_reason  = trigger_reason,
                train_days      = len(train_data),
                train_data      = train_data,
                pre_mae         = old_holdout_mae,
                post_mae        = post_mae,
                improvement_pct = improvement_pct,
                model_accepted  = model_accepted,
                new_model       = new_model if model_accepted else None,
            )

        # ── 6. Report
        status = "✅ ACCEPTED" if model_accepted else "❌ REJECTED (old model better)"
        print(f"     Pre-retrain MAE  : Rs.{old_holdout_mae:,.0f}")
        print(f"     Post-retrain MAE : Rs.{post_mae:,.0f}")
        print(f"     Improvement      : {improvement_pct:+.1f}%")
        print(f"     Decision         : {status}")
        if run_id:
            print(f"     MLflow run ID    : {run_id[:8]}...")
            if model_accepted:
                print(f"     Model Registry   : demand_{_category_slug(category)} @ production")

        result = RetrainResult(
            category=category, retrain_date=retrain_date,
            trigger_reason=trigger_reason, window_days=self.window_days,
            pre_retrain_mae=old_holdout_mae, post_retrain_mae=post_mae,
            mae_improvement=improvement, mae_improvement_pct=improvement_pct,
            model_accepted=model_accepted, mlflow_run_id=run_id,
            new_baseline_mae=new_baseline,
        )

        self._retrain_log.append(result)

        # ── 7. Persist accepted model to disk so CI can reuse it next run
        if model_accepted and new_model is not None:
            slug      = _category_slug(category)
            save_path = os.path.join(self.model_dir, f'{slug}.pkl')
            joblib.dump(new_model, save_path)
            print(f"     Model saved      : {save_path}")

        return result, new_model if model_accepted else None

    # ── MLflow logging ─────────────────────────────────────

    def _log_to_mlflow(
        self, category, retrain_date, trigger_reason, train_days,
        train_data, pre_mae, post_mae, improvement_pct, model_accepted, new_model
    ) -> Optional[str]:
        try:
            with mlflow.start_run(run_name=f"{category.split()[0]}_{retrain_date}") as run:
                # Log training dataset — populates the ‘Dataset’ column in the UI
                # Source points to the master CSV; name identifies the category slice
                dataset = mlflow.data.from_pandas(
                    train_data,
                    source="data/processed/final_demand_series.csv",
                    name="final_demand_series",
                    targets="y",
                )
                mlflow.log_input(dataset, context="training")
                # Tags — searchable metadata
                mlflow.set_tags({
                    'category'      : category,
                    'retrain_date'  : retrain_date,
                    'trigger_reason': trigger_reason,
                    'model_type'    : 'prophet',
                    'model_accepted': str(model_accepted),
                })

                # Parameters — what was used for this retrain
                mlflow.log_params({
                    'window_days'   : self.window_days,
                    'holdout_days'  : self.holdout_days,
                    'train_days'    : train_days,
                    'seasonality'   : 'multiplicative',
                    'changepoint_scale': 0.1,
                })

                # Metrics — what changed
                mlflow.log_metrics({
                    'pre_retrain_mae'    : round(pre_mae, 2),
                    'post_retrain_mae'   : round(post_mae, 2),
                    'mae_improvement'    : round(pre_mae - post_mae, 2),
                    'mae_improvement_pct': round(improvement_pct, 4),
                })

                # Log model artifact + register to Model Registry if accepted
                if model_accepted and new_model is not None:
                    try:
                        # Define input/output schema so MLflow UI shows model details
                        signature = ModelSignature(
                            inputs =Schema([ColSpec("string", "ds")]),
                            outputs=Schema([
                                ColSpec("double", "yhat"),
                                ColSpec("double", "yhat_lower"),
                                ColSpec("double", "yhat_upper"),
                            ]),
                        )
                        mlflow.prophet.log_model(new_model, 'prophet_model', signature=signature)
                        # Register to Model Registry — creates versioned production model
                        model_name = f"demand_{_category_slug(category)}"
                        reg = mlflow.register_model(
                            model_uri=f"runs:/{run.info.run_id}/prophet_model",
                            name=model_name,
                        )
                        # Tag this version as 'production' immediately
                        client = mlflow.MlflowClient()
                        client.set_registered_model_alias(model_name, "production", reg.version)
                    except Exception:
                        # prophet MLflow flavor may need extra install
                        pass

                return run.info.run_id

        except Exception as e:
            print(f"     ⚠️  MLflow logging failed: {e}")
            return None

    # ── Getters ───────────────────────────────────────────

    def get_retrain_log(self) -> list:
        return [r.to_dict() for r in self._retrain_log]

    def get_retrain_summary(self) -> pd.DataFrame:
        if not self._retrain_log:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self._retrain_log])

    def save_retrain_log(self, path: str = 'reports/drift_logs/retrain_log.csv'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = self.get_retrain_summary()
        if not df.empty:
            df.to_csv(path, index=False)
            print(f"  Retrain log saved: {path}")
