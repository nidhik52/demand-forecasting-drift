"""
Prophet Forecaster — Clean Wrapper Class
=========================================
Project : Drift-Aware Continuous Learning Framework
File    : src/forecasting/prophet_model.py

PURPOSE
-------
Wraps Prophet in a clean class interface so that:
  - FastAPI can import and use it without knowing Prophet internals
  - The retraining pipeline can swap models cleanly
  - The drift detector gets consistent actual/predicted pairs
  - The inventory module gets yhat_lower/yhat_upper consistently

All other modules import ProphetForecaster, not Prophet directly.
This is the "model-agnostic" architecture in practice — to swap
Prophet for XGBoost, you only change this file.
"""

import logging
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from typing import Any, Optional

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet: Any = None  # type: ignore[assignment,misc]
    PROPHET_AVAILABLE = False


class ProphetForecaster:
    """
    Clean wrapper around Facebook Prophet for daily demand forecasting.

    Parameters
    ----------
    category            : str   — product category name (for logging)
    yearly_seasonality  : bool  — model annual demand cycles
    weekly_seasonality  : bool  — model Mon-Sun demand patterns
    seasonality_mode    : str   — 'multiplicative' (default) or 'additive'
    changepoint_scale   : float — flexibility of trend changes (0.05 = stable)
    interval_width      : float — confidence interval width (0.95 = 95% CI)

    Usage
    -----
    forecaster = ProphetForecaster(category='Electronics & Tech')
    forecaster.fit(train_df)                        # train_df: ds, y columns
    forecast = forecaster.predict(periods=30)       # next 30 days
    forecast = forecaster.predict_dates(date_list)  # specific dates
    mae = forecaster.evaluate(test_df)              # compute MAE on test
    """

    def __init__(
        self,
        category:           str   = 'unknown',
        yearly_seasonality: str   = 'auto',
        weekly_seasonality: str   = 'auto',
        seasonality_mode:   str   = 'multiplicative',
        changepoint_scale:  float = 0.05,
        interval_width:     float = 0.95,
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Run: pip install prophet")

        self.category           = category
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.seasonality_mode   = seasonality_mode
        self.changepoint_scale  = changepoint_scale
        self.interval_width     = interval_width

        self.model:    Any = None
        self.train_df: Optional[pd.DataFrame] = None
        self._is_fitted = False
        self._version   = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> 'ProphetForecaster':
        """
        Fit Prophet on training data.

        Parameters
        ----------
        train_df : pd.DataFrame with columns ['ds', 'y']
                   ds = datetime, y = demand value

        Returns self (for chaining).
        """
        if not {'ds', 'y'}.issubset(train_df.columns):
            raise ValueError("train_df must have columns 'ds' and 'y'")

        self.train_df = train_df[['ds', 'y']].copy()
        self.train_df['ds'] = pd.to_datetime(self.train_df['ds'])

        self.model = Prophet(
            yearly_seasonality      = self.yearly_seasonality,
            weekly_seasonality      = self.weekly_seasonality,
            daily_seasonality       = 'auto',
            seasonality_mode        = self.seasonality_mode,
            changepoint_prior_scale = self.changepoint_scale,
            interval_width          = self.interval_width,
        )
        self.model.fit(self.train_df)
        self._is_fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, periods: int = 30) -> pd.DataFrame:
        """
        Forecast the next `periods` days after training cutoff.

        Returns pd.DataFrame with columns:
          ds, yhat, yhat_lower, yhat_upper, trend, weekly, yearly
        """
        self._check_fitted()
        future   = self.model.make_future_dataframe(periods=periods, freq='D',
                                                     include_history=False)
        forecast = self.model.predict(future)
        # Clip negative predictions
        forecast['yhat']       = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]

    def predict_dates(self, dates) -> pd.DataFrame:
        """
        Forecast for a specific list of dates.
        Used in walk-forward evaluation — one day at a time.

        Parameters
        ----------
        dates : list/array of datetime-like values

        Returns pd.DataFrame with ds, yhat, yhat_lower, yhat_upper
        """
        self._check_fitted()
        future   = pd.DataFrame({'ds': pd.to_datetime(dates)})
        forecast = self.model.predict(future)
        forecast['yhat']       = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def predict_one(self, date) -> dict:
        """
        Forecast for a single date. Returns dict.
        Used in real-time playback controller.
        """
        result = self.predict_dates([date])
        row    = result.iloc[0]
        return {
            'ds'         : str(row['ds'].date()),
            'yhat'       : round(float(row['yhat']), 2),
            'yhat_lower' : round(float(row['yhat_lower']), 2),
            'yhat_upper' : round(float(row['yhat_upper']), 2),
        }

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluate model on a held-out test set.

        Parameters
        ----------
        test_df : pd.DataFrame with columns ['ds', 'y']

        Returns dict with MAE, RMSE, MAPE
        """
        self._check_fitted()
        forecast  = self.predict_dates(test_df['ds'])
        actual    = test_df['y'].to_numpy(dtype=float)
        predicted = forecast['yhat'].to_numpy(dtype=float)

        mae  = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mask = actual != 0
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

        return {
            'category' : self.category,
            'n_days'   : len(actual),
            'MAE'      : round(mae, 2),
            'RMSE'     : round(rmse, 2),
            'MAPE'     : round(mape, 2),
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_components(self) -> pd.DataFrame:
        """Return trend + seasonality components for plotting."""
        self._check_fitted()
        future = self.model.make_future_dataframe(periods=0)
        return self.model.predict(future)

    def increment_version(self):
        """Call after retraining to track model version."""
        self._version += 1

    @property
    def version(self) -> int:
        return self._version

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def train_cutoff(self):
        """Last date in training data."""
        if self.train_df is not None:
            return self.train_df['ds'].max()
        return None

    def _check_fitted(self):
        if not self._is_fitted or self.model is None:
            raise RuntimeError(
                f"ProphetForecaster for '{self.category}' is not fitted. "
                "Call .fit(train_df) first."
            )

    def __repr__(self):
        cutoff = self.train_cutoff
        status = f"fitted (cutoff={cutoff.date()})" if self._is_fitted and cutoff is not None else "not fitted"
        return f"ProphetForecaster(category='{self.category}', v{self._version}, {status})"


# ── ForecasterRegistry — one per category ─────────────────────────────────────

class ForecasterRegistry:
    """
    Manages one ProphetForecaster per product category.
    Used by FastAPI endpoints to serve forecasts.

    Usage
    -----
    registry = ForecasterRegistry()
    registry.fit_all(demand_df, train_cutoff='2025-09-30')
    forecast  = registry.predict('Electronics & Tech', periods=30)
    metrics   = registry.evaluate('Electronics & Tech', test_df)
    registry.refit('Electronics & Tech', new_train_df)  # after drift retrain
    """

    def __init__(self):
        self._forecasters: dict[str, ProphetForecaster] = {}

    def fit_all(
        self,
        demand_df:     pd.DataFrame,
        train_cutoff:  str,
        **prophet_kwargs,
    ) -> 'ForecasterRegistry':
        """
        Fit one ProphetForecaster per category up to train_cutoff.

        Parameters
        ----------
        demand_df    : full demand time series (ds, category, y)
        train_cutoff : date string 'YYYY-MM-DD' — training cutoff
        """
        cutoff     = pd.Timestamp(train_cutoff)
        categories = sorted(demand_df['category'].unique())

        for cat in categories:
            cat_df     = demand_df[demand_df['category'] == cat].sort_values('ds')
            train_data = cat_df[cat_df['ds'] <= cutoff][['ds', 'y']]
            forecaster = ProphetForecaster(category=cat, **prophet_kwargs)
            forecaster.fit(train_data)
            self._forecasters[cat] = forecaster

        return self

    def predict(self, category: str, periods: int = 30) -> pd.DataFrame:
        return self._get(category).predict(periods)

    def predict_one(self, category: str, date) -> dict:
        return self._get(category).predict_one(date)

    def evaluate(self, category: str, test_df: pd.DataFrame) -> dict:
        return self._get(category).evaluate(test_df)

    def refit(self, category: str, new_train_df: pd.DataFrame):
        """Replace model for one category after retraining."""
        forecaster = self._forecasters.get(category)
        if forecaster:
            forecaster.fit(new_train_df)
            forecaster.increment_version()
        else:
            new_f = ProphetForecaster(category=category)
            new_f.fit(new_train_df)
            self._forecasters[category] = new_f

    def get_version(self, category: str) -> int:
        return self._get(category).version

    def get_all_summaries(self) -> dict:
        return {
            cat: {
                'version'     : f.version,
                'is_fitted'   : f.is_fitted,
                'train_cutoff': str(f.train_cutoff.date()) if f.train_cutoff else None,
            }
            for cat, f in self._forecasters.items()
        }

    def categories(self) -> list:
        return list(self._forecasters.keys())

    def _get(self, category: str) -> ProphetForecaster:
        if category not in self._forecasters:
            raise KeyError(
                f"Category '{category}' not in registry. "
                f"Available: {list(self._forecasters.keys())}"
            )
        return self._forecasters[category]

    def __repr__(self):
        cats = list(self._forecasters.keys())
        return f"ForecasterRegistry({len(cats)} categories: {cats})"
