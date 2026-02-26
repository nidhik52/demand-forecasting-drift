"""
Drift Detector — Rolling MAE with Dual Window
==============================================
Project : Drift-Aware Continuous Learning Framework
File    : src/drift_detection/drift_detector.py

HOW IT WORKS (plain language)
------------------------------
Every day the model makes a forecast. We compare it to what actually
happened (the forecast error). This detector tracks that error over
time using two sliding windows:

  Short window (7 days)  — catches abrupt drift fast
                           e.g. Electronics spike on Nov 1
                           
  Long window  (30 days) — catches gradual drift reliably
                           e.g. Health ramp from Aug to Dec

Drift is flagged when EITHER window's rolling MAE exceeds
1.5x the baseline (computed on the clean validation period).

The flag must persist for 3 consecutive days before retraining
is triggered — this avoids reacting to single-day noise spikes.

WHY DUAL WINDOW (paper contribution)
-------------------------------------
Most drift detection papers use a single window. A single short
window fires too many false positives on gradual drift. A single
long window misses abrupt drift for several days. The dual-window
approach independently monitors both timescales simultaneously.

Expected detection times on this dataset:
  Electronics (abrupt +50%): short window fires Nov 1, long ~Nov 4
  Health (gradual +40%):     long window fires ~Nov 14-16
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import json
from datetime import datetime


# ─────────────────────────────────────────────────────────
# Data class — one status snapshot per day per category
# ─────────────────────────────────────────────────────────

@dataclass
class DriftStatus:
    date:              str
    category:          str
    actual:            float
    predicted:         float
    daily_error:       float          # |actual - predicted|
    short_mae:         float          # 7-day rolling MAE
    long_mae:          float          # 30-day rolling MAE
    short_ratio:       float          # short_mae / baseline_mae
    long_ratio:        float          # long_mae  / baseline_mae
    baseline_mae:      float
    short_flagged:     bool           # short window > threshold
    long_flagged:      bool           # long window  > threshold
    is_drifting:       bool           # either flagged for min_days
    consecutive_days:  int            # days either window has been flagged
    retrain_triggered: bool           # True on the day retrain fires

    def to_dict(self):
        return {
            'date'             : self.date,
            'category'         : self.category,
            'actual'           : round(self.actual, 2),
            'predicted'        : round(self.predicted, 2),
            'daily_error'      : round(self.daily_error, 2),
            'short_mae'        : round(self.short_mae, 2),
            'long_mae'         : round(self.long_mae, 2),
            'short_ratio'      : round(self.short_ratio, 3),
            'long_ratio'       : round(self.long_ratio, 3),
            'baseline_mae'     : round(self.baseline_mae, 2),
            'short_flagged'    : self.short_flagged,
            'long_flagged'     : self.long_flagged,
            'is_drifting'      : self.is_drifting,
            'consecutive_days' : self.consecutive_days,
            'retrain_triggered': self.retrain_triggered,
        }


# ─────────────────────────────────────────────────────────
# DriftDetector — one instance per category
# ─────────────────────────────────────────────────────────

class DriftDetector:
    """
    Rolling MAE dual-window drift detector for one product category.

    Parameters
    ----------
    baseline_mae    : float  — MAE on clean validation window (Oct 2025)
    threshold       : float  — flag when rolling_mae > threshold * baseline
                               default 1.5 = flag when 50% worse than baseline
    short_window    : int    — short window size in days (default 7)
                               catches abrupt drift fast
    long_window     : int    — long window size in days (default 30)
                               catches gradual drift reliably
    min_days        : int    — consecutive flagged days before retrain fires
                               default 3 — avoids reacting to single spikes
    category        : str    — category name (for logging)
    """

    def __init__(
        self,
        baseline_mae: float,
        threshold:    float = 1.5,
        short_window: int   = 7,
        long_window:  int   = 30,
        min_days:     int   = 3,
        category:     str   = 'unknown',
    ):
        self.baseline_mae  = baseline_mae
        self.threshold     = threshold
        self.short_window  = short_window
        self.long_window   = long_window
        self.min_days      = min_days
        self.category      = category

        # Sliding windows of daily absolute errors
        self._short_errors = deque(maxlen=short_window)
        self._long_errors  = deque(maxlen=long_window)

        # State
        self._consecutive_days = 0
        self._retrain_count    = 0
        self._history          = []   # list of DriftStatus dicts

    # ── Core update — call once per day ──────────────────

    def update(
        self,
        actual:    float,
        predicted: float,
        date:      Optional[str] = None,
    ) -> DriftStatus:
        """
        Feed one day's actual and predicted demand.
        Returns a DriftStatus with all computed metrics.

        Parameters
        ----------
        actual    : today's real demand value
        predicted : what the model forecast for today
        date      : date string (optional, for logging)

        Returns
        -------
        DriftStatus dataclass (also stored in self._history)
        """
        daily_error = abs(actual - predicted)
        self._short_errors.append(daily_error)
        self._long_errors.append(daily_error)

        short_mae = float(np.mean(self._short_errors))
        long_mae  = float(np.mean(self._long_errors))

        short_ratio = short_mae / self.baseline_mae
        long_ratio  = long_mae  / self.baseline_mae

        short_flagged = short_ratio > self.threshold
        long_flagged  = long_ratio  > self.threshold
        either_flagged = short_flagged or long_flagged

        if either_flagged:
            self._consecutive_days += 1
        else:
            self._consecutive_days = 0

        retrain_triggered = (
            self._consecutive_days >= self.min_days
            and either_flagged
            # Only trigger once per drift event — reset after triggering
            and self._consecutive_days == self.min_days
        )

        if retrain_triggered:
            self._retrain_count += 1

        status = DriftStatus(
            date              = date or str(datetime.now().date()),
            category          = self.category,
            actual            = actual,
            predicted         = predicted,
            daily_error       = daily_error,
            short_mae         = short_mae,
            long_mae          = long_mae,
            short_ratio       = short_ratio,
            long_ratio        = long_ratio,
            baseline_mae      = self.baseline_mae,
            short_flagged     = short_flagged,
            long_flagged      = long_flagged,
            is_drifting       = either_flagged,
            consecutive_days  = self._consecutive_days,
            retrain_triggered = retrain_triggered,
        )

        self._history.append(status.to_dict())
        return status

    # ── Reset after retraining ────────────────────────────

    def reset(self, new_baseline_mae: Optional[float] = None):
        """
        Call after retraining completes.
        Clears windows and resets consecutive day counter.
        Optionally updates baseline to post-retrain performance.

        Parameters
        ----------
        new_baseline_mae : if provided, updates the detector's baseline
                           to the new model's performance level
        """
        self._short_errors.clear()
        self._long_errors.clear()
        self._consecutive_days = 0
        if new_baseline_mae is not None:
            self.baseline_mae = new_baseline_mae

    # ── Getters ───────────────────────────────────────────

    @property
    def current_short_mae(self) -> float:
        return float(np.mean(self._short_errors)) if self._short_errors else 0.0

    @property
    def current_long_mae(self) -> float:
        return float(np.mean(self._long_errors)) if self._long_errors else 0.0

    @property
    def current_short_ratio(self) -> float:
        return self.current_short_mae / self.baseline_mae

    @property
    def current_long_ratio(self) -> float:
        return self.current_long_mae / self.baseline_mae

    @property
    def is_drifting(self) -> bool:
        return self._consecutive_days > 0

    @property
    def consecutive_days(self) -> int:
        return self._consecutive_days

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def get_history(self) -> list:
        return self._history.copy()

    def get_summary(self) -> dict:
        return {
            'category'        : self.category,
            'baseline_mae'    : round(self.baseline_mae, 2),
            'current_short_mae': round(self.current_short_mae, 2),
            'current_long_mae' : round(self.current_long_mae, 2),
            'short_ratio'     : round(self.current_short_ratio, 3),
            'long_ratio'      : round(self.current_long_ratio, 3),
            'is_drifting'     : self.is_drifting,
            'consecutive_days': self.consecutive_days,
            'retrain_count'   : self.retrain_count,
            'threshold'       : self.threshold,
        }

    def __repr__(self):
        return (
            f"DriftDetector(category='{self.category}', "
            f"baseline={self.baseline_mae:.0f}, "
            f"short_ratio={self.current_short_ratio:.2f}x, "
            f"long_ratio={self.current_long_ratio:.2f}x, "
            f"drifting={self.is_drifting})"
        )


# ─────────────────────────────────────────────────────────
# DriftDetectorRegistry — manages one detector per category
# ─────────────────────────────────────────────────────────

class DriftDetectorRegistry:
    """
    Manages a DriftDetector for each product category.
    Single entry point for the pipeline to use.

    Usage
    -----
    registry = DriftDetectorRegistry(baseline_maes)
    status   = registry.update('Electronics & Tech', actual, predicted, date)
    if status.retrain_triggered:
        # trigger retraining for this category
        registry.reset('Electronics & Tech', new_baseline_mae=new_mae)
    """

    def __init__(
        self,
        baseline_maes: dict,           # {category: baseline_mae}
        threshold:     float = 1.5,
        short_window:  int   = 7,
        long_window:   int   = 30,
        min_days:      int   = 3,
    ):
        self._detectors = {
            cat: DriftDetector(
                baseline_mae  = mae,
                threshold     = threshold,
                short_window  = short_window,
                long_window   = long_window,
                min_days      = min_days,
                category      = cat,
            )
            for cat, mae in baseline_maes.items()
        }

    def update(
        self,
        category:  str,
        actual:    float,
        predicted: float,
        date:      Optional[str] = None,
    ) -> DriftStatus:
        return self._detectors[category].update(actual, predicted, date)

    def reset(self, category: str, new_baseline_mae: Optional[float] = None):
        self._detectors[category].reset(new_baseline_mae)

    def get_all_summaries(self) -> dict:
        return {cat: d.get_summary() for cat, d in self._detectors.items()}

    def get_history(self, category: str) -> list:
        return self._detectors[category].get_history()

    def get_all_history(self) -> dict:
        return {cat: d.get_history() for cat, d in self._detectors.items()}

    def __getitem__(self, category: str) -> DriftDetector:
        return self._detectors[category]
