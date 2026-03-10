"""
drift_detection.py
──────────────────
Monitors forecast accuracy in real time using a rolling Mean Absolute
Error (MAE) window.  When the rolling MAE exceeds a configurable
threshold, a drift event is flagged and logged.

Concept drift in this context
──────────────────────────────
Concept drift occurs when the statistical relationship between inputs
and the target (demand) changes over time — e.g. a product goes viral,
a supply chain disruption happens, or seasonality shifts. The model's
forecast errors grow, and the rolling MAE crosses the threshold.

Algorithm
─────────
  For each incoming day:
    1. Compute absolute error  = |actual − forecast|
    2. Append to a per-SKU rolling buffer (last N days)
    3. rolling_MAE = mean of that buffer
    4. If buffer is full AND rolling_MAE > threshold → drift detected

Drift events are appended to  data/drift_logs/drift_events.csv.
"""

from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DRIFT_LOG_DIR  = Path("data/drift_logs")
DRIFT_LOG_PATH = DRIFT_LOG_DIR / "drift_events.csv"

# ── Default parameters (overridable via constructor) ──────────────────────────
DEFAULT_WINDOW    = 14      # rolling window in days
DEFAULT_THRESHOLD = 20.0    # MAE units above which drift is declared


class DriftDetector:
    """
    Stateful, per-SKU rolling-error monitor.

    Usage
    ─────
        detector = DriftDetector()
        result   = detector.update(
            sku="ELEC-001",
            date="2025-03-15",
            actual=42,
            forecast=30,
        )
        if result["drift_detected"]:
            # trigger retraining …
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.window    = window
        self.threshold = threshold
        # {sku: deque of absolute errors, maxlen=window}
        self._buffers: dict[str, deque] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        sku: str,
        date: datetime | str,
        actual: float,
        forecast: float,
    ) -> dict:
        """
        Record one new observation and return a drift-check result.

        Returns a dict with keys:
          sku, date, actual, forecast,
          absolute_error, rolling_mae, drift_detected
        """
        abs_error = abs(actual - forecast)

        # Initialise a new buffer for first-seen SKUs
        if sku not in self._buffers:
            self._buffers[sku] = deque(maxlen=self.window)

        self._buffers[sku].append(abs_error)

        rolling_mae = sum(self._buffers[sku]) / len(self._buffers[sku])

        # Drift is only flagged once the buffer is full (need a full window
        # of errors before we can reliably declare drift)
        drift_detected = (
            len(self._buffers[sku]) == self.window
            and rolling_mae > self.threshold
        )

        result = {
            "sku":             sku,
            "date":            str(date),
            "actual":          round(actual, 2),
            "forecast":        round(forecast, 2),
            "absolute_error":  round(abs_error, 2),
            "rolling_mae":     round(rolling_mae, 2),
            "drift_detected":  drift_detected,
        }

        if drift_detected:
            _log_drift_event(result, threshold=self.threshold)

        return result

    def reset(self, sku: str | None = None) -> None:
        """
        Clear the error buffer after a model is retrained.

        Pass a specific SKU to reset only that SKU's buffer,
        or call with no argument to reset all buffers.
        """
        if sku:
            self._buffers.pop(sku, None)
        else:
            self._buffers.clear()

    def rolling_mae(self, sku: str) -> float | None:
        """Return the current rolling MAE for a SKU, or None if no data."""
        buf = self._buffers.get(sku)
        if not buf:
            return None
        return round(sum(buf) / len(buf), 2)


# ── Logging helper (module-level, used by the class) ─────────────────────────

def _log_drift_event(event: dict, threshold: float) -> None:
    """Append one drift event row to the CSV log."""
    DRIFT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([{
        "timestamp":      datetime.now().isoformat(),
        "sku":            event["sku"],
        "date":           event["date"],
        "rolling_mae":    event["rolling_mae"],
        "threshold":      threshold,
        "drift_detected": True,
        "event_type":     "drift_detected",
    }])

    write_header = not DRIFT_LOG_PATH.exists()
    row.to_csv(DRIFT_LOG_PATH, mode="a", header=write_header, index=False)


def load_drift_log() -> pd.DataFrame:
    """
    Return all logged drift events as a DataFrame.
    Returns an empty DataFrame if no events have been logged yet.
    """
    if DRIFT_LOG_PATH.exists():
        return pd.read_csv(DRIFT_LOG_PATH, parse_dates=["timestamp"])
    return pd.DataFrame(
        columns=["timestamp", "sku", "date", "rolling_mae", "threshold", "drift_detected"]
    )
