"""Shared helpers for the JCMM_1268 revision experiments (new file).

Metric definitions follow manuscript Eqs. 8-11: MAE and RMSE over all
observations, MAPE and the 20% tolerance share over nonzero actual demand.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_DEMAND = REPO_ROOT / "data/processed/daily_demand.csv"
REVISION_DATA = REPO_ROOT / "data/revision"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_daily_demand():
    df = pd.read_csv(DAILY_DEMAND)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def error_metrics(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    nz = actual > 0
    ape = np.abs(err[nz]) / actual[nz]
    return {
        "n": int(len(actual)),
        "n_nonzero": int(nz.sum()),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_pct": float(np.mean(ape) * 100),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Tol20_pct": float(np.mean(ape <= 0.20) * 100),
    }


def injected_windows(demand, seed=42, window_count=4, window_size=12):
    """Replays the random draws of src/preprocessing.py::inject_random_drift.

    The SKU order is taken from the saved daily_demand.csv, which is the frame
    that inject_random_drift received and returned without re-sorting.
    Windows are inclusive: Date.between(start, start + window_size).
    """
    rng = np.random.default_rng(seed)
    min_date, max_date = demand["Date"].min(), demand["Date"].max()
    total_days = (max_date - min_date).days
    sku_list = demand["SKU"].unique().tolist()
    rows = []
    for k in range(window_count):
        start_offset = int(rng.integers(0, total_days - window_size))
        start = min_date + pd.Timedelta(days=start_offset)
        end = start + pd.Timedelta(days=window_size)
        skus = rng.choice(sku_list, size=max(1, len(sku_list) // 6), replace=False)
        multiplier = float(rng.uniform(1.6, 2.4))
        for sku in skus:
            rows.append({"Draw": k + 1, "SKU": str(sku), "Window_Start": start,
                         "Window_End": end, "Multiplier": multiplier})
    return pd.DataFrame(rows)
