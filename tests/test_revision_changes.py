"""Tests for the JCMM_1268 revision changes (new file). Run: pytest -q tests/"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("DATABASE_URL", "sqlite://")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pipeline  # noqa: E402
from revision.common import error_metrics, injected_windows, load_daily_demand, sha256  # noqa: E402


class Const:
    def __init__(self, v):
        self.v = v


def predict_const(model, df):
    return np.full(len(df), model.v, dtype=float)


def history(n=30, value=10.0, start="2025-07-01"):
    return pd.DataFrame({"Date": pd.date_range(start, periods=n), "Demand": np.full(n, value)})


def test_gate_accepts_better_candidate_and_never_fits_on_holdout():
    train = history()
    seen = {}

    def fit(df):
        seen["max_date"] = df["Date"].max()
        return Const(10.0)

    d = pipeline.mape_safety_gate(train, Const(20.0), pd.Timestamp("2025-06-30"), fit, predict_const)
    assert d["accepted"] and d["reason"] == "candidate_lower_mape"
    assert d["mape_candidate"] == 0.0 and d["mape_incumbent"] == pytest.approx(100.0)
    assert d["holdout_start"] == train["Date"].iloc[-7] and d["holdout_end"] == train["Date"].iloc[-1]
    assert seen["max_date"] < d["holdout_start"] == d["candidate_train_end"] + pd.Timedelta(days=1)


def test_gate_rejects_worse_candidate():
    d = pipeline.mape_safety_gate(history(), Const(10.0), pd.Timestamp("2025-06-30"),
                                  lambda df: Const(15.0), predict_const)
    assert not d["accepted"] and d["reason"] == "candidate_not_lower"


def test_gate_rejects_ties_strict_inequality():
    d = pipeline.mape_safety_gate(history(), Const(12.0), pd.Timestamp("2025-06-30"),
                                  lambda df: Const(8.0), predict_const)
    assert d["mape_candidate"] == pytest.approx(d["mape_incumbent"]) and not d["accepted"]


def test_gate_zero_denominator_keeps_incumbent():
    d = pipeline.mape_safety_gate(history(value=0.0), Const(1.0), pd.Timestamp("2025-06-30"),
                                  lambda df: Const(0.0), predict_const)
    assert not d["accepted"] and d["reason"] == "no_nonzero_holdout_days"
    assert d["n_holdout_nonzero"] == 0 and d["mape_candidate"] is None


def test_gate_refuses_incumbent_trained_on_holdout():
    train = history()
    d = pipeline.mape_safety_gate(train, Const(20.0), train["Date"].iloc[-2],
                                  lambda df: Const(10.0), predict_const)
    assert not d["accepted"] and d["reason"] == "incumbent_overlaps_holdout"


def test_rejection_starts_cooldown_but_keeps_baseline():
    det = pipeline.RollingDriftDetector(threshold=2.0, window=7, min_days=3, cooldown_days=7)
    day = pd.Timestamp("2025-07-01")
    for i in range(8):
        det.update("X", 10.0, 9.0, day + pd.Timedelta(days=i))
    base = det._baselines["X"]
    det.record_rejection("X", day + pd.Timedelta(days=8))
    assert det._baselines["X"] == base and len(det._errors["X"]) == 7
    status = det.update("X", 10.0, 9.0, day + pd.Timedelta(days=9))
    assert status["in_cooldown"]


def test_reconstructed_windows_match_editor_dates():
    w = injected_windows(load_daily_demand())
    spans = sorted({(s.date().isoformat(), e.date().isoformat())
                    for s, e in zip(w["Window_Start"], w["Window_End"])})
    assert spans == [("2024-03-05", "2024-03-17"), ("2024-09-18", "2024-09-30"),
                     ("2024-10-15", "2024-10-27"), ("2025-02-09", "2025-02-21")]
    assert len(w) == 48 and (w.groupby("Draw").size() == 12).all()
    assert "APPL-001" not in set(w["SKU"])


def test_error_metrics_definitions():
    m = error_metrics([10, 0, 5], [8, 1, 5])
    assert m["MAE"] == pytest.approx(1.0) and m["n_nonzero"] == 2
    assert m["MAPE_pct"] == pytest.approx(10.0) and m["Tol20_pct"] == pytest.approx(100.0)


def test_output_dir_run_leaves_archived_outputs_untouched(tmp_path, monkeypatch):
    processed = REPO / "data/processed"
    before = {p.name: sha256(p) for p in processed.glob("*.csv")}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "MODEL_TYPE", "baseline")
    out = tmp_path / "out"
    pipeline.run_pipeline("2024-03-01", "2024-03-20", history_start="2024-01-01", seed=1,
                          output_dir=str(out), inventory_file=str(processed / "inventory_master.csv"))
    after = {p.name: sha256(p) for p in processed.glob("*.csv")}
    assert before == after
    m = pd.read_csv(out / "metrics.csv")
    assert m["Date"].min() == "2024-03-01" and m["SKU"].nunique() == 72
    assert (out / "gate_decisions.csv").exists() and (out / "system_events.csv").exists()
    inv = pd.read_csv(out / "inventory_master.csv")
    assert pd.to_numeric(inv["Current_Stock"], errors="coerce").notna().all()
