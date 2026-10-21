"""Leakage-free rolling-origin evaluation (JCMM_1268 revision, new file).

For every target date d in [eval_start, eval_end] and every SKU, a Prophet
model is fitted with pipeline.train_prophet on that SKU's observations from
history_start through d-1 only, and predicts d (one-day-ahead forecast, the
same horizon as the pipeline's daily monitoring). Baselines use the same
(SKU, d) pairs: naive y(d-1), weekly seasonal-naive y(d-7), and the
pipeline's historical-mean model (pipeline.train_baseline) fitted on the same
training window. Nothing on or after d enters any fit.

Usage:
  python revision/rolling_origin_eval.py --eval_start 2025-07-03 --eval_end 2025-07-31 \
      --history_start 2024-01-01 --out data/revision/rolling_origin_jul2025
"""
import argparse
import json
import logging
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite://")  # importing pipeline must not touch inventory.db
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from revision.common import DAILY_DEMAND, error_metrics, load_daily_demand, sha256


def _fit_one(task):
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").setLevel(logging.ERROR)
    import pipeline
    sku, target, train = task
    t0 = time.perf_counter()
    model = pipeline.train_prophet(train)
    fit_s = time.perf_counter() - t0
    pred = float(pipeline.predict_prophet(model, pd.DataFrame({"Date": [target]}))[0])
    return {
        "SKU": sku, "Date": target, "Prophet": pred,
        "HistMean": float(pipeline.train_baseline(train)),
        "Train_Start": train["Date"].min(), "Train_End": train["Date"].max(),
        "N_Train": len(train), "Fit_Seconds": fit_s,
        "Seasonalities": ",".join(sorted(model.seasonalities.keys())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_start", default="2025-07-03")
    ap.add_argument("--eval_end", default="2025-07-31")
    ap.add_argument("--history_start", default="2024-01-01")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="data/revision/rolling_origin_jul2025")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    demand = load_daily_demand()
    demand = demand[demand["Date"] >= args.history_start].sort_values(["SKU", "Date"])
    targets = pd.date_range(args.eval_start, args.eval_end, freq="D")

    tasks = []
    for sku, g in demand.groupby("SKU"):
        for d in targets:
            train = g.loc[g["Date"] < d, ["Date", "Demand"]]  # strictly before the forecast date
            tasks.append((sku, d, train))

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_fit_one, tasks, chunksize=8))
    wall = time.perf_counter() - t0

    pred = pd.DataFrame(rows)
    lookup = demand.set_index(["SKU", "Date"])["Demand"]
    pred["Actual"] = [lookup[(s, d)] for s, d in zip(pred["SKU"], pred["Date"])]
    pred["Naive"] = [lookup[(s, d - pd.Timedelta(days=1))] for s, d in zip(pred["SKU"], pred["Date"])]
    pred["SeasonalNaive"] = [lookup[(s, d - pd.Timedelta(days=7))] for s, d in zip(pred["SKU"], pred["Date"])]
    assert (pred["Train_End"] < pred["Date"]).all(), "training data reached the forecast date"
    assert pred[["Prophet", "Naive", "SeasonalNaive", "HistMean", "Actual"]].notna().all().all()
    pred = pred.sort_values(["Date", "SKU"])
    pred.to_csv(out / "predictions.csv", index=False)

    summary = summarise(pred)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    per_sku = (pred.assign(AE=(pred["Actual"] - pred["Prophet"]).abs())
               .groupby("SKU")["AE"].mean().rename("Prophet_MAE").reset_index())
    per_sku.to_csv(out / "per_sku_prophet_mae.csv", index=False)

    import prophet, numpy, cmdstanpy
    manifest = {
        "script": "revision/rolling_origin_eval.py",
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "started_utc": started.isoformat(timespec="seconds"),
        "wall_seconds": round(wall, 1),
        "fits": len(pred),
        "mean_fit_seconds": round(float(pred["Fit_Seconds"].mean()), 4),
        "input_file": str(DAILY_DEMAND.relative_to(Path(__file__).resolve().parent.parent)),
        "input_sha256": sha256(DAILY_DEMAND),
        "history_start": args.history_start,
        "eval_window": [args.eval_start, args.eval_end],
        "horizon_days": 1,
        "training_cutoff": "all observations dated strictly before the forecast date",
        "model": "pipeline.train_prophet (Prophet daily/weekly/yearly seasonality='auto')",
        "effective_seasonalities": sorted(pred["Seasonalities"].unique().tolist()),
        "skus": int(pred["SKU"].nunique()),
        "python": platform.python_version(),
        "packages": {"prophet": prophet.__version__, "pandas": pd.__version__,
                     "numpy": numpy.__version__, "cmdstanpy": cmdstanpy.__version__},
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "workers": args.workers,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(manifest, indent=2))


def summarise(pred):
    rows = []
    for col, label in [("Prophet", "Prophet (rolling origin, 1-day ahead)"),
                       ("Naive", "Naive last value"),
                       ("SeasonalNaive", "Weekly seasonal naive"),
                       ("HistMean", "Historical mean (pipeline baseline model)")]:
        allm = error_metrics(pred["Actual"], pred[col])
        a = pred[pred["SKU"] == "APPL-001"]
        am = error_metrics(a["Actual"], a[col])
        rows.append({"Forecast": label, **{k: allm[k] for k in allm},
                     **{f"APPL001_{k}": am[k] for k in am}})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
