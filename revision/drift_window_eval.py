"""Scores a pipeline event log against the seed-42 injected drift windows
(JCMM_1268 revision, new file). Definitions are fixed in
revision/preregistered_drift_eval.json and were written before any run.

Usage:
  python revision/drift_window_eval.py --run_dir data/revision/pipeline_drift_2024-02-01_2025-03-31
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from revision.common import injected_windows, load_daily_demand

PREREG = Path(__file__).resolve().parent / "preregistered_drift_eval.json"


def load_events(run_dir):
    ev = pd.read_csv(Path(run_dir) / "system_events.csv")
    ev["Date"] = pd.to_datetime(ev["timestamp"]).dt.normalize()
    ev["SKU"] = ev["message"].str.extract(r"^(\S+)")[0]
    return ev


def evaluate(events, windows, run_start, run_end, all_skus, grace):
    drift = events[events["event_type"] == "DRIFT"][["SKU", "Date"]].drop_duplicates()
    windows = windows.copy()
    windows["Interval_End"] = windows["Window_End"] + pd.Timedelta(days=grace)

    pair_rows = []
    for w in windows.itertuples():
        hits = drift[(drift["SKU"] == w.SKU) & drift["Date"].between(w.Window_Start, w.Interval_End)]
        first = hits["Date"].min() if len(hits) else pd.NaT
        pair_rows.append({"Draw": w.Draw, "SKU": w.SKU, "Window_Start": w.Window_Start.date(),
                          "Window_End": w.Window_End.date(), "Interval_End": w.Interval_End.date(),
                          "Multiplier": round(w.Multiplier, 4), "Detected": bool(len(hits)),
                          "First_Alert": first.date() if len(hits) else None,
                          "Delay_Days": int((first - w.Window_Start).days) if len(hits) else None,
                          "Alert_Days_In_Interval": int(len(hits))})
    pairs = pd.DataFrame(pair_rows)

    def in_any_interval(row):
        w = windows[windows["SKU"] == row.SKU]
        return bool(((row.Date >= w["Window_Start"]) & (row.Date <= w["Interval_End"])).any())

    drift = drift.assign(In_Interval=[in_any_interval(r) for r in drift.itertuples()])
    fa = drift[~drift["In_Interval"]].sort_values(["SKU", "Date"])
    # episodes = maximal runs of consecutive dates per SKU
    episodes = int((fa.groupby("SKU")["Date"].diff().dt.days.fillna(99) != 1).sum()) if len(fa) else 0

    days = pd.date_range(run_start, run_end, freq="D")
    injected_skudays = 0
    for w in windows.itertuples():
        injected_skudays += len(pd.date_range(max(w.Window_Start, days[0]), min(w.Interval_End, days[-1])))
    non_injected = len(all_skus) * len(days) - injected_skudays

    tp = int(pairs["Detected"].sum())
    delays = pairs.loc[pairs["Detected"], "Delay_Days"].astype(float)
    summary = {
        "injected_pairs": int(len(pairs)),
        "TP": tp, "FN": int(len(pairs) - tp),
        "pair_recall_pct": round(100 * tp / len(pairs), 2),
        "delay_days_min": None if delays.empty else float(delays.min()),
        "delay_days_median": None if delays.empty else float(delays.median()),
        "delay_days_mean": None if delays.empty else round(float(delays.mean()), 2),
        "delay_days_max": None if delays.empty else float(delays.max()),
        "drift_alert_days_total": int(len(drift)),
        "drift_alert_days_in_interval": int(drift["In_Interval"].sum()),
        "false_alarm_alert_days": int(len(fa)),
        "false_alarm_episodes": episodes,
        "false_alarm_skus": int(fa["SKU"].nunique()),
        "non_injected_sku_days": int(non_injected),
        "false_alarm_alert_days_per_1000_non_injected_sku_days": round(1000 * len(fa) / non_injected, 3),
        "alert_day_precision_pct": round(100 * drift["In_Interval"].mean(), 2) if len(drift) else None,
        "detected_by_draw": pairs.groupby("Draw")["Detected"].sum().astype(int).to_dict(),
    }
    return pairs, fa, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    prereg = json.loads(PREREG.read_text())
    run = prereg["run_period"]

    demand = load_daily_demand()
    windows = injected_windows(demand)
    events = load_events(args.run_dir)
    pairs, fa, summary = evaluate(events, windows, run["start"], run["end"],
                                  demand["SKU"].unique(), prereg["grace_days_after_window_end"])

    out = Path(args.run_dir) / "drift_eval"
    out.mkdir(exist_ok=True)
    windows.assign(Window_Start=windows["Window_Start"].dt.date, Window_End=windows["Window_End"].dt.date) \
        .to_csv(out / "injected_windows.csv", index=False)
    pairs.to_csv(out / "pair_detection.csv", index=False)
    fa.to_csv(out / "false_alarm_alert_days.csv", index=False)
    counts = events["event_type"].value_counts().to_dict()
    summary["event_counts"] = {k: int(v) for k, v in counts.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
