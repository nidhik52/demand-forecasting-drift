"""Regenerates manuscript Figures 2 and 3 from revision outputs (new file).

Figure 2: APPL-001 rolling 7-day MAE of the leakage-free rolling-origin Prophet
forecasts, 3-31 July 2025 (no event markers: the July run logged no APPL-001
event). Figure 3: risk-level counts of the 2,088 per-day assessments in the
July pipeline run.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/revision/figures"
INK, MUTED, GRID = "#111827", "#4b5563", "#e5e7eb"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9ca3af")
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure2():
    p = pd.read_csv(REPO / "data/revision/rolling_origin_jul2025/predictions.csv", parse_dates=["Date"])
    a = p[p["SKU"] == "APPL-001"].sort_values("Date").set_index("Date")
    ae = (a["Actual"] - a["Prophet"]).abs()
    roll = ae.rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9.49, 5.0), dpi=200)  # aspect of the Figure 2 frame in the manuscript
    ax.plot(roll.index, roll.values, color="#2563eb", linewidth=2, marker="o", markersize=4.5)
    ax.set_ylim(0, max(2.1, roll.max() * 1.15))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[3, 7, 11, 15, 19, 23, 27, 31]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Forecast date (each forecast uses data up to the previous day)", color=INK, fontsize=11)
    ax.set_ylabel("Rolling 7-day MAE (units/day)", color=INK, fontsize=11)
    ax.set_title("Figure 2. APPL-001 rolling 7-day MAE, July 3-31, 2025\n(leakage-free one-day-ahead Prophet forecasts)",
                 loc="left", fontsize=13, fontweight="bold", color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "figure2_appl001_rolling_mae.png")
    roll.rename("Rolling7_MAE").to_csv(OUT / "figure2_data.csv")


def figure3():
    iv = pd.read_csv(REPO / "data/revision/pipeline_july_2025-07-03_2025-07-31/inventory_recommendations.csv")
    order = ["SAFE", "WARNING", "CRITICAL"]
    colors = {"SAFE": "#16a34a", "WARNING": "#d97706", "CRITICAL": "#dc2626"}
    counts = iv["Risk_Level"].value_counts().reindex(order).fillna(0).astype(int)
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(9.38, 5.0), dpi=200)  # aspect of the Figure 3 frame in the manuscript
    bars = ax.bar(order, counts.values, width=0.55, color=[colors[k] for k in order], edgecolor="white", linewidth=2)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + total * 0.012, f"{v:,} ({100 * v / total:.1f}%)",
                ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_xlabel("Risk level", color=INK, fontsize=11)
    ax.set_ylabel("SKU-day assessments", color=INK, fontsize=11)
    ax.set_title(f"Figure 3. Inventory risk distribution across {total:,} daily SKU assessments",
                 loc="left", fontsize=13, fontweight="bold", color=INK)
    style(ax)
    ax.tick_params(axis="x", labelsize=11, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "figure3_risk_distribution.png")
    counts.rename("Count").to_csv(OUT / "figure3_data.csv")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure2()
    figure3()
    print("figures written to", OUT)
