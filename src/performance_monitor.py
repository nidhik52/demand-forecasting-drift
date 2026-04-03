import pandas as pd

from pathlib import Path

from src.config import METRICS_FILE


def calculate_metrics(actual, predicted):
    return abs(actual - predicted)

def log_metrics(sku, actual, predicted, date):

    error = abs(actual - predicted)

    row = pd.DataFrame([{
        "Date": date,
        "SKU": sku,
        "Actual": actual,
        "Predicted": predicted,
        "Error": error
    }])

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if Path(METRICS_FILE).exists():
        row.to_csv(METRICS_FILE, mode="a", index=False, header=False)
    else:
        row.to_csv(METRICS_FILE, index=False)