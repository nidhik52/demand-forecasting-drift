import pandas as pd
import os
from datetime import datetime


EVENT_FILE = "PROJECT ROOT / data/processed/system_events.csv"


def log_event(event_type, message, date):

    os.makedirs("data/processed", exist_ok=True)

    if os.path.exists(EVENT_FILE):
        df = pd.read_csv(EVENT_FILE)
    else:
        df = pd.DataFrame(columns=["timestamp", "date", "event_type", "message"])

    new_row = {
        "timestamp": date,   # ✅ FIXED (NOT datetime.now)
        "date": date,
        "event_type": event_type,
        "message": message
    }

    df = pd.concat([df, pd.DataFrame([new_row])])

    df.to_csv(EVENT_FILE, index=False)