import pandas as pd
from datetime import datetime
from pathlib import Path

EVENT_LOG_FILE = Path("data/processed/system_events.csv")


def log_event(event_type, message, event_time=None):

    EVENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if event_time is None:
        event_time = datetime.now()

    log_row = {
        "timestamp": event_time.strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "message": message
    }

    try:
        df = pd.read_csv(EVENT_LOG_FILE)
    except:
        df = pd.DataFrame(columns=["timestamp", "event_type", "message"])

    df = pd.concat([df, pd.DataFrame([log_row])], ignore_index=True)

    df.to_csv(EVENT_LOG_FILE, index=False)

    print(f"[{log_row['timestamp']}] {event_type} → {message}")