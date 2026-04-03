import pandas as pd
from datetime import datetime
from src.config import EVENT_LOG_FILE


def log_event(event_type: str, message: str, event_time=None, date=None):
    """Append a single event to the `EVENT_LOG_FILE`.

    Accepts either `event_time` or `date` (legacy callers use both names).
    Writes rows with columns: `Timestamp,Event_Type,Message`.
    """

    ts = event_time or date or datetime.now()
    ts = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")

    # ensure parent directory exists
    EVENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if EVENT_LOG_FILE.exists():
        df = pd.read_csv(EVENT_LOG_FILE)
    else:
        df = pd.DataFrame(columns=["Timestamp", "Event_Type", "Message"])

    new_row = {"Timestamp": ts, "Event_Type": event_type, "Message": message}

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(EVENT_LOG_FILE, index=False)