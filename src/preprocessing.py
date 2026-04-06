import pandas as pd
import numpy as np
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DAILY_DEMAND_FILE, PROCESSED_DIR, RAW_SALES_FILE

RAW_DATA_FILE = RAW_SALES_FILE


# --------------------------------------------------
# Load raw dataset
# --------------------------------------------------
def load_data():

    print("📥 Loading raw dataset...")

    df = pd.read_csv(RAW_DATA_FILE)

    df["Date_of_Sale"] = pd.to_datetime(df["Date_of_Sale"])

    print(f"✅ Loaded {len(df)} rows")

    return df


# --------------------------------------------------
# Simplify noisy regions
# --------------------------------------------------
def clean_regions(df):

    print("🧹 Cleaning regions...")

    regions = ["North", "South", "East", "West", "Central"]

    df["Sales_Region"] = np.random.choice(regions, size=len(df))

    print("✅ Regions cleaned")

    return df


# --------------------------------------------------
# Generate synthetic quantity
# --------------------------------------------------
def synthesize_quantity(df):

    print("🔢 Generating synthetic quantity...")

    def generate_quantity(category):

        if category == "Electronics":
            return np.random.randint(1, 3)
        elif category == "Groceries":
            return np.random.randint(2, 6)
        elif category == "Clothing":
            return np.random.randint(1, 4)
        else:
            return np.random.randint(1, 5)

    df["Quantity"] = df["Product_Category"].apply(generate_quantity)

    print("✅ Quantity generated")

    return df


# --------------------------------------------------
# Create daily demand per SKU
# --------------------------------------------------
def create_daily_demand(df):

    print("📊 Creating daily demand per SKU...")

    demand = (
        df.groupby(["Date_of_Sale", "SKU", "SKU_Name"])["Quantity"]
        .sum()
        .reset_index()
    )

    demand = demand.rename(
        columns={
            "Date_of_Sale": "Date",
            "Quantity": "Demand"
        }
    )

    print(f"✅ Daily demand created ({len(demand)} rows)")

    return demand


# --------------------------------------------------
# Fill missing dates
# --------------------------------------------------
def fill_missing_dates(demand):

    print("📅 Filling missing dates...")

    demand["Date"] = pd.to_datetime(demand["Date"])

    skus = demand["SKU"].unique()

    start = demand["Date"].min()
    end = demand["Date"].max()

    full_dates = pd.date_range(start, end, freq="D")

    rows = []

    for sku in skus:

        sku_df = demand[demand["SKU"] == sku]

        name = sku_df["SKU_Name"].iloc[0]

        sku_df = sku_df.set_index("Date").reindex(full_dates)

        sku_df["Demand"] = sku_df["Demand"].fillna(0)

        sku_df["SKU"] = sku
        sku_df["SKU_Name"] = name

        sku_df = sku_df.reset_index().rename(columns={"index": "Date"})

        rows.append(sku_df)

    filled = pd.concat(rows)

    print("✅ Missing dates handled")

    return filled


# --------------------------------------------------
# Extend dataset to Dec 2025
# --------------------------------------------------
def extend_dataset_to_2025(demand):

    print("📆 Extending dataset to 2025...")

    demand["Date"] = pd.to_datetime(demand["Date"])

    max_date = demand["Date"].max()

    target_end = pd.Timestamp("2025-12-31")

    if max_date >= target_end:
        print("✅ Already extended")
        return demand

    missing_dates = pd.date_range(max_date + pd.Timedelta(days=1), target_end)

    sku_info = demand[["SKU", "SKU_Name"]].drop_duplicates()

    rows = []

    for date in missing_dates:

        for _, row in sku_info.iterrows():

            base = demand[demand["SKU"] == row["SKU"]]["Demand"].mean()

            value = int(base * np.random.uniform(0.8, 1.2))

            rows.append({
                "Date": date,
                "SKU": row["SKU"],
                "SKU_Name": row["SKU_Name"],
                "Demand": max(value, 1)
            })

    new_data = pd.DataFrame(rows)

    combined = pd.concat([demand, new_data]).sort_values("Date")

    print("✅ Dataset extended")

    return combined


# --------------------------------------------------
# Inject random drift windows
# --------------------------------------------------
def inject_random_drift(demand, seed=42, window_count=4, window_size=12):
    print("🧪 Injecting random drift windows...")

    rng = np.random.default_rng(seed)
    demand = demand.copy()
    demand["Date"] = pd.to_datetime(demand["Date"])

    min_date = demand["Date"].min()
    max_date = demand["Date"].max()

    if pd.isna(min_date) or pd.isna(max_date):
        return demand

    total_days = (max_date - min_date).days
    if total_days <= window_size:
        return demand

    sku_list = demand["SKU"].unique().tolist()
    if not sku_list:
        return demand

    for _ in range(window_count):
        start_offset = int(rng.integers(0, total_days - window_size))
        window_start = min_date + pd.Timedelta(days=start_offset)
        window_end = window_start + pd.Timedelta(days=window_size)

        drift_skus = rng.choice(sku_list, size=max(1, len(sku_list) // 6), replace=False)
        multiplier = float(rng.uniform(1.6, 2.4))

        mask = (
            demand["Date"].between(window_start, window_end)
            & demand["SKU"].isin(drift_skus)
        )

        demand.loc[mask, "Demand"] = (demand.loc[mask, "Demand"] * multiplier).round(0)

    demand["Demand"] = demand["Demand"].clip(lower=0)

    print("✅ Drift windows injected")

    return demand


# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------
def save_processed_data(demand):

    print("💾 Saving processed dataset...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    demand.to_csv(DAILY_DEMAND_FILE, index=False)

    print(f"✅ Saved → {DAILY_DEMAND_FILE}")


def preprocess_data(raw_df):
    df = clean_regions(raw_df.copy())
    df = synthesize_quantity(df)

    demand = create_daily_demand(df)
    demand = fill_missing_dates(demand)
    demand = extend_dataset_to_2025(demand)
    demand = inject_random_drift(demand)

    return demand


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    print("\n🚀 Starting preprocessing pipeline\n")

    df = load_data()
    df = clean_regions(df)
    df = synthesize_quantity(df)

    demand = create_daily_demand(df)
    demand = fill_missing_dates(demand)
    demand = extend_dataset_to_2025(demand)
    demand = inject_random_drift(demand)

    save_processed_data(demand)

    print("\n✅ Preprocessing completed successfully\n")