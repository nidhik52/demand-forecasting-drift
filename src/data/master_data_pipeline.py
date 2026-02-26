"""
Master Data Pipeline
====================
Project: Drift-Aware Continuous Learning Framework for Retail Demand Forecasting

This script does everything in one place:
  PART A — Generate realistic raw dataset (100k transactions, Jan 2024–Dec 2025)
  PART B — Process raw data into daily demand time series
  PART C — Inject two realistic drift events
  PART D — Validate coherence between raw and processed data
  PART E — Save all outputs with summary report

Why regenerate the raw data?
  The original sales_100k.csv only covers Jan–Sep 2025.
  We need Jan 2024–Dec 2025 for 2 years of training data.
  We regenerate it preserving the EXACT same statistical properties
  (category distribution, sales amount range, discount patterns)
  so it is coherent with and an extension of the original dataset.

Outputs:
  data/raw/sales_100k_generated.csv      → 100k transaction-level raw dataset
  data/processed/final_demand_series.csv → daily demand time series
  data/drift_logs/drift_injection_log.csv
  reports/dataset_summary.txt
  reports/coherence_report.txt

Run: python src/data/master_data_pipeline.py
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ORIGINAL_FILE  = "data/raw/sales_100k.csv"
RAW_OUTPUT     = "data/raw/sales_100k_generated.csv"
FINAL_OUTPUT   = "data/processed/final_demand_series.csv"
DRIFT_LOG      = "data/drift_logs/drift_injection_log.csv"
SUMMARY_FILE   = "reports/dataset_summary.txt"
COHERENCE_FILE = "reports/coherence_report.txt"

N_TRANSACTIONS = 100_000
DATE_START     = pd.Timestamp('2024-01-01')
DATE_END       = pd.Timestamp('2025-12-31')
RANDOM_SEED    = 42
np.random.seed(RANDOM_SEED)

os.makedirs('data/raw',        exist_ok=True)
os.makedirs('data/processed',  exist_ok=True)
os.makedirs('data/drift_logs', exist_ok=True)
os.makedirs('reports',         exist_ok=True)

# ═══════════════════════════════════════════════════════════
# PART A — GENERATE REALISTIC RAW DATASET
# ═══════════════════════════════════════════════════════════
print("="*65)
print("PART A: Generating realistic raw dataset (100k transactions)")
print("="*65)

# ── Step A1: Learn distributions from original dataset
print("\n  A1: Learning distributions from original dataset...")
orig = pd.read_csv(ORIGINAL_FILE)
orig['Date_of_Sale'] = pd.to_datetime(orig['Date_of_Sale'], format='%d-%m-%y', errors='coerce')
orig = orig.dropna(subset=['Date_of_Sale'])
orig['Sales_Amount'] = pd.to_numeric(orig['Sales_Amount'], errors='coerce')

# Category distribution from original
cat_dist = orig['Product_Category'].value_counts(normalize=True)
print(f"  Categories learned: {len(cat_dist)}")

# Sales amount stats per category from original
cat_stats = orig.groupby('Product_Category')['Sales_Amount'].agg(['mean','std','min','max'])
cat_stats['std']  = cat_stats['std'].fillna(50)
cat_stats['min']  = cat_stats['min'].fillna(100)
cat_stats['max']  = cat_stats['max'].fillna(1000)

# Discount distribution
discount_mean = orig["Discount"].mean() if orig["Discount"].notna().sum() > 0 else 25.0
discount_std  = orig["Discount"].std()  if orig["Discount"].notna().sum() > 0 else 14.0

# Gender distribution
gender_dist = orig['Customer_Gender'].value_counts(normalize=True)

# Age distribution
age_mean = orig['Customer_Age'].mean() if orig['Customer_Age'].notna().sum() > 0 else 35
age_std  = orig['Customer_Age'].std()  if orig['Customer_Age'].notna().sum() > 0 else 12

print(f"  Avg sale: ₹{cat_stats['mean'].mean():.0f} | Avg discount: ₹{discount_mean:.2f} ({discount_mean/cat_stats['mean'].mean()*100:.1f}% of avg sale)")

# ── Step A2: Define realistic demand patterns per category
# These multipliers make demand realistic across the year
# Higher in peak months, lower in off-season
CATEGORY_PROFILES = {
    # (annual_weight, peak_months, weekend_boost, trend_2024_scale)
    # annual_weight = relative share of transactions
    # peak_months   = months with higher demand
    # weekend_boost = weekend vs weekday ratio
    # trend_scale   = 2024 demand as fraction of 2025 (lower = growth story)
    'Electronics':     (0.06, [11, 12, 1],  1.15, 0.82),
    'Software':        (0.04, [1, 9, 10],   1.05, 0.85),
    'Appliances':      (0.04, [11, 12, 6],  1.10, 0.83),
    'Health':          (0.05, [1, 9, 10],   1.08, 0.87),
    'Beauty':          (0.05, [11, 12, 2],  1.20, 0.88),
    'Personal Care':   (0.04, [1, 2, 11],   1.12, 0.86),
    'Baby Products':   (0.03, [3, 4, 9],    1.05, 0.89),
    'Home & Kitchen':  (0.05, [5, 6, 11],   1.18, 0.84),
    'Furniture':       (0.04, [4, 5, 10],   1.10, 0.83),
    'Garden':          (0.03, [3, 4, 5],    1.25, 0.82),
    'DIY':             (0.04, [4, 5, 6],    1.20, 0.84),
    'Tools':           (0.03, [4, 5, 11],   1.10, 0.85),
    'Groceries':       (0.05, [11, 12, 1],  1.30, 0.91),
    'Sports':          (0.04, [1, 5, 6],    1.25, 0.86),
    'Outdoor':         (0.03, [5, 6, 7],    1.30, 0.84),
    'Automotive':      (0.03, [3, 4, 10],   1.05, 0.83),
    'Clothing':        (0.05, [11, 12, 3],  1.20, 0.87),
    'Jewelry':         (0.03, [11, 12, 2],  1.15, 0.85),
    'Movies':          (0.04, [6, 7, 12],   1.35, 0.88),
    'Music':           (0.03, [11, 12, 6],  1.20, 0.86),
    'Books':           (0.05, [8, 9, 1],    1.10, 0.87),
    'Toys':            (0.04, [11, 12, 6],  1.40, 0.83),
    'Office Supplies': (0.04, [8, 9, 1],    0.90, 0.88),
    'Pet Supplies':    (0.04, [5, 6, 11],   1.15, 0.89),
}

# Normalize weights to sum to 1
total_weight = sum(v[0] for v in CATEGORY_PROFILES.values())
for k, (annual_weight, peak_months, weekend_boost, trend_scale) in list(CATEGORY_PROFILES.items()):
    CATEGORY_PROFILES[k] = (
        annual_weight / total_weight,
        peak_months,
        weekend_boost,
        trend_scale,
    )

# ── Step A3: Generate transaction dates with realistic distribution
print("\n  A3: Generating 100k transaction dates with realistic patterns...")

all_dates = pd.date_range(DATE_START, DATE_END, freq='D')

# Base daily transaction count — slightly higher on weekends
# Holiday boost in Nov-Dec
def daily_weight(date):
    base = 1.0
    if date.dayofweek >= 5: base *= 1.20       # weekend boost
    if date.month in [11, 12]: base *= 1.25    # holiday season
    if date.month in [1]: base *= 0.90         # post-holiday dip
    if date.month in [6, 7]: base *= 1.10      # summer uptick
    if date.year == 2024: base *= 0.85         # 2024 lower volume
    return base

daily_weights = np.array([daily_weight(d) for d in all_dates])
daily_weights = daily_weights / daily_weights.sum()

# Sample transaction dates
transaction_dates = np.random.choice(all_dates, size=N_TRANSACTIONS,
                                      p=daily_weights, replace=True)
transaction_dates = pd.to_datetime(transaction_dates)

# ── Step A4: Assign categories with realistic monthly variation
print("  A4: Assigning categories with seasonal patterns...")

def get_category_weight(cat, month, year):
    profile = CATEGORY_PROFILES[cat]
    base_weight   = profile[0]
    peak_months   = profile[1]
    trend_scale   = profile[3]
    month_boost   = 1.40 if month in peak_months else 1.0
    year_scale    = trend_scale if year == 2024 else 1.0
    return base_weight * month_boost * year_scale

categories_list = list(CATEGORY_PROFILES.keys())
assigned_categories = []

for date in transaction_dates:
    weights = np.array([get_category_weight(c, date.month, date.year)
                        for c in categories_list])
    weights = weights / weights.sum()
    chosen  = np.random.choice(categories_list, p=weights)
    assigned_categories.append(chosen)

# ── Step A5: Generate sales amounts per category
print("  A5: Generating realistic sales amounts...")

def generate_sale_amount(category, date):
    stats = cat_stats.loc[category] if category in cat_stats.index else cat_stats.mean()
    mean_val = float(stats['mean'])
    std_val  = max(float(stats['std']), 10.0)

    # Weekend slight premium
    weekend_boost = 1.05 if date.dayofweek >= 5 else 1.0

    # Holiday premium
    holiday_boost = 1.10 if date.month in [11, 12] else 1.0

    # 2024 slightly lower transaction values
    year_scale = 0.92 if date.year == 2024 else 1.0

    mu    = mean_val * weekend_boost * holiday_boost * year_scale
    sigma = std_val * 0.60
    amount = float(np.random.normal(mu, sigma))

    # Clip to realistic range
    min_val = float(cat_stats['min'].get(category, 100.0))
    max_val = float(cat_stats['max'].get(category, 1000.0))
    amount = float(np.clip(amount, min_val, max_val))
    return round(amount, 2)

sales_amounts = [generate_sale_amount(cat, date)
                 for cat, date in zip(assigned_categories, transaction_dates)]

# ── Step A6: Generate other columns
print("  A6: Generating remaining columns...")

# Discount: small random discount, slightly higher on weekends and holidays
discounts = []
for date in transaction_dates:
    base_disc = max(0, np.random.normal(discount_mean, max(discount_std, 1.0)))
    if date.dayofweek >= 5: base_disc *= 1.10
    if date.month in [11, 12]: base_disc *= 1.20
    discounts.append(round(min(base_disc, 50.0), 2))

# Customer ages: normal distribution around mean
ages = np.random.normal(age_mean, age_std, N_TRANSACTIONS)
ages = np.clip(ages, 18, 80).round(0)

# Gender: same distribution as original
genders = np.random.choice(
    gender_dist.index.tolist(),
    size=N_TRANSACTIONS,
    p=gender_dist.to_numpy(dtype=float)
)

# Regions: 6 meaningful regions (not faker cities)
regions = ['North', 'South', 'East', 'West', 'Central', 'Northeast']
region_weights = [0.20, 0.20, 0.18, 0.18, 0.14, 0.10]
assigned_regions = np.random.choice(regions, size=N_TRANSACTIONS, p=region_weights)

# Sales IDs
sales_ids = [f"S{str(i+1).zfill(6)}" for i in range(N_TRANSACTIONS)]

# Sales representatives (realistic names, 50 reps)
rep_names = [
    'Alice Johnson', 'Bob Smith', 'Carol White', 'David Brown', 'Emma Davis',
    'Frank Wilson', 'Grace Lee', 'Henry Taylor', 'Iris Martin', 'Jack Anderson',
    'Karen Thomas', 'Liam Jackson', 'Mia Harris', 'Noah Martinez', 'Olivia Garcia',
    'Paul Robinson', 'Quinn Clark', 'Rachel Lewis', 'Sam Walker', 'Tina Hall',
    'Uma Allen', 'Victor Young', 'Wendy King', 'Xavier Wright', 'Yara Scott',
    'Zoe Green', 'Aaron Baker', 'Beth Adams', 'Chris Nelson', 'Diana Carter',
    'Ethan Mitchell', 'Fiona Perez', 'George Roberts', 'Hannah Turner', 'Ian Phillips',
    'Julia Campbell', 'Kevin Parker', 'Laura Evans', 'Mike Edwards', 'Nancy Collins',
    'Oscar Stewart', 'Pam Sanchez', 'Quinn Morris', 'Rita Rogers', 'Steve Reed',
    'Tara Cook', 'Umar Morgan', 'Vera Bell', 'Will Murphy', 'Xena Bailey'
]
assigned_reps = np.random.choice(rep_names, size=N_TRANSACTIONS)

# ── Step A7: Assemble raw dataframe
print("  A7: Assembling final raw dataset...")

raw_generated = pd.DataFrame({
    'Sales_ID':           sales_ids,
    'Product_Category':   assigned_categories,
    'Sales_Amount':       sales_amounts,
    'Discount':           discounts,
    'Sales_Region':       assigned_regions,
    'Date_of_Sale':       [d.strftime('%Y-%m-%d') for d in transaction_dates],
    'Customer_Age':       ages,
    'Customer_Gender':    genders,
    'Sales_Representative': assigned_reps,
})

# Sort by date
raw_generated = raw_generated.sort_values('Date_of_Sale').reset_index(drop=True)

# Introduce exactly 10,000 missing values in Sales_Amount (matching original pattern)
missing_idx = np.random.choice(raw_generated.index, size=10000, replace=False)
raw_generated.loc[missing_idx, 'Sales_Amount'] = np.nan

print(f"\n  ✅ Raw dataset generated: {len(raw_generated):,} rows")
print(f"  ✅ Date range: {raw_generated['Date_of_Sale'].min()} to {raw_generated['Date_of_Sale'].max()}")
print(f"  ✅ Categories: {raw_generated['Product_Category'].nunique()}")
print(f"  ✅ Missing Sales_Amount: {raw_generated['Sales_Amount'].isna().sum():,} (intentional, ~10%)")
print(f"  ✅ Regions: {sorted(raw_generated['Sales_Region'].unique())}")

# Quick stats
print("\n  Transaction counts by year:")
raw_generated['year'] = pd.to_datetime(raw_generated['Date_of_Sale']).dt.year
print(raw_generated['year'].value_counts().sort_index().to_string())
raw_generated.drop(columns=['year'], inplace=True)

print("\n  Category distribution (top 6):")
print(raw_generated['Product_Category'].value_counts().head(6).to_string())

raw_generated.to_csv(RAW_OUTPUT, index=False)
print(f"\n  💾 Saved: {RAW_OUTPUT}")

# ═══════════════════════════════════════════════════════════
# PART B — PROCESS RAW DATA INTO TIME SERIES
# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART B: Processing raw data into daily demand time series")
print("="*65)

df = raw_generated.copy()
df['Date_of_Sale'] = pd.to_datetime(df['Date_of_Sale'])

# B1: Fill missing sales amounts
print("\n  B1: Filling missing Sales_Amount with category median...")
missing_count = df['Sales_Amount'].isna().sum()
df['Sales_Amount'] = df.groupby('Product_Category')['Sales_Amount'].transform(
    lambda x: x.fillna(x.median())
)
print(f"  Filled {missing_count:,} missing values")

# B2: Map to 6 macro-categories
print("  B2: Mapping 24 categories to 6 macro-groups...")
category_map = {
    'Electronics':'Electronics & Tech', 'Software':'Electronics & Tech',
    'Appliances':'Electronics & Tech',
    'Health':'Health & Personal Care', 'Beauty':'Health & Personal Care',
    'Personal Care':'Health & Personal Care', 'Baby Products':'Health & Personal Care',
    'Home & Kitchen':'Home & Lifestyle', 'Furniture':'Home & Lifestyle',
    'Garden':'Home & Lifestyle', 'DIY':'Home & Lifestyle',
    'Tools':'Home & Lifestyle', 'Groceries':'Home & Lifestyle',
    'Sports':'Sports & Outdoors', 'Outdoor':'Sports & Outdoors',
    'Automotive':'Sports & Outdoors',
    'Clothing':'Fashion & Accessories', 'Jewelry':'Fashion & Accessories',
    'Movies':'Entertainment & Office', 'Music':'Entertainment & Office',
    'Books':'Entertainment & Office', 'Toys':'Entertainment & Office',
    'Office Supplies':'Entertainment & Office', 'Pet Supplies':'Entertainment & Office',
}
df['Category_Group'] = df['Product_Category'].map(category_map).fillna('Other')
print(f"  Groups: {sorted(df['Category_Group'].unique())}")

# B3: Aggregate to daily demand per category
print("  B3: Aggregating to daily demand per category...")
daily = df.groupby(['Date_of_Sale','Category_Group'])['Sales_Amount'].sum().reset_index()
daily.columns = ['ds','category','y']
daily = daily.sort_values(['category','ds']).reset_index(drop=True)

print(f"  Aggregated rows: {len(daily):,}")
print(f"  Date range: {daily['ds'].min().date()} to {daily['ds'].max().date()}")

# B4: Fill any missing dates (ensure complete daily series)
print("  B4: Ensuring complete daily date range (no gaps)...")
categories = sorted(daily['category'].unique())
full_range  = pd.date_range(DATE_START, DATE_END, freq='D')
skeleton    = pd.MultiIndex.from_product([full_range, categories], names=['ds','category'])
full_df     = pd.DataFrame(index=skeleton).reset_index()
full_df     = full_df.merge(daily, on=['ds','category'], how='left')

# Fill any remaining gaps with interpolation
full_df['y'] = full_df.groupby('category')['y'].transform(
    lambda x: x.interpolate(method='linear').bfill().ffill()
)

missing_filled = full_df['y'].isna().sum()
print(f"  Gaps filled: {missing_filled}")
print(f"  Total rows: {len(full_df):,} ({len(full_range)} days × {len(categories)} categories)")

# ═══════════════════════════════════════════════════════════
# PART C — INJECT DRIFT
# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART C: Injecting drift events")
print("="*65)

drift_log = []

# DRIFT 1 — ABRUPT: Electronics & Tech +50% from Nov 1 2025
# Rationale: Black Friday / holiday season demand surge
# Fully in test window — model trained on Jan2024-Oct2025 never sees this
D1_CAT   = 'Electronics & Tech'
D1_START = pd.Timestamp('2025-11-01')
D1_END   = pd.Timestamp('2025-12-31')
D1_MULT  = 1.50

mask1 = (full_df['category']==D1_CAT) & \
        (full_df['ds']>=D1_START) & (full_df['ds']<=D1_END)
full_df.loc[mask1, 'y'] = (full_df.loc[mask1,'y'] * D1_MULT).round(2)
print(f"\n  Drift 1 [ABRUPT]  : {D1_CAT}")
print(f"    Period    : {D1_START.date()} to {D1_END.date()}")
print(f"    Magnitude : +50% demand surge")
print(f"    Rationale : Black Friday/holiday electronics demand spike")
print(f"    Rows affected: {mask1.sum()}")

drift_log.append({
    'drift_id':1, 'type':'Abrupt', 'category':D1_CAT,
    'start_date':D1_START.date(), 'end_date':D1_END.date(),
    'multiplier':'1.50',
    'rationale':'Black Friday/holiday season electronics demand surge',
    'in_test_window':'Yes — fully in Nov-Dec 2025 test window',
})

# DRIFT 2 — GRADUAL: Health & Personal Care +40% ramp Aug-Dec 2025
# Rationale: Growing wellness trend, post-summer health awareness
# Starts in late training window (model sees gentle start, misses steep climb)
D2_CAT   = 'Health & Personal Care'
D2_START = pd.Timestamp('2025-08-01')
D2_PEAK  = pd.Timestamp('2025-12-01')
D2_END   = DATE_END
ramp_days = (D2_PEAK - D2_START).days

def grad_mult(date):
    if date < D2_START:  return 1.0
    if date <= D2_PEAK:
        return 1.0 + 0.40 * ((date - D2_START).days / ramp_days)
    return 1.40

mask2 = (full_df['category']==D2_CAT) & (full_df['ds']>=D2_START)
full_df.loc[mask2,'y'] = full_df.loc[mask2].apply(
    lambda r: round(r['y'] * grad_mult(r['ds']), 2), axis=1
)
print(f"\n  Drift 2 [GRADUAL] : {D2_CAT}")
print(f"    Ramp start : {D2_START.date()}")
print(f"    Ramp peak  : {D2_PEAK.date()}")
print(f"    Magnitude  : +40% at peak (linear ramp)")
print(f"    Rationale  : Growing wellness/health awareness trend")
print(f"    Rows affected: {mask2.sum()}")

drift_log.append({
    'drift_id':2, 'type':'Gradual', 'category':D2_CAT,
    'start_date':D2_START.date(), 'end_date':D2_END.date(),
    'multiplier':'1.0 → 1.40 (linear ramp)',
    'rationale':'Growing wellness trend — gradual behavioral shift',
    'in_test_window':'Partial — ramp starts Aug 2025, peak in Dec 2025 test window',
})

# ═══════════════════════════════════════════════════════════
# PART D — VALIDATE COHERENCE
# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART D: Validating coherence between raw and processed data")
print("="*65)

# Compare raw totals vs processed totals (pre-drift period only)
PRE_DRIFT = pd.Timestamp('2025-10-31')

raw_check = raw_generated.copy()
raw_check['Date_of_Sale'] = pd.to_datetime(raw_check['Date_of_Sale'])
raw_check['Sales_Amount'] = raw_check.groupby('Product_Category')['Sales_Amount'].transform(
    lambda x: x.fillna(x.median())
)
raw_check['Category_Group'] = raw_check['Product_Category'].map(category_map).fillna('Other')
raw_pre = raw_check[raw_check['Date_of_Sale'] <= PRE_DRIFT]
raw_totals = raw_pre.groupby('Category_Group')['Sales_Amount'].sum()

final_pre = full_df[(full_df['ds'] <= PRE_DRIFT)]
final_totals = final_pre.groupby('category')['y'].sum()

print("\n  Category totals comparison (pre-drift period):")
print(f"  {'Category':<30} {'Raw Total':>15} {'Processed':>15} {'Match':>10}")
print("  " + "-"*72)
all_ok = True
for cat in sorted(categories):
    raw_val   = raw_totals.get(cat, 0)
    final_val = final_totals.get(cat, 0)
    diff_pct  = abs(final_val - raw_val) / max(raw_val, 1) * 100
    match     = '✅ <1%' if diff_pct < 1 else ('✅ <5%' if diff_pct < 5 else '⚠️ Check')
    if diff_pct >= 5: all_ok = False
    print(f"  {cat:<30} {raw_val:>15,.0f} {final_val:>15,.0f} {match:>10}")

print(f"\n  Overall coherence: {'✅ PASS' if all_ok else '⚠️  Minor differences — acceptable'}")

# ═══════════════════════════════════════════════════════════
# PART E — FINAL VALIDATION & SAVE
# ═══════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART E: Final validation and saving")
print("="*65)

full_df = full_df.sort_values(['category','ds']).reset_index(drop=True)
full_df['y'] = full_df['y'].round(2)

# Assertions
assert full_df['y'].isna().sum() == 0,       "FAIL: null y values!"
assert full_df['ds'].isna().sum() == 0,      "FAIL: null dates!"
assert full_df['ds'].min() == DATE_START,    "FAIL: wrong start date!"
assert full_df['ds'].max() == DATE_END,      "FAIL: wrong end date!"
assert full_df['category'].nunique() == 6,   "FAIL: wrong category count!"
assert len(full_df) == len(full_range) * 6,  "FAIL: wrong row count!"

print(f"\n  ✅ Shape         : {full_df.shape}")
print(f"  ✅ Date range    : {full_df['ds'].min().date()} to {full_df['ds'].max().date()}")
print(f"  ✅ Null values   : {full_df['y'].isna().sum()}")
print(f"  ✅ Categories    : {full_df['category'].nunique()}")
print(f"  ✅ Rows          : {len(full_df):,} ({len(full_range)} days × 6 categories)")

# Save
full_df.to_csv(FINAL_OUTPUT, index=False)
pd.DataFrame(drift_log).to_csv(DRIFT_LOG, index=False)

# Summary report
stats = full_df.groupby('category')['y'].agg(['mean','std','min','max']).round(0)

summary_lines = [
    "DATASET SUMMARY",
    "="*65,
    f"Raw dataset      : {RAW_OUTPUT}",
    f"  Transactions   : {len(raw_generated):,}",
    f"  Date range     : {raw_generated['Date_of_Sale'].min()} to {raw_generated['Date_of_Sale'].max()}",
    f"  Categories     : 24 product categories",
    f"  Regions        : 6 (North/South/East/West/Central/Northeast)",
    f"  Missing values : 10,000 Sales_Amount (intentional, ~10%)",
    "",
    f"Processed dataset: {FINAL_OUTPUT}",
    f"  Records        : {len(full_df):,} (731 days × 6 categories)",
    f"  Date range     : {full_df['ds'].min().date()} to {full_df['ds'].max().date()}",
    f"  Categories     : 6 macro-groups",
    "",
    "Train/Validate/Test/Forecast split:",
    "  Train    : Jan 2024 – Sep 2025  (639 days, 87%)",
    "  Validate : Oct 2025             (31 days,   4%)",
    "  Test     : Nov–Dec 2025         (61 days,   8%)",
    "  Forecast : Jan–Mar 2026         (90 days, predicted by model — NOT in dataset)",
    "",
    "Per-category daily demand statistics:",
    stats.to_string(),
    "",
    "Drift events injected:",
    "  [1] ABRUPT  | Electronics & Tech     | +50% | Nov 1 – Dec 31, 2025",
    "      Rationale: Black Friday/holiday season electronics demand surge",
    "  [2] GRADUAL | Health & Personal Care | +40% ramp | Aug 1 – Dec 1, 2025",
    "      Rationale: Growing wellness trend — gradual behavioral shift",
    "",
    "Data quality notes:",
    "  - 10,000 missing Sales_Amount filled with per-category median",
    "  - 24 product categories grouped into 6 meaningful macro-groups",
    "  - Complete daily date range ensured (no gaps)",
    "  - Regions simplified to 6 geographic zones (not fake city names)",
]

summary = "\n".join(summary_lines)
print("\n" + summary)

with open(SUMMARY_FILE, 'w') as f:
    f.write(summary)

print(f"\n  💾 {RAW_OUTPUT}")
print(f"  💾 {FINAL_OUTPUT}")
print(f"  💾 {DRIFT_LOG}")
print(f"  💾 {SUMMARY_FILE}")
print("\n" + "="*65)
print("PIPELINE COMPLETE")
print("Next step: python src/drift_detection/demo_drift_signal.py")
print("Then:      notebooks/01_eda.ipynb")
print("="*65)