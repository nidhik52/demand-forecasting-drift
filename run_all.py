import subprocess
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
START_DATE = "2025-07-01"
END_DATE = "2025-07-05"

RUN_PREPROCESSING = True
RUN_INVENTORY_INIT = True
RUN_API = False

PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------
# HELPERS
# -----------------------------
def run_command(command):
    print(f"\n🚀 Running: {command}\n")
    result = subprocess.run(command, shell=True, cwd=PROJECT_ROOT)
    
    if result.returncode != 0:
        print(f"❌ Error running: {command}")
        exit(1)


# -----------------------------
# STEP 1: PREPROCESSING (OPTIONAL)
# -----------------------------
if RUN_PREPROCESSING:
    run_command("python -m src.preprocessing")

# -----------------------------
# STEP 2: INVENTORY INIT (OPTIONAL)
# -----------------------------
if RUN_INVENTORY_INIT:
    run_command("python -m src.generate_inventory")

# -----------------------------
# STEP 3: RUN PIPELINE
# -----------------------------
run_command(f"python pipeline.py --start {START_DATE} --end {END_DATE}")

if RUN_API:
    print("\n🌐 Starting FastAPI server...\n")
    run_command("uvicorn api:app --reload")
