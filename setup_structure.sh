#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Project Structure Setup Script
# Run this ONCE inside your GitHub Codespace terminal
# Command: bash setup_structure.sh
# ─────────────────────────────────────────────────────────────

echo "Setting up project folder structure..."

# ── Data folders
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/drift_logs

# ── Notebooks (for EDA and experimentation)
mkdir -p notebooks

# ── Source code modules
mkdir -p src/data
mkdir -p src/forecasting
mkdir -p src/drift_detection
mkdir -p src/retraining
mkdir -p src/inventory
mkdir -p src/monitoring

# ── API backend
mkdir -p api

# ── Frontend dashboard
mkdir -p dashboard

# ── MLflow experiment tracking storage
mkdir -p mlruns

# ── Docker
mkdir -p docker

# ── GitHub Actions CI/CD
mkdir -p .github/workflows

# ── Tests
mkdir -p tests

# ── Reports and paper assets
mkdir -p reports/figures
mkdir -p reports/paper

# ── Create __init__.py files so src is a proper Python package
touch src/__init__.py
touch src/data/__init__.py
touch src/forecasting/__init__.py
touch src/drift_detection/__init__.py
touch src/retraining/__init__.py
touch src/inventory/__init__.py
touch src/monitoring/__init__.py

# ── Create placeholder files so folders show up in git
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/drift_logs/.gitkeep
touch reports/figures/.gitkeep
touch reports/paper/.gitkeep

echo ""
echo "✅ Folder structure created successfully!"
echo ""
echo "Project structure:"
find . -not -path './.git/*' -not -path './mlruns/*' | sort | head -60
