# ── Stage 1: Python backend ──────────────────────────────────
FROM python:3.11-slim AS backend

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-prophet.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prophet.txt

# Copy ALL application code + pre-seeded data
COPY api.py pipeline.py pipeline_baseline.py dashboard_server.py ./
COPY src/ ./src/
# BUG 1 FIX: copy pre-seeded processed data so API returns data immediately
COPY data/ ./data/

# ── Stage 2: React frontend ───────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm install --legacy-peer-deps

COPY dashboard/ ./
# CI=false prevents treating warnings as errors
RUN CI=false npm run build

# ── Stage 3: Final image ──────────────────────────────────────
FROM python:3.11-slim AS final

WORKDIR /app

# System deps needed at runtime (Prophet / cmdstan)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git curl \
    && rm -rf /var/lib/apt/lists/*

# Python packages
COPY requirements.txt requirements-prophet.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prophet.txt

# Pre-install cmdstan so first pipeline run doesn't time out
RUN python -c "import cmdstanpy; cmdstanpy.install_cmdstan(version='2.33.1', overwrite=False)" || true

# Copy backend code + data from backend stage
COPY --from=backend /app /app

# Copy React build from frontend stage
COPY --from=frontend /dashboard/build /app/dashboard/build

# Create writable directories for runtime output
RUN mkdir -p /app/data/processed /app/data/raw /app/models/prophet /app/models/baseline /app/mlruns

EXPOSE 8000

# BUG 10 FIX: use PORT env var (Render sets this automatically)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
