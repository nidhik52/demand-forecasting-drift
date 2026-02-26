# ─────────────────────────────────────────────────────────────
# Dockerfile — Drift-Aware Demand Forecasting API
# ─────────────────────────────────────────────────────────────
# FW7 IMPLEMENTATION — Containerised Deployment
#
# Builds a production-ready container for the FastAPI inference
# service. The container:
#   - Serves Prophet forecasts via REST API (/forecast, /drift, /inventory)
#   - Exposes MLflow tracking at /mlruns (volume-mounted)
#   - Runs as a non-root user for security
#
# BUILD
#   docker build -t demand-forecasting-api -f docker/Dockerfile .
#
# RUN (single container)
#   docker run -p 8000:8000 -v $(pwd)/mlruns:/app/mlruns demand-forecasting-api
#
# USE docker-compose (recommended — starts API + MLflow UI together)
#   docker compose -f docker/docker-compose.yml up
# ─────────────────────────────────────────────────────────────

# --- Stage 1: Builder ---
# Install heavy dependencies (prophet, cmdstanpy) in a throwaway layer
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for prophet / cmdstanpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/        ./src/
COPY api/        ./api/
COPY data/processed/ ./data/processed/
COPY mlruns/     ./mlruns/

# Ensure reports directories exist
RUN mkdir -p reports/figures reports/drift_logs data/drift_logs

# Switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

# FastAPI on port 8000
EXPOSE 8000

# Health check — hits the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start uvicorn with 2 workers
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
