# Use official Python slim image for backend
FROM python:3.11-slim as backend

# Set workdir
WORKDIR /app

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
COPY requirements-prophet.txt requirements-prophet.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prophet.txt

# Copy backend code
COPY api.py pipeline.py pipeline_baseline.py ./
COPY src/ ./src/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI backend
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Frontend stage ---
FROM node:20-slim as frontend
WORKDIR /dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm install --omit=dev --legacy-peer-deps
COPY dashboard/ ./
RUN npm run build

# --- Final stage: minimal image ---
FROM python:3.11-slim as final
WORKDIR /app

# Copy backend from backend stage
COPY --from=backend /app /app

# Copy frontend build from frontend stage
COPY --from=frontend /dashboard/build /app/dashboard/build

# Install runtime Python dependencies in final image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-prophet.txt

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI backend (serves API + static dashboard)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
