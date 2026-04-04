FROM node:18 AS frontend-builder

WORKDIR /frontend

COPY dashboard/package*.json ./
RUN npm install --no-audit --no-fund

COPY dashboard/ ./
RUN npm run build

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /frontend/build ./dashboard/build

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]