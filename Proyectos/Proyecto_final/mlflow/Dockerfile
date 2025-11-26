FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libpq-dev && \
    pip install --no-cache-dir mlflow==2.4.1 gunicorn psycopg2-binary && \
    apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*

# create mlflow directories
RUN mkdir -p /mlflow/artifacts
WORKDIR /mlflow

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--backend-store-uri", "postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}", "--default-artifact-root", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5000"]
