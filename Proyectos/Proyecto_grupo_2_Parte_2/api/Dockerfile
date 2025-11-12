FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with multiple workers for better concurrency
# Workers are set via environment variable UVICORN_WORKERS (default: 2)
# Using uvicorn with multiple workers for better async performance
# Note: With multiple workers, each worker runs in a separate process
# This provides better CPU utilization for CPU-bound ML inference tasks
# Each worker can handle many concurrent requests asynchronously
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2} --access-log --log-level info --timeout-keep-alive 75"]

