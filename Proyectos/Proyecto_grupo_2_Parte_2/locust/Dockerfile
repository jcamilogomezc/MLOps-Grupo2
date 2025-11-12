FROM python:3.11-slim

WORKDIR /locust

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy locustfile
COPY locustfile.py .

# Expose Locust web UI port
EXPOSE 8089

# Default command: run Locust web UI
# Can be overridden in Kubernetes deployment
# The host will be set via environment variable in Kubernetes
CMD ["locust", "-f", "locustfile.py", "--host=http://diabetes-api:8000", "--web-host=0.0.0.0", "--web-port=8089"]

