# Diabetes Readmission Prediction UI

A Streamlit-based web interface for interacting with the Diabetes Readmission Prediction model deployed in production.

## Features

- **Interactive Form**: Input patient information through a user-friendly interface
- **Default Values**: Pre-populated form with example patient data
- **Model Information**: Display of the production model version and metadata
- **Real-time Prediction**: Get predictions with probability distributions
- **Visual Results**: Color-coded predictions and probability charts

## Requirements

- Python 3.10+
- Streamlit 1.39.0+
- Requests library
- Access to the Diabetes API service

## Local Development

### Prerequisites

1. Ensure the API service is running and accessible
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

The UI will be available at `http://localhost:8501`

### Configuration

Set environment variables to configure the API endpoint:

```bash
export API_URL=http://localhost:8000  # Local API
# Or for Kubernetes:
export API_URL=http://diabetes-api:8000  # Kubernetes service name
```

## Docker Deployment

### Building the Docker Image

```bash
docker build -t diabetes-ui:latest .
```

### Running with Docker

```bash
docker run -p 8501:8501 \
  -e API_URL=http://host.docker.internal:8000 \
  diabetes-ui:latest
```

## Kubernetes Deployment

### Prerequisites

- Minikube running
- Kubernetes cluster access
- API service deployed

### Deploying to Kubernetes

1. Build and deploy using the deployment script:

```bash
cd k8s
./scripts/deploy-ui.sh
```

2. Access the UI:

```bash
# Get Minikube IP
minikube ip

# Access via NodePort (default: 30085)
# http://<minikube-ip>:30085

# Or use minikube service
minikube service diabetes-ui --url
```

### Manual Deployment

```bash
# Build image in Minikube Docker daemon
eval $(minikube -p minikube docker-env)
docker build -t diabetes-ui:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/manifests/ui/ui-deployment.yaml
kubectl apply -f k8s/manifests/ui/ui-service.yaml

# Check status
kubectl get pods -l app=diabetes-ui
kubectl get service diabetes-ui
```

## Usage

1. **Load Default Values**: Click the "Load Default Values" button to populate the form with example data
2. **Fill Patient Information**: Complete the form with patient demographics, admission details, medical information, and medications
3. **Predict**: Click "Predict Readmission" to get the prediction
4. **Review Results**: View the prediction (NO, <30, or >30 days) and probability distribution

## Model Information

The UI displays:
- Model name: `diabetes_readmission_model`
- Model version from MLflow
- Stage: `Production`
- Preprocessing status (encoders, scaler)

## API Integration

The UI connects to the Diabetes API service at the endpoint configured via `API_URL` environment variable:
- `/model-info`: Get model metadata
- `/predict`: Submit prediction request

## Troubleshooting

### API Connection Issues

If the UI cannot connect to the API:
1. Verify the API service is running: `kubectl get pods -l app=diabetes-api`
2. Check API URL configuration: Ensure `API_URL` points to the correct service
3. For Kubernetes: Use internal service name `http://diabetes-api:8000`
4. For local development: Use `http://localhost:8000`

### Model Information Not Loading

If model information is not displayed:
1. Verify MLflow is accessible from the API service
2. Check that a model is registered as "Production" in MLflow
3. Review API logs: `kubectl logs -l app=diabetes-api`

## File Structure

```
ui/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image definition
└── README.md            # This file
```

## Environment Variables

- `API_URL`: URL of the Diabetes API service (default: `http://diabetes-api:8000`)
- `MLFLOW_TRACKING_URI`: MLflow tracking URI (optional, for direct MLflow access)

## Port Configuration

- Default port: `8501`
- Kubernetes NodePort: `30085`
- Configurable via Kubernetes service manifest

## Health Checks

The UI includes health check endpoints:
- Streamlit health: `/_stcore/health`
- Configured in Kubernetes deployment for liveness and readiness probes

