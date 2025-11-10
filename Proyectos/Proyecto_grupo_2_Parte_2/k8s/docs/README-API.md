# Deploying Diabetes API to Kubernetes (Minikube)

This guide explains how to deploy the FastAPI inference service to Minikube while keeping other services (MLflow, MinIO, etc.) running on Docker.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Minikube (Kubernetes)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Diabetes API Pod                                 │  │
│  │  - FastAPI Service                                │  │
│  │  - Port: 8000 (NodePort: 30080)                   │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          │ HTTP
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Docker (Host Machine)                │
│  ┌───────────────────────────────────────────────────┐  │
│  │  MLflow Server                                    │  │
│  │  - Port: 8002 (host) -> 5000 (container)          │  │
│  │  - Model Registry                                 │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────┐  │
│  │  MinIO Server                                     │  │
│  │  - Port: 9000 (S3 API)                            │  │
│  │  - Artifact Storage                               │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Minikube installed and running**
   ```bash
   minikube status
   ```

2. **Docker Compose services running**
   ```bash
   # Start MLflow and other services
   docker-compose -f docker-compose.inference.yml up -d mlflow minio mlflow-meta
   ```

3. **Verify MLflow is accessible**
   ```bash
   curl http://localhost:8002/health
   ```

## Deployment Steps

### Step 1: Start Minikube

```bash
# Start Minikube (if not already running)
minikube start --driver=docker

# Verify Minikube is running
minikube status
```

### Step 2: Configure MLflow Connection

The API needs to connect to MLflow running on Docker. We need to determine the correct host IP.

```bash
# Run the helper script to detect the correct MLflow URL
cd k8s
./scripts/setup-mlflow-connection.sh
```

This script will:
- Test if `host.minikube.internal` works
- If not, detect the host machine IP
- Provide the correct MLflow URL to use

### Step 3: Update Deployment (if needed)

If the helper script detects a different IP, update the deployment:

```bash
# Option 1: Update via kubectl (after deployment)
kubectl set env deployment/diabetes-api MLFLOW_TRACKING_URI=http://<HOST_IP>:8002

# Option 2: Edit the deployment file before deploying
# Edit k8s/manifests/api/api-deployment.yaml and update MLFLOW_TRACKING_URI
```

### Step 4: Build and Deploy API

```bash
# Navigate to k8s directory
cd k8s

# Run the deployment script
./scripts/deploy-api.sh
```

This script will:
1. Set Docker environment to use Minikube's Docker daemon
2. Build the Docker image
3. Deploy to Kubernetes
4. Create the Service (NodePort)
5. Wait for deployment to be ready

### Step 5: Verify Deployment

```bash
# Check pods
kubectl get pods -l app=diabetes-api

# Check service
kubectl get service diabetes-api

# View logs
kubectl logs -l app=diabetes-api -f

# Test health endpoint
minikube service diabetes-api --url
curl $(minikube service diabetes-api --url)/health
```

## Accessing the API

### Get Service URL

```bash
# Get the service URL
minikube service diabetes-api --url

# Or access directly via Minikube IP
MINIKUBE_IP=$(minikube ip)
curl http://$MINIKUBE_IP:30080/health
```

### API Endpoints

- **Health Check**: `GET /health`
- **Model Info**: `GET /model-info`
- **Predict**: `POST /predict`

### Example: Test Prediction

```bash
# Get API URL
API_URL=$(minikube service diabetes-api --url)

# Test health
curl $API_URL/health

# Get model info
curl $API_URL/model-info

# Make prediction (example)
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "race": "Caucasian",
      "gender": "Female",
      "age": "[70-80)",
      "admission_type_id": 1,
      "time_in_hospital": 1,
      "num_lab_procedures": 41,
      "num_medications": 1
    }
  }'
```

## Troubleshooting

### API cannot connect to MLflow

**Problem**: Pod logs show connection errors to MLflow

**Solutions**:

1. **Verify MLflow is running**:
   ```bash
   docker ps | grep mlflow
   curl http://localhost:8002/health
   ```

2. **Check the MLflow URL in deployment**:
   ```bash
   kubectl get deployment diabetes-api -o yaml | grep MLFLOW_TRACKING_URI
   ```

3. **Test connectivity from Minikube**:
   ```bash
   minikube ssh
   curl http://host.minikube.internal:8002/health
   # Or try with host IP
   curl http://<HOST_IP>:8002/health
   ```

4. **Update MLflow URL**:
   ```bash
   # Get host IP
   HOST_IP=$(ip route show | grep -i default | awk '{ print $3}')
   
   # Update deployment
   kubectl set env deployment/diabetes-api MLFLOW_TRACKING_URI=http://$HOST_IP:8002
   
   # Restart pods
   kubectl rollout restart deployment/diabetes-api
   ```

### Model not found errors

**Problem**: API returns "Model not found" errors

**Solutions**:

1. **Verify model is registered in MLflow**:
   ```bash
   # Access MLflow UI
   open http://localhost:8002
   # Or
   minikube service mlflow --url  # If MLflow is also in K8s
   ```

2. **Check model name and stage**:
   ```bash
   kubectl get deployment diabetes-api -o yaml | grep REGISTERED_MODEL
   ```

3. **Verify MLflow can access the model**:
   ```bash
   # Test from MLflow container
   docker exec -it mlflow_server mlflow models list
   ```

### Pod is not starting

**Problem**: Pod stays in `CrashLoopBackOff` or `Pending` state

**Solutions**:

1. **Check pod logs**:
   ```bash
   kubectl logs -l app=diabetes-api --tail=50
   ```

2. **Check pod events**:
   ```bash
   kubectl describe pod -l app=diabetes-api
   ```

3. **Verify image exists in Minikube**:
   ```bash
   eval $(minikube docker-env)
   docker images | grep diabetes-api
   ```

4. **Rebuild image**:
   ```bash
   ./scripts/deploy-api.sh
   ```

## Configuration

### Environment Variables

The deployment uses these environment variables (configurable in `manifests/api/api-deployment.yaml`):

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: `http://host.minikube.internal:8002`)
- `REGISTERED_MODEL_NAME`: Model name in MLflow (default: `diabetes_readmission_model`)
- `MODEL_STAGE_OR_VERSION`: Model stage or version (default: `Production`)
- `AWS_ACCESS_KEY_ID`: MinIO access key (default: `admin`)
- `AWS_SECRET_ACCESS_KEY`: MinIO secret key (default: `supersecret`)
- `MLFLOW_S3_ENDPOINT_URL`: MinIO S3 endpoint (default: `http://host.minikube.internal:9000`)

### Updating Configuration

```bash
# Update environment variables
kubectl set env deployment/diabetes-api MLFLOW_TRACKING_URI=http://new-url:8002

# Or edit the deployment
kubectl edit deployment diabetes-api

# Restart after changes
kubectl rollout restart deployment/diabetes-api
```

## Scaling

```bash
# Scale to multiple replicas
kubectl scale deployment diabetes-api --replicas=3

# Check replicas
kubectl get deployment diabetes-api
```

## Cleanup

```bash
# Delete deployment and service
kubectl delete deployment diabetes-api
kubectl delete service diabetes-api

# Remove image from Minikube
eval $(minikube docker-env)
docker rmi diabetes-api:latest
```

## Next Steps

- Set up Ingress for better URL routing
- Add monitoring and logging (Prometheus, Grafana)
- Implement auto-scaling based on load
- Set up CI/CD pipeline for automatic deployments

