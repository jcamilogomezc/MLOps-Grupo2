# Viewing API Service Logs

This guide explains how to view logs from the Diabetes API service deployed in Kubernetes (Minikube).

## Quick Commands

### View Live Logs (Follow Mode)

```bash
# View logs with live following (most common)
kubectl logs -l app=diabetes-api -f

# Or specify the deployment name
kubectl logs -f deployment/diabetes-api

# View logs from a specific pod
kubectl logs -f <pod-name>
```

### View Recent Logs

```bash
# View last 100 lines
kubectl logs -l app=diabetes-api --tail=100

# View last 50 lines with timestamps
kubectl logs -l app=diabetes-api --tail=50 --timestamps

# View logs from last 10 minutes
kubectl logs -l app=diabetes-api --since=10m
```

### View Logs from Previous Container Instances

```bash
# View logs from previous container (if pod restarted)
kubectl logs -l app=diabetes-api --previous

# View logs with previous container and follow
kubectl logs -l app=diabetes-api -f --previous
```

## Step-by-Step Instructions

### 1. Find the API Pod

First, identify the API pod name:

```bash
# List all pods with the diabetes-api label
kubectl get pods -l app=diabetes-api

# Example output:
# NAME                           READY   STATUS    RESTARTS   AGE
# diabetes-api-7d4f8b9c6d-xyz12  1/1     Running   0          5m
```

### 2. View Logs from the Pod

```bash
# Replace <pod-name> with the actual pod name from step 1
kubectl logs <pod-name>

# Example:
kubectl logs diabetes-api-7d4f8b9c6d-xyz12
```

### 3. Follow Logs in Real-Time

```bash
# Follow logs (like tail -f)
kubectl logs -f <pod-name>

# Or use the label selector (recommended)
kubectl logs -f -l app=diabetes-api
```

### 4. View Logs with Timestamps

```bash
# Add timestamps to log entries
kubectl logs -l app=diabetes-api --timestamps

# Follow with timestamps
kubectl logs -f -l app=diabetes-api --timestamps
```

## Advanced Log Viewing

### Filter Logs

```bash
# View logs and filter with grep (on your local machine)
kubectl logs -l app=diabetes-api | grep "ERROR"

# View logs from a specific time period
kubectl logs -l app=diabetes-api --since=1h
kubectl logs -l app=diabetes-api --since=30m
kubectl logs -l app=diabetes-api --since-time="2024-01-15T10:00:00Z"
```

### View Logs from Multiple Pods

If you have multiple replicas:

```bash
# View logs from all pods with the label
kubectl logs -l app=diabetes-api --all-containers=true

# View logs from a specific container (if multiple containers in pod)
kubectl logs -l app=diabetes-api -c api
```

### Export Logs to File

```bash
# Save logs to a file
kubectl logs -l app=diabetes-api > api-logs.txt

# Save logs with timestamps
kubectl logs -l app=diabetes-api --timestamps > api-logs-with-timestamps.txt

# Save logs from last hour
kubectl logs -l app=diabetes-api --since=1h > api-logs-last-hour.txt
```

## Using Minikube kubectl

If you're using Minikube's bundled kubectl:

```bash
# Use minikube kubectl
minikube kubectl -- logs -l app=diabetes-api -f

# Or set alias (add to ~/.bashrc or ~/.zshrc)
alias kubectl="minikube kubectl --"

# Then use normally
kubectl logs -l app=diabetes-api -f
```

## Common Use Cases

### Debug API Startup Issues

```bash
# View logs from the beginning
kubectl logs -l app=diabetes-api --tail=200

# Check for errors
kubectl logs -l app=diabetes-api | grep -i error

# Check model loading
kubectl logs -l app=diabetes-api | grep -i "model loaded"
```

### Monitor API Requests

```bash
# Follow logs to see incoming requests
kubectl logs -f -l app=diabetes-api

# Filter for prediction requests
kubectl logs -l app=diabetes-api | grep "predict"
```

### Check Preprocessing Warnings

```bash
# View warnings about encoders/scalers
kubectl logs -l app=diabetes-api | grep -i "warning"

# Check preprocessing status
kubectl logs -l app=diabetes-api | grep -i "preprocessing"
```

## Troubleshooting

### Pod Not Found

```bash
# Check if pod exists
kubectl get pods -l app=diabetes-api

# If no pods, check deployment
kubectl get deployment diabetes-api

# Check deployment status
kubectl describe deployment diabetes-api
```

### Logs Are Empty

```bash
# Check if pod is running
kubectl get pods -l app=diabetes-api

# Describe the pod for more info
kubectl describe pod -l app=diabetes-api

# Check pod events
kubectl get events --sort-by='.lastTimestamp' | grep diabetes-api
```

### Cannot Access Logs

```bash
# Check kubectl configuration
kubectl config current-context

# Verify Minikube is running
minikube status

# Check if you can access the cluster
kubectl cluster-info
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `kubectl logs -l app=diabetes-api -f` | Follow logs from API pods |
| `kubectl logs -l app=diabetes-api --tail=100` | View last 100 lines |
| `kubectl logs -l app=diabetes-api --since=10m` | View logs from last 10 minutes |
| `kubectl logs -l app=diabetes-api --timestamps` | View logs with timestamps |
| `kubectl logs -l app=diabetes-api --previous` | View logs from previous container |
| `kubectl get pods -l app=diabetes-api` | List API pods |
| `kubectl describe pod <pod-name>` | Get detailed pod information |

## Example: Complete Log Viewing Session

```bash
# 1. Check if API is deployed
kubectl get pods -l app=diabetes-api

# 2. View recent logs
kubectl logs -l app=diabetes-api --tail=50 --timestamps

# 3. Follow logs in real-time
kubectl logs -f -l app=diabetes-api

# 4. In another terminal, make a test request
curl http://$(minikube ip):30080/health

# 5. Watch the logs update in real-time
```

## Notes

- Logs are stored in the container and are lost when the pod is deleted
- To persist logs, consider setting up a log aggregation solution (e.g., ELK stack, Loki)
- The API logs include:
  - Model loading messages
  - Preprocessing warnings
  - Prediction requests and responses
  - Error messages
  - Health check requests

