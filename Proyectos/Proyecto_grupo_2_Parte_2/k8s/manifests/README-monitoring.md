# Monitoring Stack (Prometheus & Grafana)

This directory contains Kubernetes manifests for deploying Prometheus and Grafana to monitor the Diabetes Readmission Prediction API.

## Overview

- **Prometheus**: Metrics collection and storage
- **Grafana**: Data visualization and dashboards

## Architecture

```
API (/metrics) → Prometheus (scrapes every 10s) → Grafana (visualizes)
```

## Components

### Prometheus

- **ConfigMap**: `prometheus-configmap.yaml` - Configuration for scraping targets
- **Deployment**: `prometheus-deployment.yaml` - Prometheus server deployment
- **Service**: `prometheus-service.yaml` - NodePort service on port 30090

### Grafana

- **ConfigMap**: `grafana-configmap.yaml` - Datasource configuration (Prometheus)
- **Deployment**: `grafana-deployment.yaml` - Grafana server deployment
- **Service**: `grafana-service.yaml` - NodePort service on port 30300

## Deployment

### Quick Deploy

```bash
cd k8s
./scripts/deploy-monitoring.sh
```

### Manual Deploy

```bash
# Deploy Prometheus
kubectl apply -f manifests/prometheus/prometheus-configmap.yaml
kubectl apply -f manifests/prometheus/prometheus-deployment.yaml
kubectl apply -f manifests/prometheus/prometheus-service.yaml

# Deploy Grafana
kubectl apply -f manifests/grafana/grafana-configmap.yaml
kubectl apply -f manifests/grafana/grafana-deployment.yaml
kubectl apply -f manifests/grafana/grafana-service.yaml
```

## Access

### Prometheus

- **URL**: `http://<minikube-ip>:30090`
- **Query Examples**:
  - `predict_requests_total` - Total prediction requests
  - `predict_latency_seconds` - Prediction latency histogram
  - `predictions_total` - Predictions by class
  - `predict_errors_total` - Errors by type

### Grafana

- **URL**: `http://<minikube-ip>:30300`
- **Username**: `admin`
- **Password**: `admin`
- **Datasource**: Prometheus (pre-configured at `http://prometheus:9090`)

## API Metrics

The API exposes the following Prometheus metrics at `/metrics`:

### Counters

- `predict_requests_total` - Total number of prediction requests
- `predictions_total{prediction_class}` - Total predictions by class (NO, <30, >30)
- `predict_errors_total{error_type}` - Total errors by type

### Histograms

- `predict_latency_seconds` - Prediction request latency distribution

## Prometheus Configuration

The Prometheus configuration (`prometheus-configmap.yaml`) includes:

- **Scrape interval**: 15 seconds (global)
- **API scrape interval**: 10 seconds
- **Retention**: 15 days
- **Targets**:
  - Prometheus itself (`localhost:9090`)
  - Diabetes API (`diabetes-api:8000/metrics`)

## Grafana Configuration

Grafana is pre-configured with:

- **Prometheus datasource**: Automatically configured
- **Default credentials**: admin/admin (change in production!)

## Example Queries

### Request Rate

```promql
rate(predict_requests_total[5m])
```

### Average Latency

```promql
rate(predict_latency_seconds_sum[5m]) / rate(predict_latency_seconds_count[5m])
```

### Predictions by Class

```promql
predictions_total
```

### Error Rate

```promql
rate(predict_errors_total[5m])
```

## Troubleshooting

### Prometheus not scraping API

1. Check API is running: `kubectl get pods -l app=diabetes-api`
2. Check API metrics endpoint: `curl http://diabetes-api:8000/metrics` (from within cluster)
3. Check Prometheus targets: Go to Prometheus UI → Status → Targets
4. Check Prometheus logs: `kubectl logs -l app=prometheus`

### Grafana can't connect to Prometheus

1. Check Prometheus service: `kubectl get svc prometheus`
2. Check Grafana datasource configuration in ConfigMap
3. Check Grafana logs: `kubectl logs -l app=grafana`

### Metrics not appearing

1. Ensure API has been updated with Prometheus client
2. Make some prediction requests to generate metrics
3. Check API `/metrics` endpoint directly
4. Verify Prometheus is scraping the API

## Storage

Both Prometheus and Grafana use `emptyDir` volumes for storage. This means:

- **Data is ephemeral** - will be lost on pod restart
- For production, consider using PersistentVolumes

## Resource Limits

- **Prometheus**: 512Mi-2Gi memory, 250m-1000m CPU
- **Grafana**: 256Mi-512Mi memory, 100m-500m CPU

Adjust based on your cluster resources.

## Security Notes

⚠️ **Production Considerations**:

1. Change Grafana default credentials
2. Use PersistentVolumes for data persistence
3. Consider using Ingress with TLS instead of NodePort
4. Implement RBAC for Prometheus and Grafana
5. Use secrets for sensitive configuration

