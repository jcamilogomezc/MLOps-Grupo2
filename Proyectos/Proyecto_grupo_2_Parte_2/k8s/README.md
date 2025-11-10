# Kubernetes Deployment

This directory contains all Kubernetes-related files for deploying the Diabetes API to Minikube.

## Directory Structure

```
k8s/
├── scripts/           # Deployment and utility scripts
├── manifests/         # Kubernetes manifests (YAML files)
│   ├── api/          # API deployment manifests
│   └── database/     # Database deployment manifests
├── docs/             # Documentation and guides
├── docker/           # Docker Compose files for local services
└── README.md         # This file
```

## Quick Start

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a quick start guide.

## Scripts

All deployment and utility scripts are located in the `scripts/` directory:

- **deploy-api.sh** - Build and deploy the Diabetes API to Minikube
- **verify-deployment.sh** - Verify the API deployment status
- **fix-mlflow-connection.sh** - Fix MLflow connection issues
- **setup-mlflow-connection.sh** - Helper script to configure MLflow connection
- **start-port-forward.sh** - Start port forwarding for external access
- **install-socat.sh** - Install socat for port forwarding
- **port-forward-python.py** - Python-based port forwarder (alternative to socat)

### Usage

Run scripts from the `k8s` directory:

```bash
cd k8s
./scripts/deploy-api.sh
./scripts/verify-deployment.sh
```

## Manifests

Kubernetes manifests are organized by component:

- **manifests/api/** - API deployment, service, and load balancer configurations
- **manifests/database/** - Database deployment configurations

### Applying Manifests

```bash
# Apply API deployment
kubectl apply -f manifests/api/api-deployment.yaml
kubectl apply -f manifests/api/api-service.yaml

# Apply load balancer for external access
kubectl apply -f manifests/api/api-service-loadbalancer.yaml
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **QUICKSTART.md** - Quick start guide
- **README-API.md** - Detailed API deployment documentation
- **EXTERNAL-ACCESS-QUICK.md** - Quick guide for external access
- **EXTERNAL-ACCESS-INGRESS.md** - Ingress-based external access guide

## Docker Compose

Local services (MLflow, MinIO) are configured in `docker/`:

- **docker-compose.db.yml** - Database and MLflow services

## Common Tasks

### Deploy API

```bash
./scripts/deploy-api.sh
```

### Verify Deployment

```bash
./scripts/verify-deployment.sh
```

### Fix MLflow Connection

```bash
./scripts/fix-mlflow-connection.sh
```

### Access API Externally

```bash
# Option 1: Minikube tunnel
kubectl apply -f manifests/api/api-service-loadbalancer.yaml
minikube tunnel

# Option 2: Port forwarding
./scripts/start-port-forward.sh
```

## Requirements

- Minikube installed and running
- Docker running (for building images and running MLflow)
- kubectl or minikube kubectl (scripts will auto-detect)
- MLflow running on Docker (port 8002)

For more details, see the documentation in the `docs/` directory.

