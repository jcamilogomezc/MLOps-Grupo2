#!/bin/bash

# Script to build and deploy the Diabetes API to Minikube
# This script builds the Docker image in Minikube's Docker daemon
# and deploys it to Kubernetes

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in k8s/scripts/, so go up two levels to get project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
API_DIR="$PROJECT_ROOT/api"
K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFESTS_DIR="$K8S_DIR/manifests/api"

echo "=========================================="
echo "Deploying Diabetes API to Minikube"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Minikube is running
if ! minikube status &> /dev/null; then
    echo -e "${RED}Error: Minikube is not running. Please start Minikube first:${NC}"
    echo "  minikube start --driver=docker"
    echo ""
    echo "If you encounter network conflicts, run:"
    echo "  ./scripts/fix-minikube-network.sh"
    exit 1
fi

# Verify minikube is actually running (not just exists)
MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "")
if [ "$MINIKUBE_STATUS" != "Running" ]; then
    echo -e "${RED}Error: Minikube exists but is not running. Current status: ${MINIKUBE_STATUS}${NC}"
    echo "Please start Minikube:"
    echo "  minikube start --driver=docker"
    echo ""
    echo "If you encounter network conflicts, run:"
    echo "  ./scripts/fix-minikube-network.sh"
    exit 1
fi

echo -e "${GREEN}✓ Minikube is running${NC}"

# Check for kubectl and set up kubectl command
if command -v kubectl &> /dev/null; then
    KUBECTL_CMD="kubectl"
    echo -e "${GREEN}✓ Using kubectl from PATH${NC}"
else
    # Use minikube kubectl if kubectl is not in PATH
    if minikube kubectl -- version --client &> /dev/null; then
        KUBECTL_CMD="minikube kubectl --"
        echo -e "${GREEN}✓ Using minikube kubectl${NC}"
    else
        echo -e "${RED}Error: kubectl is not available${NC}"
        echo "Please install kubectl or ensure minikube is properly configured"
        exit 1
    fi
fi

# Set Docker environment to use Minikube's Docker daemon
echo -e "${YELLOW}Setting Docker environment to Minikube...${NC}"
eval $(minikube -p minikube docker-env)

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
echo "  API directory: $API_DIR"
cd "$API_DIR"
docker build -t diabetes-api:latest .

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Docker build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Deploy to Kubernetes
echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
echo "  K8s directory: $K8S_DIR"
echo "  Manifests directory: $MANIFESTS_DIR"
cd "$K8S_DIR"

# Verify YAML files exist
if [ ! -f "$MANIFESTS_DIR/api-deployment.yaml" ]; then
    echo -e "${RED}Error: api-deployment.yaml not found in $MANIFESTS_DIR${NC}"
    exit 1
fi

if [ ! -f "$MANIFESTS_DIR/api-service.yaml" ]; then
    echo -e "${RED}Error: api-service.yaml not found in $MANIFESTS_DIR${NC}"
    exit 1
fi

# Apply deployment
$KUBECTL_CMD apply -f "$MANIFESTS_DIR/api-deployment.yaml"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to apply deployment${NC}"
    exit 1
fi

# Apply service (NodePort for Minikube)
$KUBECTL_CMD apply -f "$MANIFESTS_DIR/api-service.yaml"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to apply service${NC}"
    exit 1
fi

# Optionally apply LoadBalancer service (for cloud deployments)
if [ -f "$MANIFESTS_DIR/api-service-loadbalancer.yaml" ]; then
    echo -e "${YELLOW}Note: LoadBalancer service found. Apply it manually if needed for cloud deployments:${NC}"
    echo "  $KUBECTL_CMD apply -f $MANIFESTS_DIR/api-service-loadbalancer.yaml"
fi

# Apply HPA if metrics server is available
if [ -f "$MANIFESTS_DIR/api-hpa.yaml" ]; then
    echo -e "${YELLOW}Applying Horizontal Pod Autoscaler...${NC}"
    # Check if metrics-server is available (required for HPA)
    if $KUBECTL_CMD get deployment metrics-server -n kube-system &> /dev/null; then
        $KUBECTL_CMD apply -f "$MANIFESTS_DIR/api-hpa.yaml"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ HPA applied successfully${NC}"
        else
            echo -e "${YELLOW}Warning: Failed to apply HPA. Metrics server might not be ready.${NC}"
            echo "  Enable metrics server in Minikube: minikube addons enable metrics-server"
        fi
    else
        echo -e "${YELLOW}Warning: Metrics server not found. HPA requires metrics-server.${NC}"
        echo "  Enable metrics server: minikube addons enable metrics-server"
        echo "  Then apply HPA manually: $KUBECTL_CMD apply -f $MANIFESTS_DIR/api-hpa.yaml"
    fi
fi

echo -e "${GREEN}✓ Deployment and Service applied successfully${NC}"

# Wait for deployment to be ready
echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
$KUBECTL_CMD wait --for=condition=available --timeout=120s deployment/diabetes-api

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Deployment might not be ready yet. Check status with:${NC}"
    echo "  $KUBECTL_CMD get pods -l app=diabetes-api"
    echo "  $KUBECTL_CMD describe pod -l app=diabetes-api"
else
    echo -e "${GREEN}✓ Deployment is ready${NC}"
fi

# Get service URL
echo ""
echo -e "${GREEN}=========================================="
echo "Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "Service Information:"
echo "  Service Name: diabetes-api"
echo "  Namespace: default"
echo ""
echo "Access the API:"
MINIKUBE_IP=$(minikube ip)
echo "  Minikube IP: $MINIKUBE_IP"
echo "  NodePort: 30080"
echo "  URL: http://$MINIKUBE_IP:30080"
echo ""
echo "Quick Commands:"
echo "  Get service URL: minikube service diabetes-api --url"
echo "  View logs: $KUBECTL_CMD logs -l app=diabetes-api -f"
echo "  Check status: $KUBECTL_CMD get pods -l app=diabetes-api"
echo "  Check service: $KUBECTL_CMD get service diabetes-api"
echo "  Check HPA: $KUBECTL_CMD get hpa diabetes-api-hpa"
echo "  Scale manually: $KUBECTL_CMD scale deployment diabetes-api --replicas=5"
echo ""
echo "Performance Optimizations:"
echo "  - Multiple replicas: 3 (configurable in deployment)"
echo "  - Multiple workers per pod: 2 (configurable via UVICORN_WORKERS env var)"
echo "  - HPA: Auto-scales based on CPU/memory (if metrics-server is enabled)"
echo "  - Async endpoints: Better concurrency handling"
echo ""
echo "API Endpoints:"
echo "  Health: http://$MINIKUBE_IP:30080/health"
echo "  Model Info: http://$MINIKUBE_IP:30080/model-info"
echo "  Predict: http://$MINIKUBE_IP:30080/predict (POST)"
echo ""
echo -e "${YELLOW}Note: Ensure MLflow is running on Docker and accessible at:${NC}"
echo "  http://host.minikube.internal:8002"
echo ""

# Check if we're on Linux and suggest host IP if needed
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${YELLOW}Linux detected: If host.minikube.internal doesn't work,${NC}"
    echo -e "${YELLOW}you may need to use the host gateway IP instead.${NC}"
    echo ""
    echo "To find the host IP from Minikube:"
    echo "  minikube ssh 'route -n get default' | grep gateway"
    echo ""
    echo "Or from host machine:"
    HOST_IP=$(ip route show | grep -i default | awk '{ print $3}' | head -1)
    if [ -n "$HOST_IP" ]; then
        echo "  Detected host IP: $HOST_IP"
        echo "  Update with: $KUBECTL_CMD set env deployment/diabetes-api MLFLOW_TRACKING_URI=http://$HOST_IP:8002"
    fi
    echo ""
fi
echo ""

