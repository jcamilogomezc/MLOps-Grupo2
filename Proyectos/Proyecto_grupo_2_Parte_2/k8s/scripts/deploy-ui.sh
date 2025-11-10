#!/bin/bash

# Script to build and deploy the Diabetes UI to Minikube
# This script builds the Docker image in Minikube's Docker daemon
# and deploys it to Kubernetes

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in k8s/scripts/, so go up two levels to get project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UI_DIR="$PROJECT_ROOT/ui"
K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFESTS_DIR="$K8S_DIR/manifests/ui"

echo "=========================================="
echo "Deploying Diabetes UI to Minikube"
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

# Check if API service exists
echo -e "${YELLOW}Checking if API service is available...${NC}"
if ! $KUBECTL_CMD get service diabetes-api &> /dev/null; then
    echo -e "${YELLOW}Warning: diabetes-api service not found.${NC}"
    echo -e "${YELLOW}The UI requires the API to be deployed first.${NC}"
    echo "Please deploy the API first:"
    echo "  ./scripts/deploy-api.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ API service found${NC}"
fi

# Set Docker environment to use Minikube's Docker daemon
echo -e "${YELLOW}Setting Docker environment to Minikube...${NC}"
eval $(minikube -p minikube docker-env)

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
echo "  UI directory: $UI_DIR"
cd "$UI_DIR"
docker build -t diabetes-ui:latest .

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
if [ ! -f "$MANIFESTS_DIR/ui-deployment.yaml" ]; then
    echo -e "${RED}Error: ui-deployment.yaml not found in $MANIFESTS_DIR${NC}"
    exit 1
fi

if [ ! -f "$MANIFESTS_DIR/ui-service.yaml" ]; then
    echo -e "${RED}Error: ui-service.yaml not found in $MANIFESTS_DIR${NC}"
    exit 1
fi

# Apply deployment
$KUBECTL_CMD apply -f "$MANIFESTS_DIR/ui-deployment.yaml"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to apply deployment${NC}"
    exit 1
fi

# Apply service
$KUBECTL_CMD apply -f "$MANIFESTS_DIR/ui-service.yaml"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to apply service${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Deployment and Service applied successfully${NC}"

# Wait for deployment to be ready
echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
$KUBECTL_CMD wait --for=condition=available --timeout=120s deployment/diabetes-ui

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Deployment might not be ready yet. Check status with:${NC}"
    echo "  $KUBECTL_CMD get pods -l app=diabetes-ui"
    echo "  $KUBECTL_CMD describe pod -l app=diabetes-ui"
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
echo "  Service Name: diabetes-ui"
echo "  Namespace: default"
echo ""
echo "Access the UI:"
MINIKUBE_IP=$(minikube ip)
echo "  Minikube IP: $MINIKUBE_IP"
echo "  NodePort: 30085"
echo "  URL: http://$MINIKUBE_IP:30085"
echo ""
echo "Quick Commands:"
echo "  Get service URL: minikube service diabetes-ui --url"
echo "  View logs: $KUBECTL_CMD logs -l app=diabetes-ui -f"
echo "  Check status: $KUBECTL_CMD get pods -l app=diabetes-ui"
echo "  Check service: $KUBECTL_CMD get service diabetes-ui"
echo ""
echo -e "${YELLOW}Note: Ensure the API service is running and accessible at:${NC}"
echo "  http://diabetes-api:8000 (internal Kubernetes service)"
echo ""
echo -e "${YELLOW}The UI will connect to the API using the internal service name.${NC}"
echo ""

