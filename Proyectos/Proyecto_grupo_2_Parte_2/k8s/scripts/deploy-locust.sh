#!/bin/bash

# Script to build and deploy Locust to Minikube
# This script builds the Docker image in Minikube's Docker daemon
# and deploys it to Kubernetes

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in k8s/scripts/, so go up two levels to get project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCUST_DIR="$PROJECT_ROOT/locust"
K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFESTS_DIR="$K8S_DIR/manifests/locust"

echo "=========================================="
echo "Deploying Locust to Minikube"
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

# Check if API service exists (Locust needs the API to test)
echo -e "${YELLOW}Checking if API service is available...${NC}"
if ! $KUBECTL_CMD get service diabetes-api &> /dev/null; then
    echo -e "${YELLOW}Warning: diabetes-api service not found.${NC}"
    echo -e "${YELLOW}Locust requires the API to be deployed first.${NC}"
    echo "Please deploy the API first or use: ./scripts/deploy-api.sh"
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
echo "  Locust directory: $LOCUST_DIR"
cd "$LOCUST_DIR"
docker build -t locust:latest .

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
if [ ! -f "$MANIFESTS_DIR/locust-deployment.yaml" ]; then
    echo -e "${RED}Error: locust-deployment.yaml not found in $MANIFESTS_DIR${NC}"
    exit 1
fi

if [ ! -f "$MANIFESTS_DIR/locust-service.yaml" ]; then
    echo -e "${RED}Error: locust-service.yaml not found in $MANIFESTS_DIR${NC}"
    exit 1
fi

# Apply deployment
$KUBECTL_CMD apply -f "$MANIFESTS_DIR/locust-deployment.yaml"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to apply deployment${NC}"
    exit 1
fi

# Apply service
$KUBECTL_CMD apply -f "$MANIFESTS_DIR/locust-service.yaml"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to apply service${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Deployment and Service applied successfully${NC}"

# Wait for deployment to be ready
echo -e "${YELLOW}Waiting for deployment to be ready...${NC}"
$KUBECTL_CMD wait --for=condition=available --timeout=120s deployment/locust

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Deployment might not be ready yet. Check status with:${NC}"
    echo "  $KUBECTL_CMD get pods -l app=locust"
    echo "  $KUBECTL_CMD describe pod -l app=locust"
else
    echo -e "${GREEN}✓ Deployment is ready${NC}"
fi

# Get service URL
echo ""
echo -e "${GREEN}=========================================="
echo "Locust Deployment Complete!"
echo "==========================================${NC}"
echo ""
MINIKUBE_IP=$(minikube ip)
echo "Locust Service:"
echo "  Service Name: locust"
echo "  NodePort: 30189"
echo "  URL: http://$MINIKUBE_IP:30189"
echo ""
echo "Access Locust Web UI:"
echo "  http://$MINIKUBE_IP:30189"
echo ""
echo "Locust is configured to test:"
echo "  Target API: http://diabetes-api:8000"
echo ""
echo "Quick Commands:"
echo "  Locust logs: $KUBECTL_CMD logs -l app=locust -f"
echo "  Locust status: $KUBECTL_CMD get pods -l app=locust"
echo "  Get service URL: minikube service locust --url"
echo ""

