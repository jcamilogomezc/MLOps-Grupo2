#!/bin/bash

# Script to deploy Prometheus and Grafana to Minikube

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMETHEUS_MANIFESTS_DIR="$K8S_DIR/manifests/prometheus"
GRAFANA_MANIFESTS_DIR="$K8S_DIR/manifests/grafana"

echo "=========================================="
echo "Deploying Monitoring Stack (Prometheus & Grafana)"
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
    exit 1
fi

# Verify minikube is actually running
MINIKUBE_STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "")
if [ "$MINIKUBE_STATUS" != "Running" ]; then
    echo -e "${RED}Error: Minikube exists but is not running. Current status: ${MINIKUBE_STATUS}${NC}"
    echo "Please start Minikube:"
    echo "  minikube start --driver=docker"
    exit 1
fi

echo -e "${GREEN}✓ Minikube is running${NC}"

# Check for kubectl and set up kubectl command
if command -v kubectl &> /dev/null; then
    KUBECTL_CMD="kubectl"
    echo -e "${GREEN}✓ Using kubectl from PATH${NC}"
else
    if minikube kubectl -- version --client &> /dev/null; then
        KUBECTL_CMD="minikube kubectl --"
        echo -e "${GREEN}✓ Using minikube kubectl${NC}"
    else
        echo -e "${RED}Error: kubectl is not available${NC}"
        exit 1
    fi
fi

# Deploy Prometheus
echo ""
echo -e "${YELLOW}Deploying Prometheus...${NC}"
echo "  Manifests directory: $PROMETHEUS_MANIFESTS_DIR"

# Verify YAML files exist
if [ ! -f "$PROMETHEUS_MANIFESTS_DIR/prometheus-configmap.yaml" ]; then
    echo -e "${RED}Error: prometheus-configmap.yaml not found${NC}"
    exit 1
fi

if [ ! -f "$PROMETHEUS_MANIFESTS_DIR/prometheus-deployment.yaml" ]; then
    echo -e "${RED}Error: prometheus-deployment.yaml not found${NC}"
    exit 1
fi

if [ ! -f "$PROMETHEUS_MANIFESTS_DIR/prometheus-service.yaml" ]; then
    echo -e "${RED}Error: prometheus-service.yaml not found${NC}"
    exit 1
fi

# Apply Prometheus manifests
$KUBECTL_CMD apply -f "$PROMETHEUS_MANIFESTS_DIR/prometheus-configmap.yaml"
$KUBECTL_CMD apply -f "$PROMETHEUS_MANIFESTS_DIR/prometheus-deployment.yaml"
$KUBECTL_CMD apply -f "$PROMETHEUS_MANIFESTS_DIR/prometheus-service.yaml"

echo -e "${GREEN}✓ Prometheus manifests applied${NC}"

# Deploy Grafana
echo ""
echo -e "${YELLOW}Deploying Grafana...${NC}"
echo "  Manifests directory: $GRAFANA_MANIFESTS_DIR"

# Verify YAML files exist
if [ ! -f "$GRAFANA_MANIFESTS_DIR/grafana-configmap.yaml" ]; then
    echo -e "${RED}Error: grafana-configmap.yaml not found${NC}"
    exit 1
fi

if [ ! -f "$GRAFANA_MANIFESTS_DIR/grafana-deployment.yaml" ]; then
    echo -e "${RED}Error: grafana-deployment.yaml not found${NC}"
    exit 1
fi

if [ ! -f "$GRAFANA_MANIFESTS_DIR/grafana-service.yaml" ]; then
    echo -e "${RED}Error: grafana-service.yaml not found${NC}"
    exit 1
fi

# Apply Grafana manifests
$KUBECTL_CMD apply -f "$GRAFANA_MANIFESTS_DIR/grafana-configmap.yaml"
$KUBECTL_CMD apply -f "$GRAFANA_MANIFESTS_DIR/grafana-deployment.yaml"
$KUBECTL_CMD apply -f "$GRAFANA_MANIFESTS_DIR/grafana-service.yaml"

echo -e "${GREEN}✓ Grafana manifests applied${NC}"

# Wait for deployments to be ready
echo ""
echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
$KUBECTL_CMD wait --for=condition=available --timeout=120s deployment/prometheus || echo -e "${YELLOW}Warning: Prometheus deployment might not be ready yet${NC}"
$KUBECTL_CMD wait --for=condition=available --timeout=120s deployment/grafana || echo -e "${YELLOW}Warning: Grafana deployment might not be ready yet${NC}"

# Get service URLs
echo ""
echo -e "${GREEN}=========================================="
echo "Deployment Complete!"
echo "==========================================${NC}"
echo ""
MINIKUBE_IP=$(minikube ip)
echo "Service Information:"
echo "  Prometheus Service: prometheus"
echo "  Grafana Service: grafana"
echo "  Namespace: default"
echo ""
echo "Access Services:"
echo "  Prometheus: http://$MINIKUBE_IP:30090"
echo "  Grafana:    http://$MINIKUBE_IP:30300"
echo ""
echo "Grafana Credentials:"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "Quick Commands:"
echo "  Get Prometheus URL: minikube service prometheus --url"
echo "  Get Grafana URL:    minikube service grafana --url"
echo "  View Prometheus logs: $KUBECTL_CMD logs -l app=prometheus -f"
echo "  View Grafana logs:    $KUBECTL_CMD logs -l app=grafana -f"
echo "  Check Prometheus status: $KUBECTL_CMD get pods -l app=prometheus"
echo "  Check Grafana status:    $KUBECTL_CMD get pods -l app=grafana"
echo ""
echo "Prometheus is configured to scrape metrics from:"
echo "  - diabetes-api:8000/metrics"
echo ""

