#!/bin/bash

# Consolidated deployment script for Diabetes API, UI, Monitoring stack, and Locust
# This script can deploy all components or specific ones based on arguments

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in k8s/scripts/, so go up two levels to get project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
API_DIR="$PROJECT_ROOT/api"
UI_DIR="$PROJECT_ROOT/ui"
LOCUST_DIR="$PROJECT_ROOT/locust"
K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
API_MANIFESTS_DIR="$K8S_DIR/manifests/api"
UI_MANIFESTS_DIR="$K8S_DIR/manifests/ui"
PROMETHEUS_MANIFESTS_DIR="$K8S_DIR/manifests/prometheus"
GRAFANA_MANIFESTS_DIR="$K8S_DIR/manifests/grafana"
LOCUST_MANIFESTS_DIR="$K8S_DIR/manifests/locust"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default: deploy API, UI, and Monitoring (Locust is optional)
DEPLOY_API=true
DEPLOY_UI=true
DEPLOY_MONITORING=true
DEPLOY_LOCUST=false  # Locust is optional, not deployed by default

# Track which individual flags (--api, --ui, etc.) were used
# This helps determine if we should reset defaults when individual flags are used
API_FLAG_USED=false
UI_FLAG_USED=false
MONITORING_FLAG_USED=false
LOCUST_FLAG_USED=false
EXPLICIT_ONLY_USED=false  # Track if --all or --*-only was used

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            DEPLOY_API=true
            DEPLOY_UI=true
            DEPLOY_MONITORING=true
            DEPLOY_LOCUST=true
            EXPLICIT_ONLY_USED=true
            shift
            ;;
        --api-only)
            DEPLOY_API=true
            DEPLOY_UI=false
            DEPLOY_MONITORING=false
            DEPLOY_LOCUST=false
            EXPLICIT_ONLY_USED=true
            shift
            ;;
        --ui-only)
            DEPLOY_API=false
            DEPLOY_UI=true
            DEPLOY_MONITORING=false
            DEPLOY_LOCUST=false
            EXPLICIT_ONLY_USED=true
            shift
            ;;
        --monitoring-only)
            DEPLOY_API=false
            DEPLOY_UI=false
            DEPLOY_MONITORING=true
            DEPLOY_LOCUST=false
            EXPLICIT_ONLY_USED=true
            shift
            ;;
        --locust-only)
            DEPLOY_API=false
            DEPLOY_UI=false
            DEPLOY_MONITORING=false
            DEPLOY_LOCUST=true
            EXPLICIT_ONLY_USED=true
            shift
            ;;
        --api)
            DEPLOY_API=true
            API_FLAG_USED=true
            shift
            ;;
        --ui)
            DEPLOY_UI=true
            UI_FLAG_USED=true
            shift
            ;;
        --monitoring)
            DEPLOY_MONITORING=true
            MONITORING_FLAG_USED=true
            shift
            ;;
        --locust)
            DEPLOY_LOCUST=true
            LOCUST_FLAG_USED=true
            shift
            ;;
        --skip-api)
            DEPLOY_API=false
            shift
            ;;
        --skip-ui)
            DEPLOY_UI=false
            shift
            ;;
        --skip-monitoring)
            DEPLOY_MONITORING=false
            shift
            ;;
        --with-locust)
            DEPLOY_LOCUST=true
            LOCUST_FLAG_USED=true
            shift
            ;;
        --skip-locust)
            DEPLOY_LOCUST=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Deploy Diabetes API, UI, Monitoring stack, and Locust to Minikube"
            echo ""
            echo "Options:"
            echo "  --all                Deploy all services (API, UI, Monitoring, Locust)"
            echo "  --api-only           Deploy only the API"
            echo "  --ui-only            Deploy only the UI"
            echo "  --monitoring-only    Deploy only Monitoring (Prometheus & Grafana)"
            echo "  --locust-only        Deploy only Locust (load testing)"
            echo "  --api                Deploy API (can be combined with other options)"
            echo "  --ui                 Deploy UI (can be combined with other options)"
            echo "  --monitoring         Deploy Monitoring (can be combined with other options)"
            echo "  --locust             Deploy Locust (can be combined with other options)"
            echo "  --skip-api           Skip API deployment"
            echo "  --skip-ui            Skip UI deployment"
            echo "  --skip-monitoring    Skip Monitoring deployment"
            echo "  --with-locust        Include Locust deployment (alias for --locust)"
            echo "  --skip-locust        Skip Locust deployment"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "By default, API, UI, and Monitoring are deployed. Locust is optional."
            echo ""
            echo "Examples:"
            echo "  $0                           # Deploy API, UI, and Monitoring (default)"
            echo "  $0 --all                     # Deploy all services including Locust"
            echo "  $0 --api --ui                # Deploy only API and UI"
            echo "  $0 --api-only                # Deploy only API"
            echo "  $0 --monitoring --locust     # Deploy only Monitoring and Locust"
            echo "  $0 --skip-ui                 # Deploy API and Monitoring (skip UI)"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If individual flags (--api, --ui, etc.) were used without --all or --*-only,
# reset defaults and only deploy what was explicitly requested
# This ensures that when using --api --ui, only API and UI are deployed, not Monitoring (default)
if [ "$EXPLICIT_ONLY_USED" = false ] && ([ "$API_FLAG_USED" = true ] || [ "$UI_FLAG_USED" = true ] || [ "$MONITORING_FLAG_USED" = true ] || [ "$LOCUST_FLAG_USED" = true ]); then
    # Save the final state after all flags are processed (includes skip flags)
    # At this point, all flags have been processed, so DEPLOY_* reflects the final state
    FINAL_API=$DEPLOY_API
    FINAL_UI=$DEPLOY_UI
    FINAL_MONITORING=$DEPLOY_MONITORING
    FINAL_LOCUST=$DEPLOY_LOCUST
    
    # Reset all to false (disable defaults)
    DEPLOY_API=false
    DEPLOY_UI=false
    DEPLOY_MONITORING=false
    DEPLOY_LOCUST=false
    
    # Enable only services where:
    # 1. Individual flag was used (--api, --ui, etc.)
    # 2. Final state is true (service wasn't skipped or was re-enabled)
    if [ "$API_FLAG_USED" = true ] && [ "$FINAL_API" = true ]; then
        DEPLOY_API=true
    fi
    if [ "$UI_FLAG_USED" = true ] && [ "$FINAL_UI" = true ]; then
        DEPLOY_UI=true
    fi
    if [ "$MONITORING_FLAG_USED" = true ] && [ "$FINAL_MONITORING" = true ]; then
        DEPLOY_MONITORING=true
    fi
    if [ "$LOCUST_FLAG_USED" = true ] && [ "$FINAL_LOCUST" = true ]; then
        DEPLOY_LOCUST=true
    fi
fi

echo "=========================================="
echo "Deploying to Minikube"
echo "=========================================="
echo ""
echo "Deployment plan:"
if [ "$DEPLOY_API" = true ]; then
    echo -e "  ${GREEN}✓${NC} API"
else
    echo -e "  ${YELLOW}⊘${NC} API (skipped)"
fi
if [ "$DEPLOY_UI" = true ]; then
    echo -e "  ${GREEN}✓${NC} UI"
else
    echo -e "  ${YELLOW}⊘${NC} UI (skipped)"
fi
if [ "$DEPLOY_MONITORING" = true ]; then
    echo -e "  ${GREEN}✓${NC} Monitoring (Prometheus & Grafana)"
else
    echo -e "  ${YELLOW}⊘${NC} Monitoring (skipped)"
fi
if [ "$DEPLOY_LOCUST" = true ]; then
    echo -e "  ${GREEN}✓${NC} Locust (Load Testing)"
else
    echo -e "  ${YELLOW}⊘${NC} Locust (skipped)"
fi
echo ""

# Ask for confirmation before proceeding
echo -e "${YELLOW}This will deploy the selected services to Minikube.${NC}"
echo ""
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Deployment cancelled by user.${NC}"
    exit 0
fi
echo ""

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

# Set Docker environment to use Minikube's Docker daemon (needed for API, UI, and Locust)
if [ "$DEPLOY_API" = true ] || [ "$DEPLOY_UI" = true ] || [ "$DEPLOY_LOCUST" = true ]; then
    echo -e "${YELLOW}Setting Docker environment to Minikube...${NC}"
    eval $(minikube -p minikube docker-env)
fi

# ============================================================================
# Deploy API
# ============================================================================
if [ "$DEPLOY_API" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Deploying Diabetes API"
    echo "==========================================${NC}"
    
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
    echo "  Manifests directory: $API_MANIFESTS_DIR"
    cd "$K8S_DIR"
    
    # Verify YAML files exist
    if [ ! -f "$API_MANIFESTS_DIR/api-deployment.yaml" ]; then
        echo -e "${RED}Error: api-deployment.yaml not found in $API_MANIFESTS_DIR${NC}"
        exit 1
    fi
    
    if [ ! -f "$API_MANIFESTS_DIR/api-service.yaml" ]; then
        echo -e "${RED}Error: api-service.yaml not found in $API_MANIFESTS_DIR${NC}"
        exit 1
    fi
    
    # Apply deployment
    $KUBECTL_CMD apply -f "$API_MANIFESTS_DIR/api-deployment.yaml"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to apply deployment${NC}"
        exit 1
    fi
    
    # Apply service (NodePort for Minikube)
    $KUBECTL_CMD apply -f "$API_MANIFESTS_DIR/api-service.yaml"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to apply service${NC}"
        exit 1
    fi
    
    # Optionally apply LoadBalancer service (for cloud deployments)
    if [ -f "$API_MANIFESTS_DIR/api-service-loadbalancer.yaml" ]; then
        echo -e "${YELLOW}Note: LoadBalancer service found. Apply it manually if needed for cloud deployments:${NC}"
        echo "  $KUBECTL_CMD apply -f $API_MANIFESTS_DIR/api-service-loadbalancer.yaml"
    fi
    
    # Apply HPA if metrics server is available
    if [ -f "$API_MANIFESTS_DIR/api-hpa.yaml" ]; then
        echo -e "${YELLOW}Applying Horizontal Pod Autoscaler...${NC}"
        # Check if metrics-server is available (required for HPA)
        if $KUBECTL_CMD get deployment metrics-server -n kube-system &> /dev/null; then
            $KUBECTL_CMD apply -f "$API_MANIFESTS_DIR/api-hpa.yaml"
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✓ HPA applied successfully${NC}"
            else
                echo -e "${YELLOW}Warning: Failed to apply HPA. Metrics server might not be ready.${NC}"
                echo "  Enable metrics server in Minikube: minikube addons enable metrics-server"
            fi
        else
            echo -e "${YELLOW}Warning: Metrics server not found. HPA requires metrics-server.${NC}"
            echo "  Enable metrics server: minikube addons enable metrics-server"
            echo "  Then apply HPA manually: $KUBECTL_CMD apply -f $API_MANIFESTS_DIR/api-hpa.yaml"
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
        echo -e "${GREEN}✓ API deployment is ready${NC}"
    fi
fi

# ============================================================================
# Deploy UI
# ============================================================================
if [ "$DEPLOY_UI" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Deploying Diabetes UI"
    echo "==========================================${NC}"
    
    # Check if API service exists
    echo -e "${YELLOW}Checking if API service is available...${NC}"
    if ! $KUBECTL_CMD get service diabetes-api &> /dev/null; then
        echo -e "${YELLOW}Warning: diabetes-api service not found.${NC}"
        echo -e "${YELLOW}The UI requires the API to be deployed first.${NC}"
        echo "Please deploy the API first or use: $0 --api-only"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ API service found${NC}"
    fi
    
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
    echo "  Manifests directory: $UI_MANIFESTS_DIR"
    cd "$K8S_DIR"
    
    # Verify YAML files exist
    if [ ! -f "$UI_MANIFESTS_DIR/ui-deployment.yaml" ]; then
        echo -e "${RED}Error: ui-deployment.yaml not found in $UI_MANIFESTS_DIR${NC}"
        exit 1
    fi
    
    if [ ! -f "$UI_MANIFESTS_DIR/ui-service.yaml" ]; then
        echo -e "${RED}Error: ui-service.yaml not found in $UI_MANIFESTS_DIR${NC}"
        exit 1
    fi
    
    # Apply deployment
    $KUBECTL_CMD apply -f "$UI_MANIFESTS_DIR/ui-deployment.yaml"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to apply deployment${NC}"
        exit 1
    fi
    
    # Apply service
    $KUBECTL_CMD apply -f "$UI_MANIFESTS_DIR/ui-service.yaml"
    
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
        echo -e "${GREEN}✓ UI deployment is ready${NC}"
    fi
fi

# ============================================================================
# Deploy Monitoring (Prometheus & Grafana)
# ============================================================================
if [ "$DEPLOY_MONITORING" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Deploying Monitoring Stack (Prometheus & Grafana)"
    echo "==========================================${NC}"
    
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
    
    if [ ! -f "$GRAFANA_MANIFESTS_DIR/grafana-dashboards-configmap.yaml" ]; then
        echo -e "${RED}Error: grafana-dashboards-configmap.yaml not found${NC}"
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
    $KUBECTL_CMD apply -f "$GRAFANA_MANIFESTS_DIR/grafana-dashboards-configmap.yaml"
    $KUBECTL_CMD apply -f "$GRAFANA_MANIFESTS_DIR/grafana-deployment.yaml"
    $KUBECTL_CMD apply -f "$GRAFANA_MANIFESTS_DIR/grafana-service.yaml"
    
    echo -e "${GREEN}✓ Grafana manifests applied${NC}"
    
    # Wait for deployments to be ready
    echo ""
    echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
    $KUBECTL_CMD wait --for=condition=available --timeout=120s deployment/prometheus || echo -e "${YELLOW}Warning: Prometheus deployment might not be ready yet${NC}"
    $KUBECTL_CMD wait --for=condition=available --timeout=120s deployment/grafana || echo -e "${YELLOW}Warning: Grafana deployment might not be ready yet${NC}"
    
    echo -e "${GREEN}✓ Monitoring stack deployment complete${NC}"
fi

# ============================================================================
# Deploy Locust (Load Testing)
# ============================================================================
if [ "$DEPLOY_LOCUST" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Deploying Locust (Load Testing)"
    echo "==========================================${NC}"
    
    # Check if API service exists (Locust needs the API to test)
    echo -e "${YELLOW}Checking if API service is available...${NC}"
    if ! $KUBECTL_CMD get service diabetes-api &> /dev/null; then
        echo -e "${YELLOW}Warning: diabetes-api service not found.${NC}"
        echo -e "${YELLOW}Locust requires the API to be deployed first.${NC}"
        echo "Please deploy the API first or use: $0 --api-only"
        echo ""
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}✓ API service found${NC}"
    fi
    
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
    echo "  Manifests directory: $LOCUST_MANIFESTS_DIR"
    cd "$K8S_DIR"
    
    # Verify YAML files exist
    if [ ! -f "$LOCUST_MANIFESTS_DIR/locust-deployment.yaml" ]; then
        echo -e "${RED}Error: locust-deployment.yaml not found in $LOCUST_MANIFESTS_DIR${NC}"
        exit 1
    fi
    
    if [ ! -f "$LOCUST_MANIFESTS_DIR/locust-service.yaml" ]; then
        echo -e "${RED}Error: locust-service.yaml not found in $LOCUST_MANIFESTS_DIR${NC}"
        exit 1
    fi
    
    # Apply deployment
    $KUBECTL_CMD apply -f "$LOCUST_MANIFESTS_DIR/locust-deployment.yaml"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Failed to apply deployment${NC}"
        exit 1
    fi
    
    # Apply service
    $KUBECTL_CMD apply -f "$LOCUST_MANIFESTS_DIR/locust-service.yaml"
    
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
        echo -e "${GREEN}✓ Locust deployment is ready${NC}"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${GREEN}=========================================="
echo "Deployment Complete!"
echo "==========================================${NC}"
echo ""

MINIKUBE_IP=$(minikube ip)

echo "Service Information:"
echo "  Namespace: default"
echo ""

if [ "$DEPLOY_API" = true ]; then
    echo "API Service:"
    echo "  Service Name: diabetes-api"
    echo "  NodePort: 30080"
    echo "  URL: http://$MINIKUBE_IP:30080"
    echo ""
    echo "  API Endpoints:"
    echo "    Health: http://$MINIKUBE_IP:30080/health"
    echo "    Model Info: http://$MINIKUBE_IP:30080/model-info"
    echo "    Predict: http://$MINIKUBE_IP:30080/predict (POST)"
    echo "    Metrics: http://$MINIKUBE_IP:30080/metrics"
    echo ""
    echo "  Performance Optimizations:"
    echo "    - Multiple replicas: 3 (configurable in deployment)"
    echo "    - Multiple workers per pod: 2 (configurable via UVICORN_WORKERS env var)"
    echo "    - HPA: Auto-scales based on CPU/memory (if metrics-server is enabled)"
    echo "    - Async endpoints: Better concurrency handling"
    echo "    - Resource limits: 2Gi memory, 2000m CPU per pod"
    echo ""
fi

if [ "$DEPLOY_UI" = true ]; then
    echo "UI Service:"
    echo "  Service Name: diabetes-ui"
    echo "  NodePort: 30085"
    echo "  URL: http://$MINIKUBE_IP:30085"
    echo ""
fi

if [ "$DEPLOY_MONITORING" = true ]; then
    echo "Monitoring Services:"
    echo "  Prometheus:"
    echo "    Service Name: prometheus"
    echo "    NodePort: 30090"
    echo "    URL: http://$MINIKUBE_IP:30090"
    echo "  Grafana:"
    echo "    Service Name: grafana"
    echo "    NodePort: 30300"
    echo "    URL: http://$MINIKUBE_IP:30300"
    echo "    Username: admin"
    echo "    Password: admin"
    echo ""
    echo "  Prometheus is configured to scrape metrics from:"
    echo "    - diabetes-api:8000/metrics"
    echo ""
fi

if [ "$DEPLOY_LOCUST" = true ]; then
    echo "Locust Service:"
    echo "  Service Name: locust"
    echo "  NodePort: 30189"
    echo "  URL: http://$MINIKUBE_IP:30189"
    echo ""
    echo "  Locust is configured to test:"
    echo "    Target API: http://diabetes-api:8000"
    echo ""
fi

echo "Quick Commands:"
if [ "$DEPLOY_API" = true ]; then
    echo "  API logs: $KUBECTL_CMD logs -l app=diabetes-api -f"
    echo "  API status: $KUBECTL_CMD get pods -l app=diabetes-api"
    echo "  API HPA: $KUBECTL_CMD get hpa diabetes-api-hpa"
    echo "  API scale: $KUBECTL_CMD scale deployment diabetes-api --replicas=5"
    echo "  API metrics: $KUBECTL_CMD top pods -l app=diabetes-api"
fi
if [ "$DEPLOY_UI" = true ]; then
    echo "  UI logs: $KUBECTL_CMD logs -l app=diabetes-ui -f"
    echo "  UI status: $KUBECTL_CMD get pods -l app=diabetes-ui"
fi
if [ "$DEPLOY_MONITORING" = true ]; then
    echo "  Prometheus logs: $KUBECTL_CMD logs -l app=prometheus -f"
    echo "  Grafana logs: $KUBECTL_CMD logs -l app=grafana -f"
    echo "  Prometheus status: $KUBECTL_CMD get pods -l app=prometheus"
    echo "  Grafana status: $KUBECTL_CMD get pods -l app=grafana"
fi
if [ "$DEPLOY_LOCUST" = true ]; then
    echo "  Locust logs: $KUBECTL_CMD logs -l app=locust -f"
    echo "  Locust status: $KUBECTL_CMD get pods -l app=locust"
fi
echo ""

if [ "$DEPLOY_API" = true ]; then
    echo -e "${YELLOW}Notes:${NC}"
    echo "  - Ensure MLflow is running on Docker and accessible at:"
    echo "    http://host.minikube.internal:8002"
    echo ""
    echo "  - For HPA to work, enable metrics-server:"
    echo "    minikube addons enable metrics-server"
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
fi

echo ""

