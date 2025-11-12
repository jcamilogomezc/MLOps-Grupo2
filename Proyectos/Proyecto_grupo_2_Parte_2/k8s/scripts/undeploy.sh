#!/bin/bash

# Script to undeploy services from Kubernetes
# This script can undeploy all components or specific ones based on arguments

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in k8s/scripts/, so go up two levels to get project root
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
API_MANIFESTS_DIR="$K8S_DIR/manifests/api"
UI_MANIFESTS_DIR="$K8S_DIR/manifests/ui"
PROMETHEUS_MANIFESTS_DIR="$K8S_DIR/manifests/prometheus"
GRAFANA_MANIFESTS_DIR="$K8S_DIR/manifests/grafana"
LOCUST_MANIFESTS_DIR="$K8S_DIR/manifests/locust"
DATABASE_MANIFESTS_DIR="$K8S_DIR/manifests/database"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default: undeploy nothing (require explicit selection)
UNDEPLOY_API=false
UNDEPLOY_UI=false
UNDEPLOY_MONITORING=false
UNDEPLOY_LOCUST=false
UNDEPLOY_DATABASES=false
FORCE=false  # Skip confirmation prompts

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            UNDEPLOY_API=true
            UNDEPLOY_UI=true
            UNDEPLOY_MONITORING=true
            UNDEPLOY_LOCUST=true
            UNDEPLOY_DATABASES=true
            shift
            ;;
        --api-only)
            UNDEPLOY_API=true
            shift
            ;;
        --ui-only)
            UNDEPLOY_UI=true
            shift
            ;;
        --monitoring-only)
            UNDEPLOY_MONITORING=true
            shift
            ;;
        --locust-only)
            UNDEPLOY_LOCUST=true
            shift
            ;;
        --databases-only)
            UNDEPLOY_DATABASES=true
            shift
            ;;
        --api)
            UNDEPLOY_API=true
            shift
            ;;
        --ui)
            UNDEPLOY_UI=true
            shift
            ;;
        --monitoring)
            UNDEPLOY_MONITORING=true
            shift
            ;;
        --locust)
            UNDEPLOY_LOCUST=true
            shift
            ;;
        --databases)
            UNDEPLOY_DATABASES=true
            shift
            ;;
        --force|-f)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Undeploy services from Kubernetes"
            echo ""
            echo "Options:"
            echo "  --all                Undeploy all services (API, UI, Monitoring, Locust, Databases)"
            echo "  --api-only           Undeploy only the API"
            echo "  --ui-only            Undeploy only the UI"
            echo "  --monitoring-only    Undeploy only Monitoring (Prometheus & Grafana)"
            echo "  --locust-only        Undeploy only Locust"
            echo "  --databases-only     Undeploy only Databases"
            echo "  --api                Undeploy API (can be combined with other options)"
            echo "  --ui                 Undeploy UI (can be combined with other options)"
            echo "  --monitoring         Undeploy Monitoring (can be combined with other options)"
            echo "  --locust             Undeploy Locust (can be combined with other options)"
            echo "  --databases          Undeploy Databases (can be combined with other options)"
            echo "  --force, -f          Skip confirmation prompts"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --all                    # Undeploy everything"
            echo "  $0 --api --ui               # Undeploy API and UI"
            echo "  $0 --monitoring-only        # Undeploy only monitoring stack"
            echo "  $0 --all --force            # Undeploy everything without confirmation"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if anything is selected for undeployment
if [ "$UNDEPLOY_API" = false ] && [ "$UNDEPLOY_UI" = false ] && [ "$UNDEPLOY_MONITORING" = false ] && [ "$UNDEPLOY_LOCUST" = false ] && [ "$UNDEPLOY_DATABASES" = false ]; then
    echo -e "${YELLOW}No services selected for undeployment${NC}"
    echo ""
    echo "Usage examples:"
    echo "  $0 --all                    # Undeploy all services"
    echo "  $0 --api-only               # Undeploy only the API"
    echo "  $0 --ui-only                # Undeploy only the UI"
    echo "  $0 --monitoring-only        # Undeploy only monitoring stack"
    echo "  $0 --locust-only            # Undeploy only Locust"
    echo "  $0 --databases-only         # Undeploy only databases"
    echo "  $0 --api --ui               # Undeploy API and UI"
    echo "  $0 --all --force            # Undeploy all without confirmation"
    echo ""
    echo "Use --help for full usage information"
    echo ""
    exit 1
fi

echo "=========================================="
echo "Undeploying from Kubernetes"
echo "=========================================="
echo ""
echo "Undeployment plan:"
if [ "$UNDEPLOY_API" = true ]; then
    echo -e "  ${RED}✗${NC} API"
else
    echo -e "  ${YELLOW}⊘${NC} API (skipped)"
fi
if [ "$UNDEPLOY_UI" = true ]; then
    echo -e "  ${RED}✗${NC} UI"
else
    echo -e "  ${YELLOW}⊘${NC} UI (skipped)"
fi
if [ "$UNDEPLOY_MONITORING" = true ]; then
    echo -e "  ${RED}✗${NC} Monitoring (Prometheus & Grafana)"
else
    echo -e "  ${YELLOW}⊘${NC} Monitoring (skipped)"
fi
if [ "$UNDEPLOY_LOCUST" = true ]; then
    echo -e "  ${RED}✗${NC} Locust"
else
    echo -e "  ${YELLOW}⊘${NC} Locust (skipped)"
fi
if [ "$UNDEPLOY_DATABASES" = true ]; then
    echo -e "  ${RED}✗${NC} Databases (PostgreSQL Raw Data & Clean Data)"
else
    echo -e "  ${YELLOW}⊘${NC} Databases (skipped)"
fi
echo ""

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

# Confirmation prompt (unless --force is used)
if [ "$FORCE" = false ]; then
    echo -e "${YELLOW}Warning: This will delete the selected deployments and services.${NC}"
    echo -e "${YELLOW}This action cannot be undone.${NC}"
    echo ""
    read -p "Are you sure you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Undeployment cancelled.${NC}"
        exit 0
    fi
    echo ""
fi

# ============================================================================
# Undeploy API
# ============================================================================
if [ "$UNDEPLOY_API" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Undeploying Diabetes API"
    echo "==========================================${NC}"
    
    # Delete service
    if $KUBECTL_CMD get service diabetes-api &> /dev/null; then
        echo -e "${YELLOW}Deleting API service...${NC}"
        $KUBECTL_CMD delete service diabetes-api
        echo -e "${GREEN}✓ API service deleted${NC}"
    else
        echo -e "${YELLOW}API service not found (may already be deleted)${NC}"
    fi
    
    # Delete deployment
    if $KUBECTL_CMD get deployment diabetes-api &> /dev/null; then
        echo -e "${YELLOW}Deleting API deployment...${NC}"
        $KUBECTL_CMD delete deployment diabetes-api
        echo -e "${GREEN}✓ API deployment deleted${NC}"
    else
        echo -e "${YELLOW}API deployment not found (may already be deleted)${NC}"
    fi
fi

# ============================================================================
# Undeploy UI
# ============================================================================
if [ "$UNDEPLOY_UI" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Undeploying Diabetes UI"
    echo "==========================================${NC}"
    
    # Delete service
    if $KUBECTL_CMD get service diabetes-ui &> /dev/null; then
        echo -e "${YELLOW}Deleting UI service...${NC}"
        $KUBECTL_CMD delete service diabetes-ui
        echo -e "${GREEN}✓ UI service deleted${NC}"
    else
        echo -e "${YELLOW}UI service not found (may already be deleted)${NC}"
    fi
    
    # Delete deployment
    if $KUBECTL_CMD get deployment diabetes-ui &> /dev/null; then
        echo -e "${YELLOW}Deleting UI deployment...${NC}"
        $KUBECTL_CMD delete deployment diabetes-ui
        echo -e "${GREEN}✓ UI deployment deleted${NC}"
    else
        echo -e "${YELLOW}UI deployment not found (may already be deleted)${NC}"
    fi
fi

# ============================================================================
# Undeploy Monitoring (Prometheus & Grafana)
# ============================================================================
if [ "$UNDEPLOY_MONITORING" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Undeploying Monitoring Stack (Prometheus & Grafana)"
    echo "==========================================${NC}"
    
    # Undeploy Grafana
    echo ""
    echo -e "${YELLOW}Undeploying Grafana...${NC}"
    
    # Delete Grafana service
    if $KUBECTL_CMD get service grafana &> /dev/null; then
        echo -e "${YELLOW}Deleting Grafana service...${NC}"
        $KUBECTL_CMD delete service grafana
        echo -e "${GREEN}✓ Grafana service deleted${NC}"
    else
        echo -e "${YELLOW}Grafana service not found (may already be deleted)${NC}"
    fi
    
    # Delete Grafana deployment
    if $KUBECTL_CMD get deployment grafana &> /dev/null; then
        echo -e "${YELLOW}Deleting Grafana deployment...${NC}"
        $KUBECTL_CMD delete deployment grafana
        echo -e "${GREEN}✓ Grafana deployment deleted${NC}"
    else
        echo -e "${YELLOW}Grafana deployment not found (may already be deleted)${NC}"
    fi
    
    # Delete Grafana configmaps
    if $KUBECTL_CMD get configmap grafana-datasources &> /dev/null; then
        echo -e "${YELLOW}Deleting Grafana datasources configmap...${NC}"
        $KUBECTL_CMD delete configmap grafana-datasources
        echo -e "${GREEN}✓ Grafana datasources configmap deleted${NC}"
    else
        echo -e "${YELLOW}Grafana datasources configmap not found (may already be deleted)${NC}"
    fi
    
    if $KUBECTL_CMD get configmap grafana-dashboards &> /dev/null; then
        echo -e "${YELLOW}Deleting Grafana dashboards configmap...${NC}"
        $KUBECTL_CMD delete configmap grafana-dashboards
        echo -e "${GREEN}✓ Grafana dashboards configmap deleted${NC}"
    else
        echo -e "${YELLOW}Grafana dashboards configmap not found (may already be deleted)${NC}"
    fi
    
    # Undeploy Prometheus
    echo ""
    echo -e "${YELLOW}Undeploying Prometheus...${NC}"
    
    # Delete Prometheus service
    if $KUBECTL_CMD get service prometheus &> /dev/null; then
        echo -e "${YELLOW}Deleting Prometheus service...${NC}"
        $KUBECTL_CMD delete service prometheus
        echo -e "${GREEN}✓ Prometheus service deleted${NC}"
    else
        echo -e "${YELLOW}Prometheus service not found (may already be deleted)${NC}"
    fi
    
    # Delete Prometheus deployment
    if $KUBECTL_CMD get deployment prometheus &> /dev/null; then
        echo -e "${YELLOW}Deleting Prometheus deployment...${NC}"
        $KUBECTL_CMD delete deployment prometheus
        echo -e "${GREEN}✓ Prometheus deployment deleted${NC}"
    else
        echo -e "${YELLOW}Prometheus deployment not found (may already be deleted)${NC}"
    fi
    
    # Delete Prometheus configmap
    if $KUBECTL_CMD get configmap prometheus-config &> /dev/null; then
        echo -e "${YELLOW}Deleting Prometheus configmap...${NC}"
        $KUBECTL_CMD delete configmap prometheus-config
        echo -e "${GREEN}✓ Prometheus configmap deleted${NC}"
    else
        echo -e "${YELLOW}Prometheus configmap not found (may already be deleted)${NC}"
    fi
fi

# ============================================================================
# Undeploy Locust
# ============================================================================
if [ "$UNDEPLOY_LOCUST" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Undeploying Locust"
    echo "==========================================${NC}"
    
    # Delete service
    if $KUBECTL_CMD get service locust &> /dev/null; then
        echo -e "${YELLOW}Deleting Locust service...${NC}"
        $KUBECTL_CMD delete service locust
        echo -e "${GREEN}✓ Locust service deleted${NC}"
    else
        echo -e "${YELLOW}Locust service not found (may already be deleted)${NC}"
    fi
    
    # Delete deployment
    if $KUBECTL_CMD get deployment locust &> /dev/null; then
        echo -e "${YELLOW}Deleting Locust deployment...${NC}"
        $KUBECTL_CMD delete deployment locust
        echo -e "${GREEN}✓ Locust deployment deleted${NC}"
    else
        echo -e "${YELLOW}Locust deployment not found (may already be deleted)${NC}"
    fi
fi

# ============================================================================
# Undeploy Databases
# ============================================================================
if [ "$UNDEPLOY_DATABASES" = true ]; then
    echo ""
    echo -e "${BLUE}=========================================="
    echo "Undeploying Databases (PostgreSQL)"
    echo "==========================================${NC}"
    
    # Undeploy PostgreSQL Raw Data
    echo ""
    echo -e "${YELLOW}Undeploying PostgreSQL Raw Data...${NC}"
    
    # Delete service
    if $KUBECTL_CMD get service postgres-raw-data &> /dev/null; then
        echo -e "${YELLOW}Deleting postgres-raw-data service...${NC}"
        $KUBECTL_CMD delete service postgres-raw-data
        echo -e "${GREEN}✓ postgres-raw-data service deleted${NC}"
    else
        echo -e "${YELLOW}postgres-raw-data service not found (may already be deleted)${NC}"
    fi
    
    # Delete deployment
    if $KUBECTL_CMD get deployment postgres-raw-data &> /dev/null; then
        echo -e "${YELLOW}Deleting postgres-raw-data deployment...${NC}"
        $KUBECTL_CMD delete deployment postgres-raw-data
        echo -e "${GREEN}✓ postgres-raw-data deployment deleted${NC}"
    else
        echo -e "${YELLOW}postgres-raw-data deployment not found (may already be deleted)${NC}"
    fi
    
    # Undeploy PostgreSQL Clean Data
    echo ""
    echo -e "${YELLOW}Undeploying PostgreSQL Clean Data...${NC}"
    
    # Delete service
    if $KUBECTL_CMD get service postgres-clean-data &> /dev/null; then
        echo -e "${YELLOW}Deleting postgres-clean-data service...${NC}"
        $KUBECTL_CMD delete service postgres-clean-data
        echo -e "${GREEN}✓ postgres-clean-data service deleted${NC}"
    else
        echo -e "${YELLOW}postgres-clean-data service not found (may already be deleted)${NC}"
    fi
    
    # Delete deployment
    if $KUBECTL_CMD get deployment postgres-clean-data &> /dev/null; then
        echo -e "${YELLOW}Deleting postgres-clean-data deployment...${NC}"
        $KUBECTL_CMD delete deployment postgres-clean-data
        echo -e "${GREEN}✓ postgres-clean-data deployment deleted${NC}"
    else
        echo -e "${YELLOW}postgres-clean-data deployment not found (may already be deleted)${NC}"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${GREEN}=========================================="
echo "Undeployment Complete!"
echo "==========================================${NC}"
echo ""

# Wait a moment for resources to be deleted
echo -e "${YELLOW}Waiting for resources to be fully deleted...${NC}"
sleep 2

# Show remaining resources
echo ""
echo "Remaining deployments:"
$KUBECTL_CMD get deployments 2>/dev/null || echo "  (none)"

echo ""
echo "Remaining services:"
$KUBECTL_CMD get services 2>/dev/null || echo "  (none)"

echo ""
echo "Remaining configmaps:"
$KUBECTL_CMD get configmaps 2>/dev/null | grep -E "(prometheus|grafana)" || echo "  (none)"

echo ""
echo -e "${GREEN}Undeployment finished successfully!${NC}"
echo ""

