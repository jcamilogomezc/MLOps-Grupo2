#!/bin/bash

# Script to help configure MLflow connection from Kubernetes to Docker
# This script helps identify the correct host IP to access Docker services

set -e

echo "=========================================="
echo "MLflow Connection Setup Helper"
echo "=========================================="

# Get Minikube IP
MINIKUBE_IP=$(minikube ip 2>/dev/null || echo "Not running")
echo "Minikube IP: $MINIKUBE_IP"

# Try to get host IP from Minikube
echo ""
echo "Trying to access host from Minikube..."

# Test host.minikube.internal
if curl -s --connect-timeout 2 http://host.minikube.internal:8002/health > /dev/null 2>&1; then
    echo "✓ host.minikube.internal:8002 is accessible"
    MLFLOW_URL="http://host.minikube.internal:8002"
else
    echo "✗ host.minikube.internal:8002 is not accessible"
    
    # Try to get host IP from route
    HOST_IP=$(ip route show | grep -i default | awk '{ print $3}' | head -1)
    if [ -z "$HOST_IP" ]; then
        # Try alternative method
        HOST_IP=$(minikube ssh "route -n get default" 2>/dev/null | grep gateway | awk '{print $2}' || echo "")
    fi
    
    if [ -n "$HOST_IP" ]; then
        echo "Detected host IP: $HOST_IP"
        MLFLOW_URL="http://$HOST_IP:8002"
        
        # Test connection
        if curl -s --connect-timeout 2 http://$HOST_IP:8002/health > /dev/null 2>&1; then
            echo "✓ $HOST_IP:8002 is accessible"
        else
            echo "✗ $HOST_IP:8002 is not accessible"
            echo ""
            SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
            MANIFESTS_DIR="$K8S_DIR/manifests/api"
            echo "Please ensure:"
            echo "  1. MLflow is running on Docker (port 8002)"
            echo "  2. Docker port 8002 is accessible from the host"
            echo "  3. Update $MANIFESTS_DIR/api-deployment.yaml with the correct host IP"
            exit 1
        fi
    else
        echo "Could not determine host IP automatically"
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
        MANIFESTS_DIR="$K8S_DIR/manifests/api"
        echo "Please manually update $MANIFESTS_DIR/api-deployment.yaml with the correct MLFLOW_TRACKING_URI"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Recommended MLflow URL: $MLFLOW_URL"
echo "=========================================="
echo ""
echo "To update the deployment, run:"
echo "  kubectl set env deployment/diabetes-api MLFLOW_TRACKING_URI=$MLFLOW_URL"
echo ""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFESTS_DIR="$K8S_DIR/manifests/api"
echo "Or manually edit $MANIFESTS_DIR/api-deployment.yaml and update:"
echo "  - name: MLFLOW_TRACKING_URI"
echo "    value: \"$MLFLOW_URL\""
echo ""

