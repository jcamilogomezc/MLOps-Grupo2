#!/bin/bash

# Simple script to start port forwarding from VM IP to Minikube
# This forwards:
#   - Port 8000 on the VM to port 30080 on Minikube (API service)
#   - Port 8010 on the VM to port 30085 on Minikube (UI service)

VM_IP="${VM_IP:-10.43.100.95}"
API_EXTERNAL_PORT="${API_EXTERNAL_PORT:-8000}"
UI_EXTERNAL_PORT="${UI_EXTERNAL_PORT:-8010}"

# Get Minikube IP
MINIKUBE_IP=$(minikube ip 2>/dev/null)
if [ -z "$MINIKUBE_IP" ]; then
    echo "Error: Minikube is not running"
    echo "Start Minikube with: minikube start"
    exit 1
fi

API_MINIKUBE_PORT=30080
UI_MINIKUBE_PORT=30085

# Cleanup function to kill background processes on exit
cleanup() {
    echo ""
    echo "Stopping port forwarding..."
    if [ -n "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    if [ -n "$UI_PID" ]; then
        kill $UI_PID 2>/dev/null
    fi
    exit 0
}

# Set up trap to call cleanup on script exit
trap cleanup SIGINT SIGTERM

echo "=========================================="
echo "Port Forwarding Setup"
echo "=========================================="
echo "API Forwarding:"
echo "  From: $VM_IP:$API_EXTERNAL_PORT"
echo "  To: $MINIKUBE_IP:$API_MINIKUBE_PORT"
echo ""
echo "UI Forwarding:"
echo "  From: $VM_IP:$UI_EXTERNAL_PORT"
echo "  To: $MINIKUBE_IP:$UI_MINIKUBE_PORT"
echo ""
echo "Access Services:"
echo "  API: http://$VM_IP:$API_EXTERNAL_PORT"
echo "  UI:  http://$VM_IP:$UI_EXTERNAL_PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Check if socat is available
if command -v socat &> /dev/null; then
    echo "Using socat for port forwarding..."
    echo ""
    
    # Start API port forwarding in background
    socat TCP-LISTEN:$API_EXTERNAL_PORT,fork,reuseaddr TCP:$MINIKUBE_IP:$API_MINIKUBE_PORT &
    API_PID=$!
    echo "API port forwarding started (PID: $API_PID)"
    
    # Start UI port forwarding in background
    socat TCP-LISTEN:$UI_EXTERNAL_PORT,fork,reuseaddr TCP:$MINIKUBE_IP:$UI_MINIKUBE_PORT &
    UI_PID=$!
    echo "UI port forwarding started (PID: $UI_PID)"
    echo ""
    echo "Port forwarding is active. Press Ctrl+C to stop."
    echo ""
    
    # Wait for background processes
    wait
else
    echo "Error: socat is not installed"
    echo ""
    
    # Detect package manager and provide appropriate install command
    if command -v dnf &> /dev/null; then
        PKG_MGR="dnf"
        INSTALL_CMD="sudo dnf install -y socat"
    elif command -v yum &> /dev/null; then
        PKG_MGR="yum"
        INSTALL_CMD="sudo yum install -y socat"
    elif command -v apt-get &> /dev/null; then
        PKG_MGR="apt-get"
        INSTALL_CMD="sudo apt-get install -y socat"
    else
        PKG_MGR="unknown"
        INSTALL_CMD=""
    fi
    
    if [ "$PKG_MGR" != "unknown" ]; then
        echo "Install socat with:"
        echo "  $INSTALL_CMD"
        echo ""
    else
        echo "Install socat using your system's package manager"
        echo ""
    fi
    
    echo "Or use alternative methods:"
    echo ""
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    K8S_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
    MANIFESTS_DIR="$K8S_DIR/manifests/api"
    echo "1. Use minikube tunnel (no installation needed):"
    echo "   kubectl apply -f $MANIFESTS_DIR/api-service-loadbalancer.yaml"
    echo "   minikube tunnel"
    echo ""
    echo "2. Use iptables port forwarding (requires root):"
    echo "   # API forwarding:"
    echo "   sudo iptables -t nat -A PREROUTING -p tcp --dport $API_EXTERNAL_PORT -j DNAT --to-destination $MINIKUBE_IP:$API_MINIKUBE_PORT"
    echo "   sudo iptables -t nat -A POSTROUTING -p tcp -d $MINIKUBE_IP --dport $API_MINIKUBE_PORT -j MASQUERADE"
    echo "   # UI forwarding:"
    echo "   sudo iptables -t nat -A PREROUTING -p tcp --dport $UI_EXTERNAL_PORT -j DNAT --to-destination $MINIKUBE_IP:$UI_MINIKUBE_PORT"
    echo "   sudo iptables -t nat -A POSTROUTING -p tcp -d $MINIKUBE_IP --dport $UI_MINIKUBE_PORT -j MASQUERADE"
    echo ""
    echo "3. Use Python port forwarder (no installation needed):"
    echo "   python3 $SCRIPT_DIR/port-forward-python.py"
    echo ""
    
    # Check if Python 3 is available and offer to use it
    if command -v python3 &> /dev/null; then
        echo "Python 3 is available! You can use the Python port forwarder:"
        echo "   python3 $SCRIPT_DIR/port-forward-python.py"
        echo ""
        read -p "Do you want to use Python port forwarder instead? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 "$SCRIPT_DIR/port-forward-python.py"
            exit 0
        fi
    fi
    
    exit 1
fi

