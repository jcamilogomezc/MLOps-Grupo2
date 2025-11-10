#!/bin/bash

# Simple script to start port forwarding from VM IP to Minikube
# This forwards port 8000 on the VM to port 30080 on Minikube

VM_IP="${VM_IP:-10.43.100.95}"
EXTERNAL_PORT="${EXTERNAL_PORT:-8000}"

# Get Minikube IP
MINIKUBE_IP=$(minikube ip 2>/dev/null)
if [ -z "$MINIKUBE_IP" ]; then
    echo "Error: Minikube is not running"
    echo "Start Minikube with: minikube start"
    exit 1
fi

MINIKUBE_PORT=30080

echo "=========================================="
echo "Port Forwarding Setup"
echo "=========================================="
echo "From: $VM_IP:$EXTERNAL_PORT"
echo "To: $MINIKUBE_IP:$MINIKUBE_PORT"
echo ""
echo "Access API at: http://$VM_IP:$EXTERNAL_PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Check if socat is available
if command -v socat &> /dev/null; then
    echo "Using socat for port forwarding..."
    echo ""
    socat TCP-LISTEN:$EXTERNAL_PORT,fork,reuseaddr TCP:$MINIKUBE_IP:$MINIKUBE_PORT
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
    echo "   sudo iptables -t nat -A PREROUTING -p tcp --dport $EXTERNAL_PORT -j DNAT --to-destination $MINIKUBE_IP:$MINIKUBE_PORT"
    echo "   sudo iptables -t nat -A POSTROUTING -p tcp -d $MINIKUBE_IP --dport $MINIKUBE_PORT -j MASQUERADE"
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

