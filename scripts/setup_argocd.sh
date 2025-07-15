#!/bin/bash
# ArgoCD setup script for FlowOps MLOps platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ARGOCD_NAMESPACE="argocd"
ARGOCD_VERSION="v2.9.3"  # Use stable version
FLOWOPS_NAMESPACE="flowops-production"
STAGING_NAMESPACE="flowops-staging"

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to wait for deployment
wait_for_deployment() {
    local namespace=$1
    local deployment=$2
    local timeout=${3:-300}
    
    print_message "$YELLOW" "Waiting for deployment $deployment in namespace $namespace..."
    
    kubectl wait --for=condition=available \
        --timeout=${timeout}s \
        deployment/$deployment \
        -n $namespace
}

# Function to check if kubectl is available
check_prerequisites() {
    print_message "$BLUE" "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        print_message "$RED" "Error: kubectl is not installed"
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        print_message "$YELLOW" "Warning: helm is not installed (optional)"
    fi
    
    # Test cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        print_message "$RED" "Error: Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    print_message "$GREEN" "✓ Prerequisites check passed"
}

# Function to install ArgoCD
install_argocd() {
    print_message "$GREEN" "=== Installing ArgoCD ==="
    
    # Create namespace
    kubectl create namespace $ARGOCD_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Install ArgoCD using official manifests
    print_message "$BLUE" "Installing ArgoCD $ARGOCD_VERSION..."
    kubectl apply -n $ARGOCD_NAMESPACE -f \
        https://raw.githubusercontent.com/argoproj/argo-cd/$ARGOCD_VERSION/manifests/install.yaml
    
    # Wait for ArgoCD components to be ready
    print_message "$YELLOW" "Waiting for ArgoCD components to be ready..."
    wait_for_deployment $ARGOCD_NAMESPACE argocd-server
    wait_for_deployment $ARGOCD_NAMESPACE argocd-repo-server
    wait_for_deployment $ARGOCD_NAMESPACE argocd-application-controller
    
    print_message "$GREEN" "✓ ArgoCD installed successfully"
}

# Function to configure ArgoCD access
configure_argocd_access() {
    print_message "$GREEN" "=== Configuring ArgoCD Access ==="
    
    # Patch ArgoCD server service to LoadBalancer (for cloud) or NodePort (for local)
    if [[ "${ARGOCD_EXTERNAL_ACCESS}" == "loadbalancer" ]]; then
        print_message "$BLUE" "Configuring LoadBalancer access..."
        kubectl patch svc argocd-server -n $ARGOCD_NAMESPACE -p '{"spec": {"type": "LoadBalancer"}}'
    else
        print_message "$BLUE" "Configuring NodePort access..."
        kubectl patch svc argocd-server -n $ARGOCD_NAMESPACE -p '{"spec": {"type": "NodePort"}}'
    fi
    
    # Wait for service to get external IP or NodePort
    print_message "$YELLOW" "Waiting for ArgoCD server to be accessible..."
    sleep 30
    
    # Get ArgoCD admin password
    print_message "$BLUE" "Retrieving ArgoCD admin password..."
    ARGOCD_PASSWORD=$(kubectl -n $ARGOCD_NAMESPACE get secret argocd-initial-admin-secret \
        -o jsonpath="{.data.password}" | base64 -d)
    
    # Get ArgoCD server URL
    if [[ "${ARGOCD_EXTERNAL_ACCESS}" == "loadbalancer" ]]; then
        ARGOCD_SERVER=$(kubectl get svc argocd-server -n $ARGOCD_NAMESPACE \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || \
            kubectl get svc argocd-server -n $ARGOCD_NAMESPACE \
            -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        ARGOCD_URL="https://${ARGOCD_SERVER}"
    else
        NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="ExternalIP")].address}' || \
                  kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
        NODE_PORT=$(kubectl get svc argocd-server -n $ARGOCD_NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
        ARGOCD_URL="https://${NODE_IP}:${NODE_PORT}"
    fi
    
    print_message "$GREEN" "✓ ArgoCD Access Configuration:"
    print_message "$GREEN" "   URL: $ARGOCD_URL"
    print_message "$GREEN" "   Username: admin"
    print_message "$GREEN" "   Password: $ARGOCD_PASSWORD"
}

# Function to install ArgoCD CLI
install_argocd_cli() {
    print_message "$GREEN" "=== Installing ArgoCD CLI ==="
    
    if command -v argocd &> /dev/null; then
        print_message "$YELLOW" "ArgoCD CLI already installed"
        return
    fi
    
    # Download and install ArgoCD CLI
    print_message "$BLUE" "Downloading ArgoCD CLI..."
    
    case $(uname -s) in
        Darwin)
            PLATFORM="darwin"
            ;;
        Linux)
            PLATFORM="linux"
            ;;
        *)
            print_message "$RED" "Unsupported platform: $(uname -s)"
            return
            ;;
    esac
    
    case $(uname -m) in
        x86_64)
            ARCH="amd64"
            ;;
        arm64|aarch64)
            ARCH="arm64"
            ;;
        *)
            print_message "$RED" "Unsupported architecture: $(uname -m)"
            return
            ;;
    esac
    
    curl -sSL -o argocd-${PLATFORM}-${ARCH} \
        https://github.com/argoproj/argo-cd/releases/download/${ARGOCD_VERSION}/argocd-${PLATFORM}-${ARCH}
    
    chmod +x argocd-${PLATFORM}-${ARCH}
    sudo mv argocd-${PLATFORM}-${ARCH} /usr/local/bin/argocd
    
    print_message "$GREEN" "✓ ArgoCD CLI installed"
}

# Function to apply FlowOps ArgoCD configurations
apply_flowops_configs() {
    print_message "$GREEN" "=== Applying FlowOps ArgoCD Configurations ==="
    
    # Create target namespaces
    kubectl create namespace $FLOWOPS_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace $STAGING_NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply ArgoCD project
    print_message "$BLUE" "Creating ArgoCD project..."
    kubectl apply -f infrastructure/argocd/project.yaml
    
    # Apply repository configuration (update with actual repo URL first)
    if [[ -n "${GITHUB_REPO_URL}" ]]; then
        sed "s|https://github.com/flowops/flowops|${GITHUB_REPO_URL}|g" \
            infrastructure/argocd/repository.yaml | kubectl apply -f -
    else
        print_message "$YELLOW" "Warning: GITHUB_REPO_URL not set, using default repository URL"
        kubectl apply -f infrastructure/argocd/repository.yaml
    fi
    
    # Apply sync policies
    kubectl apply -f infrastructure/argocd/sync-policy.yaml
    
    # Apply applications
    print_message "$BLUE" "Creating ArgoCD applications..."
    kubectl apply -f infrastructure/argocd/application.yaml
    kubectl apply -f infrastructure/argocd/application-staging.yaml
    
    # Apply ApplicationSets if enabled
    if [[ "${ENABLE_APPLICATIONSETS}" == "true" ]]; then
        print_message "$BLUE" "Creating ApplicationSets..."
        kubectl apply -f infrastructure/argocd/applicationset.yaml
    fi
    
    print_message "$GREEN" "✓ FlowOps ArgoCD configurations applied"
}

# Function to configure RBAC
configure_rbac() {
    print_message "$GREEN" "=== Configuring RBAC ==="
    
    # Create service account for ArgoCD applications
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: flowops-argocd
  namespace: $ARGOCD_NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: flowops-argocd-controller
rules:
- apiGroups: [""]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["apps"]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["argoproj.io"]
  resources: ["*"]
  verbs: ["*"]
- apiGroups: ["monitoring.coreos.com"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: flowops-argocd-controller
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: flowops-argocd-controller
subjects:
- kind: ServiceAccount
  name: flowops-argocd
  namespace: $ARGOCD_NAMESPACE
EOF
    
    print_message "$GREEN" "✓ RBAC configured"
}

# Function to verify installation
verify_installation() {
    print_message "$GREEN" "=== Verifying Installation ==="
    
    # Check ArgoCD components
    print_message "$BLUE" "Checking ArgoCD components..."
    kubectl get pods -n $ARGOCD_NAMESPACE
    
    # Check ArgoCD applications
    print_message "$BLUE" "Checking ArgoCD applications..."
    kubectl get applications -n $ARGOCD_NAMESPACE
    
    # Check if applications are synced
    if command -v argocd &> /dev/null; then
        print_message "$BLUE" "Checking application sync status..."
        
        # Login to ArgoCD (if possible)
        if [[ -n "${ARGOCD_URL}" ]] && [[ -n "${ARGOCD_PASSWORD}" ]]; then
            echo "$ARGOCD_PASSWORD" | argocd login $ARGOCD_URL --username admin --password-stdin --insecure
            argocd app list
        fi
    fi
    
    print_message "$GREEN" "✓ Installation verification complete"
}

# Function to cleanup (for testing)
cleanup_argocd() {
    print_message "$YELLOW" "=== Cleaning up ArgoCD ==="
    
    read -p "Are you sure you want to remove ArgoCD? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace $ARGOCD_NAMESPACE --ignore-not-found
        kubectl delete namespace $FLOWOPS_NAMESPACE --ignore-not-found
        kubectl delete namespace $STAGING_NAMESPACE --ignore-not-found
        print_message "$GREEN" "✓ ArgoCD cleanup complete"
    else
        print_message "$YELLOW" "Cleanup cancelled"
    fi
}

# Main function
main() {
    local action=${1:-"install"}
    
    print_message "$GREEN" "=== FlowOps ArgoCD Setup ==="
    
    case $action in
        "install")
            check_prerequisites
            install_argocd
            configure_argocd_access
            install_argocd_cli
            configure_rbac
            apply_flowops_configs
            verify_installation
            ;;
        "config-only")
            check_prerequisites
            apply_flowops_configs
            ;;
        "verify")
            verify_installation
            ;;
        "cleanup")
            cleanup_argocd
            ;;
        *)
            print_message "$RED" "Error: Unknown action '$action'"
            print_message "$YELLOW" "Valid actions: install, config-only, verify, cleanup"
            exit 1
            ;;
    esac
    
    print_message "$GREEN" "=== Setup Complete ==="
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --external-access)
            ARGOCD_EXTERNAL_ACCESS="$2"
            shift 2
            ;;
        --repo-url)
            GITHUB_REPO_URL="$2"
            shift 2
            ;;
        --enable-applicationsets)
            ENABLE_APPLICATIONSETS="true"
            shift
            ;;
        --namespace)
            ARGOCD_NAMESPACE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [action] [options]"
            echo ""
            echo "Actions:"
            echo "  install      Full ArgoCD installation and configuration (default)"
            echo "  config-only  Apply FlowOps configurations only"
            echo "  verify       Verify installation"
            echo "  cleanup      Remove ArgoCD installation"
            echo ""
            echo "Options:"
            echo "  --external-access TYPE    External access type: loadbalancer or nodeport"
            echo "  --repo-url URL           GitHub repository URL"
            echo "  --enable-applicationsets Enable ApplicationSets"
            echo "  --namespace NAME         ArgoCD namespace (default: argocd)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$ACTION" ]]; then
                ACTION="$1"
            fi
            shift
            ;;
    esac
done

# Run main function
main "${ACTION:-install}"