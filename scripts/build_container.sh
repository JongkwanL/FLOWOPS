#!/bin/bash
# Container build script using nerdctl for MLOps platform

set -e

# Configuration
REGISTRY=${REGISTRY:-"ghcr.io"}
ORG=${ORG:-"flowops"}
IMAGE_NAME=${IMAGE_NAME:-"mlops-model"}
VERSION=${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo "latest")}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

# Build targets
TARGETS=("runtime" "serving" "development")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Check if nerdctl is installed
check_nerdctl() {
    if ! command -v nerdctl &> /dev/null; then
        print_message "$RED" "Error: nerdctl is not installed"
        print_message "$YELLOW" "Install nerdctl from: https://github.com/containerd/nerdctl/releases"
        exit 1
    fi
    print_message "$GREEN" "✓ nerdctl found: $(nerdctl --version)"
}

# Check if buildkit is available
check_buildkit() {
    if ! nerdctl info 2>/dev/null | grep -q "buildkit"; then
        print_message "$YELLOW" "Warning: BuildKit not detected, builds may be slower"
        print_message "$YELLOW" "Consider installing buildkitd for better performance"
    else
        print_message "$GREEN" "✓ BuildKit available"
    fi
}

# Build container image
build_image() {
    local target=$1
    local tag_suffix=""
    
    case $target in
        "serving")
            tag_suffix="-serving"
            ;;
        "development")
            tag_suffix="-dev"
            ;;
        *)
            tag_suffix=""
            ;;
    esac
    
    local full_tag="${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}${tag_suffix}"
    local latest_tag="${REGISTRY}/${ORG}/${IMAGE_NAME}:latest${tag_suffix}"
    
    print_message "$YELLOW" "Building ${target} stage as ${full_tag}..."
    
    # Build with BuildKit features
    nerdctl build \
        --target ${target} \
        --tag ${full_tag} \
        --tag ${latest_tag} \
        --build-arg BUILD_DATE=${BUILD_DATE} \
        --build-arg VERSION=${VERSION} \
        --label "org.opencontainers.image.created=${BUILD_DATE}" \
        --label "org.opencontainers.image.version=${VERSION}" \
        --label "org.opencontainers.image.source=https://github.com/${ORG}/flowops" \
        --label "org.opencontainers.image.title=FlowOps MLOps Platform" \
        --label "org.opencontainers.image.description=ML model ${target} container" \
        --cache-from type=registry,ref=${REGISTRY}/${ORG}/${IMAGE_NAME}:cache-${target} \
        --cache-to type=registry,ref=${REGISTRY}/${ORG}/${IMAGE_NAME}:cache-${target},mode=max \
        --progress=plain \
        --file Containerfile \
        .
    
    print_message "$GREEN" "✓ Built ${full_tag}"
    
    # Optionally convert to eStargz for lazy pulling (faster in K8s)
    if [[ "${ENABLE_ESTARGZ}" == "true" ]]; then
        print_message "$YELLOW" "Converting to eStargz format..."
        local estargz_tag="${full_tag}-estargz"
        nerdctl image convert \
            --estargz \
            --oci \
            ${full_tag} \
            ${estargz_tag}
        print_message "$GREEN" "✓ Created eStargz variant: ${estargz_tag}"
    fi
}

# Scan image for vulnerabilities
scan_image() {
    local image=$1
    
    if command -v trivy &> /dev/null; then
        print_message "$YELLOW" "Scanning image for vulnerabilities..."
        trivy image --severity HIGH,CRITICAL ${image}
    else
        print_message "$YELLOW" "Trivy not installed, skipping vulnerability scan"
    fi
}

# Push image to registry
push_image() {
    local target=$1
    local tag_suffix=""
    
    case $target in
        "serving")
            tag_suffix="-serving"
            ;;
        "development")
            tag_suffix="-dev"
            ;;
        *)
            tag_suffix=""
            ;;
    esac
    
    local full_tag="${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}${tag_suffix}"
    local latest_tag="${REGISTRY}/${ORG}/${IMAGE_NAME}:latest${tag_suffix}"
    
    print_message "$YELLOW" "Pushing ${full_tag} to registry..."
    
    # Login to registry if credentials are provided
    if [[ -n "${REGISTRY_USERNAME}" ]] && [[ -n "${REGISTRY_PASSWORD}" ]]; then
        echo "${REGISTRY_PASSWORD}" | nerdctl login ${REGISTRY} -u "${REGISTRY_USERNAME}" --password-stdin
    fi
    
    # Push versioned tag
    nerdctl push ${full_tag}
    print_message "$GREEN" "✓ Pushed ${full_tag}"
    
    # Push latest tag
    nerdctl push ${latest_tag}
    print_message "$GREEN" "✓ Pushed ${latest_tag}"
    
    # Push eStargz variant if enabled
    if [[ "${ENABLE_ESTARGZ}" == "true" ]]; then
        local estargz_tag="${full_tag}-estargz"
        nerdctl push ${estargz_tag}
        print_message "$GREEN" "✓ Pushed ${estargz_tag}"
    fi
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    local image=$1
    local output_file="sbom-${VERSION}.json"
    
    if command -v syft &> /dev/null; then
        print_message "$YELLOW" "Generating SBOM..."
        syft ${image} -o json > ${output_file}
        print_message "$GREEN" "✓ SBOM saved to ${output_file}"
    else
        print_message "$YELLOW" "Syft not installed, skipping SBOM generation"
    fi
}

# Sign image with cosign
sign_image() {
    local image=$1
    
    if command -v cosign &> /dev/null && [[ -n "${COSIGN_KEY}" ]]; then
        print_message "$YELLOW" "Signing image..."
        cosign sign --key ${COSIGN_KEY} ${image}
        print_message "$GREEN" "✓ Image signed"
    else
        print_message "$YELLOW" "Cosign not available or key not provided, skipping signing"
    fi
}

# Run container locally for testing
run_local() {
    local target=$1
    local tag_suffix=""
    
    case $target in
        "serving")
            tag_suffix="-serving"
            local port_mapping="-p 8080:8080 -p 8000:8000"
            ;;
        "development")
            tag_suffix="-dev"
            local port_mapping="-p 8888:8888"
            ;;
        *)
            tag_suffix=""
            local port_mapping=""
            ;;
    esac
    
    local full_tag="${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}${tag_suffix}"
    
    print_message "$YELLOW" "Running ${full_tag} locally..."
    
    # Check for GPU support
    local gpu_args=""
    if nerdctl info 2>/dev/null | grep -q "nvidia"; then
        gpu_args="--gpus all"
        print_message "$GREEN" "✓ GPU support detected"
    fi
    
    nerdctl run \
        --rm \
        --name flowops-${target} \
        ${gpu_args} \
        ${port_mapping} \
        -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://localhost:5000} \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        ${full_tag}
}

# Main execution
main() {
    print_message "$GREEN" "=== FlowOps Container Build Script ==="
    
    # Parse arguments
    ACTION=${1:-"build"}
    TARGET=${2:-"runtime"}
    
    # Validate target
    if [[ ! " ${TARGETS[@]} " =~ " ${TARGET} " ]]; then
        print_message "$RED" "Error: Invalid target '${TARGET}'"
        print_message "$YELLOW" "Valid targets: ${TARGETS[@]}"
        exit 1
    fi
    
    # Check prerequisites
    check_nerdctl
    check_buildkit
    
    case $ACTION in
        "build")
            build_image ${TARGET}
            scan_image "${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}"
            ;;
        "push")
            push_image ${TARGET}
            ;;
        "build-push")
            build_image ${TARGET}
            scan_image "${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}"
            push_image ${TARGET}
            ;;
        "run")
            run_local ${TARGET}
            ;;
        "sbom")
            generate_sbom "${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}"
            ;;
        "sign")
            sign_image "${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}"
            ;;
        "all")
            for target in "${TARGETS[@]}"; do
                build_image ${target}
                scan_image "${REGISTRY}/${ORG}/${IMAGE_NAME}:${VERSION}"
                
                if [[ "${PUSH}" == "true" ]]; then
                    push_image ${target}
                fi
            done
            ;;
        *)
            print_message "$RED" "Error: Unknown action '${ACTION}'"
            print_message "$YELLOW" "Valid actions: build, push, build-push, run, sbom, sign, all"
            exit 1
            ;;
    esac
    
    print_message "$GREEN" "=== Build complete ==="
}

# Run main function
main "$@"