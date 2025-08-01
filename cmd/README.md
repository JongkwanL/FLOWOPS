# FlowOps CLI Tool

A unified command-line interface for the FlowOps MLOps platform, replacing multiple shell scripts with a single, cross-platform binary.

## Installation

### From Source
```bash
# Clone the repository
git clone https://github.com/flowops/flowops.git
cd flowops

# Build the CLI
make build

# Install globally
make install
```

### Pre-built Binaries
Download the latest release from [GitHub Releases](https://github.com/flowops/flowops/releases).

## Usage

### Basic Commands

```bash
# Show help
flowops --help

# Build containers
flowops build                    # Build runtime image
flowops build serving            # Build serving image
flowops build all --push        # Build and push all images

# Run tests
flowops test                     # Run all tests
flowops test unit --coverage    # Run unit tests with coverage
flowops test smoke              # Run smoke tests

# Deploy to environments
flowops deploy staging          # Deploy to staging
flowops deploy production --argo-sync  # Deploy using ArgoCD

# Monitor services
flowops monitor                 # Check all services once
flowops monitor --watch         # Continuous monitoring
flowops monitor mlflow          # Monitor only MLflow
```

### Configuration

The CLI can be configured using:

1. **Command-line flags**
2. **Configuration file**: `~/.flowops.yaml` or `./.flowops.yaml`
3. **Environment variables**

Example configuration file:
```yaml
# ~/.flowops.yaml
build:
  registry: "ghcr.io"
  org: "your-org"
  push: false

deploy:
  namespace: "flowops-production"
  wait: true
  timeout: "10m"

monitor:
  interval: "30s"
  namespace: "flowops-production"
```

### Environment Variables

```bash
export FLOWOPS_REGISTRY="your-registry.com"
export FLOWOPS_ORG="your-org" 
export FLOWOPS_NAMESPACE="your-namespace"
```

## Command Reference

### `flowops build`

Build container images using nerdctl.

```bash
flowops build [target] [flags]

Targets:
  runtime      - Base runtime image (default)
  serving      - Model serving image
  development  - Development environment image
  all          - Build all targets

Flags:
  --registry string     Container registry (default "ghcr.io")
  --org string         Organization name (default "flowops")
  --image string       Image name (default "mlops-model")
  --version string     Image version (default: git commit)
  --push               Push image after build
  --no-cache           Build without cache
  --target string      Build target (default "runtime")
```

### `flowops test`

Run test suites for the FlowOps platform.

```bash
flowops test [type] [flags]

Types:
  unit         - Run unit tests
  integration  - Run integration tests
  smoke        - Run smoke tests
  security     - Run security tests
  performance  - Run performance tests
  all          - Run all tests (default)

Flags:
  --coverage           Enable coverage reporting
  --parallel int       Number of parallel test jobs (default 4)
  --check-services     Check/start services for integration tests
  --output string      Output directory for test reports (default ".")
  --timeout string     Test timeout (default "10m")
```

### `flowops deploy`

Deploy to target environments using Helm or ArgoCD.

```bash
flowops deploy [environment] [flags]

Environments:
  staging      - Deploy to staging environment
  production   - Deploy to production environment
  local        - Deploy to local development environment

Flags:
  --namespace string      Kubernetes namespace
  --dry-run              Perform a dry run
  --wait                 Wait for deployment to complete (default true)
  --timeout string       Deployment timeout (default "10m")
  --values strings       Additional values files
  --set strings          Set values on command line
  --argo-sync            Trigger ArgoCD sync instead of direct Helm
```

### `flowops monitor`

Monitor FlowOps services health and status.

```bash
flowops monitor [service] [flags]

Services:
  all      - Monitor all services (default)
  mlflow   - Monitor MLflow tracking server
  api      - Monitor model serving API
  k8s      - Monitor Kubernetes deployments
  argocd   - Monitor ArgoCD applications

Flags:
  --watch                 Watch mode - continuously monitor
  --interval duration     Watch interval (default 30s)
  --namespace string      Kubernetes namespace (default "flowops-production")
  --output string         Output format: table, json, yaml (default "table")
```

## Examples

### Complete Workflow

```bash
# 1. Build all container images
flowops build all

# 2. Run comprehensive tests
flowops test all --coverage

# 3. Deploy to staging
flowops deploy staging --wait

# 4. Monitor deployment
flowops monitor k8s --watch

# 5. Deploy to production (using ArgoCD)
flowops deploy production --argo-sync
```

### Development Workflow

```bash
# Build and test quickly
flowops build runtime --no-cache
flowops test unit

# Deploy to local environment
flowops deploy local --namespace flowops-dev

# Monitor during development
flowops monitor api --watch --interval 10s
```

### Production Operations

```bash
# Check service health
flowops monitor all

# Deploy with ArgoCD
flowops deploy production --argo-sync --wait

# Monitor continuously
flowops monitor all --watch --interval 60s
```

## Integration with Existing Tools

The FlowOps CLI replaces and integrates with existing shell scripts:

- `scripts/build_container.sh` → `flowops build`
- `scripts/run_tests.sh` → `flowops test`
- `scripts/setup_argocd.sh` → `flowops deploy --argo-sync`

All existing functionality is preserved while adding cross-platform support and better error handling.

## Troubleshooting

### Common Issues

1. **nerdctl not found**
   ```bash
   # Install nerdctl
   # See: https://github.com/containerd/nerdctl/releases
   ```

2. **kubectl not configured**
   ```bash
   # Configure kubectl
   kubectl config current-context
   ```

3. **Permission denied**
   ```bash
   # Add user to docker group or run with sudo
   sudo usermod -aG docker $USER
   ```

### Debug Mode

Enable verbose output for debugging:
```bash
flowops --verbose build all
flowops -v test unit
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run `make test lint`
6. Submit a pull request

## License

MIT License - see [LICENSE](../LICENSE) file for details.