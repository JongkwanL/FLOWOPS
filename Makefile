# FlowOps Makefile

# Variables
BINARY_NAME := flowops
VERSION := $(shell git describe --tags --always --dirty)
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')
LDFLAGS := -ldflags "-X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME)"

# Go related variables
GOCMD := go
GOBUILD := $(GOCMD) build
GOCLEAN := $(GOCMD) clean
GOTEST := $(GOCMD) test
GOGET := $(GOCMD) get
GOMOD := $(GOCMD) mod
GOTOOL := $(GOCMD) tool

# Build targets
.PHONY: all build clean test deps help

all: clean deps test build

build: build-cli build-monitor build-gateway build-drift ## Build all binaries

build-cli: ## Build the CLI binary
	@echo "Building $(BINARY_NAME)..."
	$(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME) ./cmd/flowops

build-monitor: ## Build the monitoring agent
	@echo "Building monitor agent..."
	$(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME)-monitor ./cmd/monitor

build-gateway: ## Build the API gateway
	@echo "Building API gateway..."
	$(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME)-gateway ./cmd/gateway

build-drift: ## Build the drift detection service
	@echo "Building drift detection service..."
	$(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME)-drift ./cmd/drift

build-linux: ## Build for Linux
	@echo "Building $(BINARY_NAME) for Linux..."
	GOOS=linux GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME)-linux-amd64 ./cmd/flowops

build-darwin: ## Build for macOS
	@echo "Building $(BINARY_NAME) for macOS..."
	GOOS=darwin GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME)-darwin-amd64 ./cmd/flowops
	GOOS=darwin GOARCH=arm64 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME)-darwin-arm64 ./cmd/flowops

build-windows: ## Build for Windows
	@echo "Building $(BINARY_NAME) for Windows..."
	GOOS=windows GOARCH=amd64 $(GOBUILD) $(LDFLAGS) -o bin/$(BINARY_NAME)-windows-amd64.exe ./cmd/flowops

build-all: build-linux build-darwin build-windows ## Build for all platforms

clean: ## Remove build artifacts
	@echo "Cleaning..."
	$(GOCLEAN)
	rm -rf bin/

test: ## Run tests
	@echo "Running tests..."
	$(GOTEST) -v ./...

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	$(GOTEST) -v -coverprofile=coverage.out ./...
	$(GOTOOL) cover -html=coverage.out -o coverage.html

deps: ## Download dependencies
	@echo "Downloading dependencies..."
	$(GOMOD) download
	$(GOMOD) tidy

fmt: ## Format Go code
	@echo "Formatting code..."
	$(GOCMD) fmt ./...

lint: ## Run linter
	@echo "Running linter..."
	golangci-lint run

vet: ## Run go vet
	@echo "Running go vet..."
	$(GOCMD) vet ./...

install: build ## Install the binary
	@echo "Installing $(BINARY_NAME)..."
	cp bin/$(BINARY_NAME) $(GOPATH)/bin/$(BINARY_NAME)

# Docker related targets
docker-build: ## Build Docker image
	./bin/$(BINARY_NAME) build --push=false

docker-push: ## Build and push Docker image  
	./bin/$(BINARY_NAME) build --push=true

# Development targets
dev-setup: deps ## Setup development environment
	@echo "Setting up development environment..."
	@if ! command -v golangci-lint >/dev/null 2>&1; then \
		echo "Installing golangci-lint..."; \
		curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(shell go env GOPATH)/bin v1.54.2; \
	fi

run: build ## Build and run
	./bin/$(BINARY_NAME)

run-help: build ## Show CLI help
	./bin/$(BINARY_NAME) --help

# MLOps workflow shortcuts
mlops-build: build ## Build ML containers
	./bin/$(BINARY_NAME) build all

mlops-test: build ## Run ML tests
	./bin/$(BINARY_NAME) test all --coverage

mlops-deploy-staging: build ## Deploy to staging
	./bin/$(BINARY_NAME) deploy staging --wait

mlops-deploy-prod: build ## Deploy to production
	./bin/$(BINARY_NAME) deploy production --argo-sync --wait

mlops-monitor: build ## Monitor all services
	./bin/$(BINARY_NAME) monitor all --watch

# Release targets
release-prep: clean deps test build-all ## Prepare release
	@echo "Release preparation complete"

.PHONY: help
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)