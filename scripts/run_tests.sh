#!/bin/bash
# Test runner script for FlowOps MLOps platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_DIR="tests"
COVERAGE_THRESHOLD=80
PARALLEL_JOBS=4

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run tests with coverage
run_tests_with_coverage() {
    local test_type=$1
    local test_path=$2
    
    print_message "$BLUE" "Running ${test_type} tests with coverage..."
    
    pytest \
        ${test_path} \
        --cov=pipelines \
        --cov=mlflow \
        --cov-report=term-missing \
        --cov-report=html:coverage_reports/${test_type} \
        --cov-report=xml:coverage_reports/${test_type}_coverage.xml \
        --cov-fail-under=${COVERAGE_THRESHOLD} \
        --junit-xml=test_reports/${test_type}_results.xml \
        -v \
        --tb=short
}

# Function to run tests without coverage
run_tests_simple() {
    local test_type=$1
    local test_path=$2
    
    print_message "$BLUE" "Running ${test_type} tests..."
    
    pytest \
        ${test_path} \
        --junit-xml=test_reports/${test_type}_results.xml \
        -v \
        --tb=short
}

# Function to setup test environment
setup_test_env() {
    print_message "$YELLOW" "Setting up test environment..."
    
    # Create directories
    mkdir -p coverage_reports
    mkdir -p test_reports
    
    # Install test dependencies
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    # Install test-specific dependencies
    pip install pytest-cov pytest-xdist pytest-html
}

# Function to cleanup test environment
cleanup_test_env() {
    print_message "$YELLOW" "Cleaning up test environment..."
    
    # Remove temporary files
    find . -name "*.pyc" -delete
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Remove test databases
    rm -f test_mlflow.db mlflow.db 2>/dev/null || true
}

# Function to run unit tests
run_unit_tests() {
    print_message "$GREEN" "=== Running Unit Tests ==="
    
    if [[ "$COVERAGE" == "true" ]]; then
        run_tests_with_coverage "unit" "${TEST_DIR}/unit"
    else
        run_tests_simple "unit" "${TEST_DIR}/unit"
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_message "$GREEN" "=== Running Integration Tests ==="
    
    # Check if services are running for integration tests
    if [[ "$CHECK_SERVICES" == "true" ]]; then
        check_services_for_integration
    fi
    
    if [[ "$COVERAGE" == "true" ]]; then
        run_tests_with_coverage "integration" "${TEST_DIR}/integration"
    else
        run_tests_simple "integration" "${TEST_DIR}/integration"
    fi
}

# Function to run smoke tests
run_smoke_tests() {
    print_message "$GREEN" "=== Running Smoke Tests ==="
    
    # Smoke tests don't need coverage
    pytest \
        ${TEST_DIR}/smoke \
        --junit-xml=test_reports/smoke_results.xml \
        -v \
        --tb=short \
        -x  # Stop on first failure for smoke tests
}

# Function to check services for integration tests
check_services_for_integration() {
    print_message "$YELLOW" "Checking services for integration tests..."
    
    # Check if MLflow is running
    if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
        print_message "$YELLOW" "MLflow not running, starting for tests..."
        mlflow server --backend-store-uri sqlite:///test_mlflow.db --port 5000 --host 0.0.0.0 &
        MLFLOW_PID=$!
        sleep 5
    fi
    
    # Check if API is running
    if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
        print_message "$YELLOW" "API not running, integration tests may fail"
    fi
}

# Function to run performance tests
run_performance_tests() {
    print_message "$GREEN" "=== Running Performance Tests ==="
    
    pytest \
        ${TEST_DIR}/integration \
        -k "performance or load_test" \
        --junit-xml=test_reports/performance_results.xml \
        -v \
        --tb=short
}

# Function to run security tests
run_security_tests() {
    print_message "$GREEN" "=== Running Security Tests ==="
    
    # Run bandit for security scanning
    if command -v bandit &> /dev/null; then
        print_message "$BLUE" "Running Bandit security scan..."
        bandit -r pipelines/ mlflow/ -f json -o test_reports/security_bandit.json
    fi
    
    # Run safety for dependency vulnerability scanning
    if command -v safety &> /dev/null; then
        print_message "$BLUE" "Running Safety dependency scan..."
        safety check --json --output test_reports/security_safety.json
    fi
    
    # Run security-specific tests if they exist
    if [[ -d "${TEST_DIR}/security" ]]; then
        pytest \
            ${TEST_DIR}/security \
            --junit-xml=test_reports/security_results.xml \
            -v \
            --tb=short
    fi
}

# Function to generate test report
generate_test_report() {
    print_message "$GREEN" "=== Generating Test Report ==="
    
    # Generate HTML report
    pytest \
        --html=test_reports/report.html \
        --self-contained-html \
        --collect-only \
        ${TEST_DIR}/ \
        > /dev/null 2>&1 || true
    
    print_message "$GREEN" "Test reports generated in test_reports/"
    print_message "$GREEN" "Coverage reports generated in coverage_reports/"
}

# Function to run all tests
run_all_tests() {
    print_message "$GREEN" "=== Running All Tests ==="
    
    run_unit_tests
    run_integration_tests
    run_smoke_tests
    
    if [[ "$INCLUDE_SECURITY" == "true" ]]; then
        run_security_tests
    fi
    
    if [[ "$INCLUDE_PERFORMANCE" == "true" ]]; then
        run_performance_tests
    fi
}

# Function to run tests in parallel
run_parallel_tests() {
    print_message "$GREEN" "=== Running Tests in Parallel ==="
    
    pytest \
        ${TEST_DIR}/ \
        -n ${PARALLEL_JOBS} \
        --cov=pipelines \
        --cov=mlflow \
        --cov-report=term-missing \
        --cov-report=html:coverage_reports/parallel \
        --junit-xml=test_reports/parallel_results.xml \
        -v
}

# Main execution
main() {
    local test_type=${1:-"all"}
    
    print_message "$GREEN" "=== FlowOps Test Runner ==="
    
    # Setup
    setup_test_env
    
    case $test_type in
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "smoke")
            run_smoke_tests
            ;;
        "security")
            run_security_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        "parallel")
            run_parallel_tests
            ;;
        "all")
            run_all_tests
            ;;
        *)
            print_message "$RED" "Error: Unknown test type '$test_type'"
            print_message "$YELLOW" "Valid options: unit, integration, smoke, security, performance, parallel, all"
            exit 1
            ;;
    esac
    
    # Generate report
    generate_test_report
    
    # Cleanup
    cleanup_test_env
    
    print_message "$GREEN" "=== Tests Complete ==="
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --coverage)
            COVERAGE="true"
            shift
            ;;
        --check-services)
            CHECK_SERVICES="true"
            shift
            ;;
        --include-security)
            INCLUDE_SECURITY="true"
            shift
            ;;
        --include-performance)
            INCLUDE_PERFORMANCE="true"
            shift
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --coverage-threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [test_type] [options]"
            echo ""
            echo "Test types:"
            echo "  unit         Run unit tests"
            echo "  integration  Run integration tests" 
            echo "  smoke        Run smoke tests"
            echo "  security     Run security tests"
            echo "  performance  Run performance tests"
            echo "  parallel     Run tests in parallel"
            echo "  all          Run all tests (default)"
            echo ""
            echo "Options:"
            echo "  --coverage              Enable coverage reporting"
            echo "  --check-services        Check/start services for integration tests"
            echo "  --include-security      Include security tests in 'all' run"
            echo "  --include-performance   Include performance tests in 'all' run"
            echo "  --parallel-jobs N       Number of parallel jobs (default: 4)"
            echo "  --coverage-threshold N  Coverage threshold percentage (default: 80)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            if [[ -z "$TEST_TYPE" ]]; then
                TEST_TYPE="$1"
            fi
            shift
            ;;
    esac
done

# Run main function
main "${TEST_TYPE:-all}"