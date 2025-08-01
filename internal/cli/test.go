package cli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var testCmd = &cobra.Command{
	Use:   "test [type]",
	Short: "Run test suites",
	Long: `Run different types of tests for the FlowOps platform.

Available test types:
  unit         - Run unit tests
  integration  - Run integration tests  
  smoke        - Run smoke tests
  security     - Run security tests
  performance  - Run performance tests
  all          - Run all tests (default)`,
	Args: cobra.MaximumNArgs(1),
	RunE: runTest,
}

var (
	testCoverage       bool
	testParallel       int
	testCheckServices  bool
	testOutput         string
	testTimeout        string
)

func init() {
	rootCmd.AddCommand(testCmd)

	testCmd.Flags().BoolVar(&testCoverage, "coverage", false, "Enable coverage reporting")
	testCmd.Flags().IntVar(&testParallel, "parallel", 4, "Number of parallel test jobs")
	testCmd.Flags().BoolVar(&testCheckServices, "check-services", false, "Check/start services for integration tests")
	testCmd.Flags().StringVar(&testOutput, "output", ".", "Output directory for test reports")
	testCmd.Flags().StringVar(&testTimeout, "timeout", "10m", "Test timeout")
}

func runTest(cmd *cobra.Command, args []string) error {
	testType := "all"
	if len(args) > 0 {
		testType = args[0]
	}

	// Validate test type
	validTypes := []string{"unit", "integration", "smoke", "security", "performance", "all"}
	if !contains(validTypes, testType) {
		return fmt.Errorf("invalid test type '%s'. Valid types: %s", testType, strings.Join(validTypes, ", "))
	}

	if verbose {
		fmt.Printf("Running %s tests\n", testType)
		fmt.Printf("Coverage: %v\n", testCoverage)
		fmt.Printf("Parallel jobs: %d\n", testParallel)
	}

	// Setup test environment
	if err := setupTestEnv(); err != nil {
		return fmt.Errorf("failed to setup test environment: %w", err)
	}

	// Run tests based on type
	switch testType {
	case "unit":
		return runUnitTests()
	case "integration":
		return runIntegrationTests()
	case "smoke":
		return runSmokeTests()
	case "security":
		return runSecurityTests()
	case "performance":
		return runPerformanceTests()
	case "all":
		return runAllTests()
	default:
		return fmt.Errorf("unknown test type: %s", testType)
	}
}

func setupTestEnv() error {
	fmt.Println("🔧 Setting up test environment...")

	// Create output directories
	dirs := []string{
		filepath.Join(testOutput, "coverage_reports"),
		filepath.Join(testOutput, "test_reports"),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	// Check if pytest is available
	if err := checkPytest(); err != nil {
		return err
	}

	return nil
}

func runUnitTests() error {
	fmt.Println("🧪 Running unit tests...")

	args := []string{
		"tests/unit",
		"-v",
		"--tb=short",
		"--junit-xml=" + filepath.Join(testOutput, "test_reports/unit_results.xml"),
	}

	if testCoverage {
		args = append(args,
			"--cov=pipelines",
			"--cov=mlflow",
			"--cov-report=term-missing",
			"--cov-report=html:"+filepath.Join(testOutput, "coverage_reports/unit"),
			"--cov-report=xml:"+filepath.Join(testOutput, "coverage_reports/unit_coverage.xml"),
		)
	}

	return runPytest(args)
}

func runIntegrationTests() error {
	fmt.Println("🔗 Running integration tests...")

	if testCheckServices {
		if err := checkTestServices(); err != nil {
			fmt.Printf("⚠️  Warning: %v\n", err)
		}
	}

	args := []string{
		"tests/integration",
		"-v",
		"--tb=short",
		"--junit-xml=" + filepath.Join(testOutput, "test_reports/integration_results.xml"),
	}

	if testCoverage {
		args = append(args,
			"--cov=pipelines",
			"--cov-report=html:"+filepath.Join(testOutput, "coverage_reports/integration"),
		)
	}

	return runPytest(args)
}

func runSmokeTests() error {
	fmt.Println("💨 Running smoke tests...")

	args := []string{
		"tests/smoke",
		"-v",
		"--tb=short",
		"-x", // Stop on first failure
		"--junit-xml=" + filepath.Join(testOutput, "test_reports/smoke_results.xml"),
	}

	return runPytest(args)
}

func runSecurityTests() error {
	fmt.Println("🔒 Running security tests...")

	// Run bandit security scan
	if err := runBandit(); err != nil {
		fmt.Printf("⚠️  Bandit scan failed: %v\n", err)
	}

	// Run safety dependency scan
	if err := runSafety(); err != nil {
		fmt.Printf("⚠️  Safety scan failed: %v\n", err)
	}

	// Run security-specific tests if they exist
	securityDir := "tests/security"
	if _, err := os.Stat(securityDir); err == nil {
		args := []string{
			securityDir,
			"-v",
			"--tb=short",
			"--junit-xml=" + filepath.Join(testOutput, "test_reports/security_results.xml"),
		}
		return runPytest(args)
	}

	fmt.Println("✅ Security scans completed")
	return nil
}

func runPerformanceTests() error {
	fmt.Println("⚡ Running performance tests...")

	args := []string{
		"tests/integration",
		"-k", "performance or load_test",
		"-v",
		"--tb=short",
		"--junit-xml=" + filepath.Join(testOutput, "test_reports/performance_results.xml"),
	}

	return runPytest(args)
}

func runAllTests() error {
	fmt.Println("🎯 Running all tests...")

	tests := []func() error{
		runUnitTests,
		runIntegrationTests,
		runSmokeTests,
	}

	for _, testFunc := range tests {
		if err := testFunc(); err != nil {
			return err
		}
	}

	fmt.Println("✅ All tests completed successfully")
	return nil
}

func runPytest(args []string) error {
	cmd := exec.Command("pytest", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), "PYTHONPATH=.")

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("pytest failed: %w", err)
	}

	return nil
}

func runBandit() error {
	fmt.Println("🔍 Running Bandit security scan...")

	outputFile := filepath.Join(testOutput, "test_reports/security_bandit.json")
	cmd := exec.Command("bandit", "-r", "pipelines/", "mlflow/", "-f", "json", "-o", outputFile)
	
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("bandit scan failed: %w", err)
	}

	return nil
}

func runSafety() error {
	fmt.Println("🛡️  Running Safety dependency scan...")

	outputFile := filepath.Join(testOutput, "test_reports/security_safety.json")
	cmd := exec.Command("safety", "check", "--json", "--output", outputFile)
	
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("safety scan failed: %w", err)
	}

	return nil
}

func checkTestServices() error {
	// Check if MLflow is running
	if err := checkService("http://localhost:5000/health", "MLflow"); err != nil {
		return err
	}

	// Check if API is running
	if err := checkService("http://localhost:8080/health", "API"); err != nil {
		return err
	}

	return nil
}

func checkService(url, name string) error {
	cmd := exec.Command("curl", "-s", "-f", url)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("%s service not available at %s", name, url)
	}
	fmt.Printf("✅ %s service is running\n", name)
	return nil
}

func checkPytest() error {
	cmd := exec.Command("pytest", "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("pytest not found. Please install: pip install pytest pytest-cov")
	}
	return nil
}