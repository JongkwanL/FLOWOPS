package cli

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var deployCmd = &cobra.Command{
	Use:   "deploy [environment]",
	Short: "Deploy to target environment",
	Long: `Deploy the FlowOps platform to different environments using Helm and ArgoCD.

Available environments:
  staging     - Deploy to staging environment
  production  - Deploy to production environment  
  local       - Deploy to local development environment`,
	Args: cobra.MaximumNArgs(1),
	RunE: runDeploy,
}

var (
	deployNamespace   string
	deployDryRun      bool
	deployWait        bool
	deployTimeout     string
	deployValues      []string
	deploySet         []string
	deployArgoSync    bool
)

func init() {
	rootCmd.AddCommand(deployCmd)

	deployCmd.Flags().StringVar(&deployNamespace, "namespace", "", "Kubernetes namespace (default: flowops-{environment})")
	deployCmd.Flags().BoolVar(&deployDryRun, "dry-run", false, "Perform a dry run")
	deployCmd.Flags().BoolVar(&deployWait, "wait", true, "Wait for deployment to complete")
	deployCmd.Flags().StringVar(&deployTimeout, "timeout", "10m", "Deployment timeout")
	deployCmd.Flags().StringSliceVar(&deployValues, "values", nil, "Additional values files")
	deployCmd.Flags().StringSliceVar(&deploySet, "set", nil, "Set values on command line")
	deployCmd.Flags().BoolVar(&deployArgoSync, "argo-sync", false, "Trigger ArgoCD sync instead of direct Helm")
}

func runDeploy(cmd *cobra.Command, args []string) error {
	environment := "staging"
	if len(args) > 0 {
		environment = args[0]
	}

	// Validate environment
	validEnvs := []string{"staging", "production", "local"}
	if !contains(validEnvs, environment) {
		return fmt.Errorf("invalid environment '%s'. Valid environments: %s", environment, strings.Join(validEnvs, ", "))
	}

	// Set default namespace if not specified
	if deployNamespace == "" {
		deployNamespace = fmt.Sprintf("flowops-%s", environment)
	}

	if verbose {
		fmt.Printf("Deploying to environment: %s\n", environment)
		fmt.Printf("Namespace: %s\n", deployNamespace)
		fmt.Printf("Dry run: %v\n", deployDryRun)
	}

	// Check prerequisites
	if err := checkPrerequisites(); err != nil {
		return err
	}

	// Deploy using ArgoCD or Helm
	if deployArgoSync {
		return deployWithArgo(environment)
	} else {
		return deployWithHelm(environment)
	}
}

func deployWithHelm(environment string) error {
	fmt.Printf("🚀 Deploying with Helm to %s environment...\n", environment)

	// Check if helm chart exists
	chartPath := "infrastructure/helm/flowops"
	if !dirExists(chartPath) {
		return fmt.Errorf("helm chart not found at %s", chartPath)
	}

	// Build helm command
	args := []string{
		"upgrade", "--install",
		fmt.Sprintf("flowops-%s", environment),
		chartPath,
		"--namespace", deployNamespace,
		"--create-namespace",
	}

	// Add values files
	defaultValuesFile := filepath.Join(chartPath, "values.yaml")
	if fileExists(defaultValuesFile) {
		args = append(args, "--values", defaultValuesFile)
	}

	envValuesFile := filepath.Join(chartPath, fmt.Sprintf("values-%s.yaml", environment))
	if fileExists(envValuesFile) {
		args = append(args, "--values", envValuesFile)
	}

	// Add additional values files
	for _, valuesFile := range deployValues {
		args = append(args, "--values", valuesFile)
	}

	// Add set values
	for _, setValue := range deploySet {
		args = append(args, "--set", setValue)
	}

	// Add environment-specific values
	args = append(args, "--set", fmt.Sprintf("environment=%s", environment))

	if deployWait {
		args = append(args, "--wait", "--timeout", deployTimeout)
	}

	if deployDryRun {
		args = append(args, "--dry-run")
	}

	// Execute helm command
	cmd := exec.Command("helm", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm deployment failed: %w", err)
	}

	if !deployDryRun {
		fmt.Printf("✅ Successfully deployed to %s environment\n", environment)
		
		// Show deployment status
		if err := showDeploymentStatus(environment); err != nil {
			fmt.Printf("⚠️  Could not get deployment status: %v\n", err)
		}
	}

	return nil
}

func deployWithArgo(environment string) error {
	fmt.Printf("🎯 Triggering ArgoCD sync for %s environment...\n", environment)

	appName := fmt.Sprintf("flowops-%s", environment)
	if environment == "production" {
		appName = "flowops-mlops"
	}

	// Check if argocd CLI is available
	if err := checkArgoCLI(); err != nil {
		return fmt.Errorf("ArgoCD CLI not available: %w", err)
	}

	// Sync application
	args := []string{
		"app", "sync", appName,
		"--timeout", deployTimeout,
	}

	if deployDryRun {
		args = append(args, "--dry-run")
	}

	cmd := exec.Command("argocd", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("argocd sync failed: %w", err)
	}

	if !deployDryRun {
		fmt.Printf("✅ Successfully triggered ArgoCD sync for %s\n", environment)
		
		// Wait for sync to complete if requested
		if deployWait {
			return waitForArgoSync(appName)
		}
	}

	return nil
}

func waitForArgoSync(appName string) error {
	fmt.Printf("⏳ Waiting for ArgoCD sync to complete...\n")

	cmd := exec.Command("argocd", "app", "wait", appName, "--timeout", deployTimeout)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to wait for sync: %w", err)
	}

	return nil
}

func showDeploymentStatus(environment string) error {
	fmt.Println("\n📊 Deployment Status:")

	// Get pods status
	cmd := exec.Command("kubectl", "get", "pods", "-n", deployNamespace, "-o", "wide")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to get pods status: %w", err)
	}

	// Get services status
	fmt.Println("\n🌐 Services:")
	cmd = exec.Command("kubectl", "get", "services", "-n", deployNamespace)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to get services status: %w", err)
	}

	return nil
}

func checkPrerequisites() error {
	// Check kubectl
	if err := checkKubectl(); err != nil {
		return err
	}

	// Check helm
	if err := checkHelm(); err != nil {
		return err
	}

	// Check cluster connectivity
	if err := checkClusterConnectivity(); err != nil {
		return err
	}

	return nil
}

func checkKubectl() error {
	cmd := exec.Command("kubectl", "version", "--client")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("kubectl not found. Please install kubectl")
	}
	return nil
}

func checkHelm() error {
	cmd := exec.Command("helm", "version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm not found. Please install helm")
	}
	return nil
}

func checkArgoCLI() error {
	cmd := exec.Command("argocd", "version", "--client")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("argocd CLI not found. Please install argocd CLI")
	}
	return nil
}

func checkClusterConnectivity() error {
	cmd := exec.Command("kubectl", "cluster-info")
	cmd.Stdout = nil // Suppress output
	cmd.Stderr = nil
	
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("cannot connect to Kubernetes cluster. Please check your kubeconfig")
	}
	return nil
}

func dirExists(dirname string) bool {
	info, err := os.Stat(dirname)
	if os.IsNotExist(err) {
		return false
	}
	return info.IsDir()
}