package cli

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"github.com/flowops/flowops/internal/monitor"
)

var agentCmd = &cobra.Command{
	Use:   "agent",
	Short: "Run monitoring agent",
	Long: `Run the FlowOps monitoring agent as a standalone service.
	
The agent continuously monitors FlowOps services and exports
Prometheus metrics. It can run as a daemon or in foreground mode.`,
	RunE: runAgent,
}

var (
	agentConfigFile    string
	agentMetricsAddr   string
	agentCheckInterval time.Duration
	agentTimeout       time.Duration
	agentDaemon        bool
)

func init() {
	rootCmd.AddCommand(agentCmd)

	agentCmd.Flags().StringVar(&agentConfigFile, "config-file", "configs/monitor-config.yaml", "Configuration file path")
	agentCmd.Flags().StringVar(&agentMetricsAddr, "metrics-addr", ":9090", "Metrics server address")
	agentCmd.Flags().DurationVar(&agentCheckInterval, "check-interval", 30*time.Second, "Health check interval")
	agentCmd.Flags().DurationVar(&agentTimeout, "timeout", 10*time.Second, "Health check timeout")
	agentCmd.Flags().BoolVar(&agentDaemon, "daemon", false, "Run as daemon (background)")
}

func runAgent(cmd *cobra.Command, args []string) error {
	if verbose {
		fmt.Println("🤖 Starting FlowOps monitoring agent")
		fmt.Printf("   Config: %s\n", agentConfigFile)
		fmt.Printf("   Metrics: %s\n", agentMetricsAddr)
		fmt.Printf("   Interval: %v\n", agentCheckInterval)
		fmt.Printf("   Timeout: %v\n", agentTimeout)
	}

	// Create monitoring agent
	agent := monitor.NewAgent(agentCheckInterval, agentTimeout)

	// Setup default services
	if err := setupDefaultMonitoringServices(agent); err != nil {
		return fmt.Errorf("failed to setup monitoring services: %w", err)
	}

	// Start monitoring
	if err := agent.Start(); err != nil {
		return fmt.Errorf("failed to start monitoring: %w", err)
	}

	// Start metrics server
	go func() {
		if verbose {
			fmt.Printf("📊 Metrics server starting on %s\n", agentMetricsAddr)
		}
		if err := agent.StartMetricsServer(agentMetricsAddr); err != nil {
			fmt.Printf("❌ Metrics server failed: %v\n", err)
			os.Exit(1)
		}
	}()

	if !agentDaemon {
		fmt.Println("✅ FlowOps monitoring agent started successfully")
		fmt.Printf("📊 Metrics available at: http://localhost%s/metrics\n", agentMetricsAddr)
		fmt.Printf("🔍 Status endpoint: http://localhost%s/status\n", agentMetricsAddr)
		fmt.Println("🛑 Press Ctrl+C to stop...")
	}

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	<-sigChan

	if verbose {
		fmt.Println("\n🛑 Shutting down monitoring agent...")
	}

	agent.Stop()

	if verbose {
		fmt.Println("✅ Monitoring agent stopped")
	}

	return nil
}

func setupDefaultMonitoringServices(agent *monitor.Agent) error {
	// MLflow tracking server
	agent.AddService(&monitor.ServiceConfig{
		Name:         "mlflow",
		URL:          "http://localhost:5000",
		HealthPath:   "/health",
		CheckType:    "http",
		Timeout:      10 * time.Second,
		Interval:     30 * time.Second,
		ExpectedCode: 200,
		Headers: map[string]string{
			"User-Agent": "FlowOps-Monitor/1.0",
		},
	})

	// Try Kubernetes MLflow if local fails
	agent.AddService(&monitor.ServiceConfig{
		Name:         "mlflow-k8s",
		URL:          "http://mlflow.flowops-production.svc.cluster.local:5000",
		HealthPath:   "/health",
		CheckType:    "http",
		Timeout:      15 * time.Second,
		Interval:     60 * time.Second,
		ExpectedCode: 200,
	})

	// Model serving API
	agent.AddService(&monitor.ServiceConfig{
		Name:         "model-api",
		URL:          "http://localhost:8080",
		HealthPath:   "/health",
		CheckType:    "http",
		Timeout:      10 * time.Second,
		Interval:     30 * time.Second,
		ExpectedCode: 200,
		Headers: map[string]string{
			"User-Agent": "FlowOps-Monitor/1.0",
		},
	})

	// Model serving API in Kubernetes
	agent.AddService(&monitor.ServiceConfig{
		Name:         "model-api-k8s",
		URL:          "http://model-serving.flowops-production.svc.cluster.local:8080",
		HealthPath:   "/health",
		CheckType:    "http",
		Timeout:      15 * time.Second,
		Interval:     45 * time.Second,
		ExpectedCode: 200,
	})

	// DVC status check
	agent.AddService(&monitor.ServiceConfig{
		Name:      "dvc-status",
		URL:       "file://.",
		CheckType: "command",
		Command:   "dvc status --json",
		Timeout:   15 * time.Second,
		Interval:  120 * time.Second,
	})

	// DVC remote connectivity
	agent.AddService(&monitor.ServiceConfig{
		Name:      "dvc-remote",
		URL:       "file://.",
		CheckType: "command",
		Command:   "dvc remote list",
		Timeout:   10 * time.Second,
		Interval:  300 * time.Second,
	})

	// Kubernetes deployments
	agent.AddService(&monitor.ServiceConfig{
		Name:      "k8s-deployments",
		URL:       "k8s://flowops-production",
		CheckType: "command",
		Command:   "kubectl get deployments -n flowops-production -o json",
		Timeout:   20 * time.Second,
		Interval:  60 * time.Second,
	})

	// Kubernetes services
	agent.AddService(&monitor.ServiceConfig{
		Name:      "k8s-services",
		URL:       "k8s://flowops-production",
		CheckType: "command",
		Command:   "kubectl get services -n flowops-production -o json",
		Timeout:   20 * time.Second,
		Interval:  120 * time.Second,
	})

	return nil
}