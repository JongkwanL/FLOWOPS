package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/flowops/flowops/internal/monitor"
)

func main() {
	var (
		configFile    = flag.String("config", "monitor-config.yaml", "Configuration file path")
		metricsAddr   = flag.String("metrics-addr", ":9090", "Metrics server address")
		checkInterval = flag.Duration("check-interval", 30*time.Second, "Health check interval")
		timeout       = flag.Duration("timeout", 10*time.Second, "Health check timeout")
		verbose       = flag.Bool("verbose", false, "Enable verbose logging")
	)
	flag.Parse()

	if *verbose {
		log.Println("Starting FlowOps monitoring agent")
		log.Printf("Config: %s", *configFile)
		log.Printf("Metrics address: %s", *metricsAddr)
		log.Printf("Check interval: %v", *checkInterval)
		log.Printf("Timeout: %v", *timeout)
	}

	// Create monitoring agent
	agent := monitor.NewAgent(*checkInterval, *timeout)

	// Load configuration if file exists
	if err := loadConfiguration(agent, *configFile, *verbose); err != nil {
		log.Printf("Warning: Could not load config file %s: %v", *configFile, err)
		log.Println("Using default service configurations...")
		setupDefaultServices(agent)
	}

	// Start monitoring
	if err := agent.Start(); err != nil {
		log.Fatalf("Failed to start monitoring: %v", err)
	}

	// Start metrics server
	go func() {
		log.Printf("Starting metrics server on %s", *metricsAddr)
		if err := agent.StartMetricsServer(*metricsAddr); err != nil {
			log.Fatalf("Metrics server failed: %v", err)
		}
	}()

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	log.Println("FlowOps monitoring agent started successfully")
	log.Println("Press Ctrl+C to stop...")

	<-sigChan
	log.Println("Shutting down monitoring agent...")

	agent.Stop()
	log.Println("Monitoring agent stopped")
}

func setupDefaultServices(agent *monitor.Agent) {
	// MLflow tracking server
	agent.AddService(&monitor.ServiceConfig{
		Name:         "mlflow",
		URL:          "http://localhost:5000",
		HealthPath:   "/health",
		CheckType:    "http",
		Timeout:      10 * time.Second,
		Interval:     30 * time.Second,
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
	})

	// DVC status check (command-based)
	agent.AddService(&monitor.ServiceConfig{
		Name:      "dvc",
		URL:       "file://.",
		CheckType: "command",
		Command:   "dvc status --json",
		Timeout:   15 * time.Second,
		Interval:  60 * time.Second,
	})

	// Kubernetes deployments check
	agent.AddService(&monitor.ServiceConfig{
		Name:      "kubernetes",
		URL:       "k8s://flowops-production",
		CheckType: "command",
		Command:   "kubectl get deployments -n flowops-production -o json",
		Timeout:   20 * time.Second,
		Interval:  60 * time.Second,
	})
}

func loadConfiguration(agent *monitor.Agent, configFile string, verbose bool) error {
	if _, err := os.Stat(configFile); os.IsNotExist(err) {
		return fmt.Errorf("config file does not exist")
	}

	// TODO: Implement YAML configuration loading
	// For now, just return an error to use defaults
	return fmt.Errorf("configuration loading not yet implemented")
}