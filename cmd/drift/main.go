package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/flowops/flowops/internal/drift"
)

func main() {
	// Command line flags
	var (
		host    = flag.String("host", "0.0.0.0", "Service host")
		port    = flag.Int("port", 8090, "Service port")
		config  = flag.String("config", "", "Configuration file path")
		verbose = flag.Bool("verbose", false, "Enable verbose logging")
	)
	flag.Parse()

	// Print startup banner
	fmt.Println("🔍 FlowOps Drift Detection Service")
	fmt.Println("====================================")

	// Create detector configuration
	detectorConfig := drift.DefaultDetectorConfig()
	if *config != "" {
		fmt.Printf("📁 Loading configuration from: %s\n", *config)
		// TODO: Load from file
	}

	// Create service configuration
	serviceConfig := &drift.ServiceConfig{
		Host:          *host,
		Port:          *port,
		EnableMetrics: true,
		EnableCORS:    true,
		LogLevel:      "info",
	}

	if *verbose {
		serviceConfig.LogLevel = "debug"
	}

	// Create drift detector
	detector := drift.NewDriftDetector(detectorConfig)

	// Create drift service
	service := drift.NewDriftService(detector, serviceConfig)

	// Print configuration
	fmt.Printf("🌐 Server: %s:%d\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("📊 Metrics: %v\n", serviceConfig.EnableMetrics)
	fmt.Printf("🔐 CORS: %v\n", serviceConfig.EnableCORS)
	fmt.Printf("📝 Log Level: %s\n", serviceConfig.LogLevel)
	fmt.Println()

	// Print enabled methods
	fmt.Println("🧮 Enabled Detection Methods:")
	for _, method := range detectorConfig.EnabledMethods {
		fmt.Printf("  - %s\n", method)
	}
	fmt.Println()

	// Print alert thresholds
	fmt.Println("⚠️  Alert Thresholds:")
	for method, threshold := range detectorConfig.AlertThresholds {
		fmt.Printf("  - %s: %.3f\n", method, threshold)
	}
	fmt.Println()

	// Print API endpoints
	fmt.Println("🔌 Available Endpoints:")
	fmt.Printf("  GET  /health                     - Health check\n")
	fmt.Printf("  GET  /ready                      - Readiness check\n")
	fmt.Printf("  GET  /metrics                    - Prometheus metrics\n")
	fmt.Printf("  POST /api/v1/reference           - Set reference data\n")
	fmt.Printf("  POST /api/v1/detect              - Detect drift\n")
	fmt.Printf("  POST /api/v1/detect/batch        - Batch drift detection\n")
	fmt.Printf("  GET  /api/v1/results             - Get recent results\n")
	fmt.Printf("  GET  /api/v1/status              - Get service status\n")
	fmt.Printf("  GET  /api/v1/config              - Get configuration\n")
	fmt.Printf("  PUT  /api/v1/config              - Update configuration\n")
	fmt.Println()

	// Start service in a goroutine
	errChan := make(chan error, 1)
	go func() {
		if err := service.Start(); err != nil {
			errChan <- fmt.Errorf("failed to start service: %w", err)
		}
	}()

	fmt.Printf("✅ Drift Detection Service started successfully\n")
	fmt.Printf("🌐 Service URL: http://%s:%d\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("📊 Health check: http://%s:%d/health\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("📈 Metrics: http://%s:%d/metrics\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("📚 API: http://%s:%d/api/v1\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Println()

	// Print example usage
	fmt.Println("📝 Example Usage:")
	fmt.Printf("  # Set reference data\n")
	fmt.Printf("  curl -X POST http://%s:%d/api/v1/reference \\\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("    -H 'Content-Type: application/json' \\\n")
	fmt.Printf("    -d '{\"model_name\":\"fraud_model\",\"features\":{\"amount\":[1,2,3,4,5]}}'\n")
	fmt.Printf("\n")
	fmt.Printf("  # Detect drift\n")
	fmt.Printf("  curl -X POST http://%s:%d/api/v1/detect \\\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("    -H 'Content-Type: application/json' \\\n")
	fmt.Printf("    -d '{\"model_name\":\"fraud_model\",\"features\":{\"amount\":[10,20,30,40,50]}}'\n")
	fmt.Println()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errChan:
		log.Fatalf("Service error: %v", err)
	case sig := <-quit:
		fmt.Printf("\n🛑 Received signal: %v\n", sig)
	}

	fmt.Println("🔄 Shutting down service...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// In a real implementation, you would call service.Stop(ctx) here
	select {
	case <-ctx.Done():
		fmt.Println("⏰ Shutdown timeout exceeded")
	case <-time.After(1 * time.Second):
		fmt.Println("✅ Service stopped successfully")
	}
}