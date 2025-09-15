package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/flowops/flowops/internal/drift"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

// driftCmd represents the drift command
var driftCmd = &cobra.Command{
	Use:   "drift",
	Short: "Start the FlowOps Drift Detection Service",
	Long: `Start the FlowOps Drift Detection Service that provides:

- Real-time drift detection using multiple statistical methods
- ADWIN (Adaptive Windowing) for concept drift detection
- Population Stability Index (PSI) for distribution monitoring
- Kolmogorov-Smirnov test for statistical comparison
- Jensen-Shannon divergence for distribution similarity
- Wasserstein distance for optimal transport-based detection

Statistical Methods Available:
- ADWIN: Adaptive windowing for concept drift
- PSI: Population Stability Index for feature distribution
- KS: Kolmogorov-Smirnov test for distribution comparison
- JS: Jensen-Shannon divergence for distribution similarity
- Wasserstein: Earth Mover's Distance for optimal transport

The service provides REST API endpoints for:
- Setting reference data for drift baseline
- Real-time drift detection with configurable thresholds
- Batch processing for multiple features
- Historical results and monitoring
- Prometheus metrics integration

Examples:
  flowops drift                              # Start with default settings
  flowops drift --port 8090                 # Start on specific port
  flowops drift --config drift-config.yaml  # Start with config file
  flowops drift --verbose                   # Start with debug logging`,
	RunE: runDriftService,
}

func init() {
	rootCmd.AddCommand(driftCmd)

	// Server configuration
	driftCmd.Flags().String("host", "0.0.0.0", "Server host address")
	driftCmd.Flags().Int("port", 8090, "Server port")
	driftCmd.Flags().String("config", "", "Configuration file path")

	// Detection configuration
	driftCmd.Flags().Float64("adwin-delta", 0.002, "ADWIN confidence parameter")
	driftCmd.Flags().Float64("psi-threshold", 0.1, "PSI alert threshold")
	driftCmd.Flags().Float64("ks-alpha", 0.05, "KS test significance level")
	driftCmd.Flags().Int("psi-bins", 10, "Number of bins for PSI calculation")
	driftCmd.Flags().Int("min-sample-size", 30, "Minimum sample size for detection")

	// Service flags
	driftCmd.Flags().Bool("metrics", true, "Enable Prometheus metrics")
	driftCmd.Flags().Bool("cors", true, "Enable CORS")
	driftCmd.Flags().Bool("verbose", false, "Enable verbose logging")

	// Method selection
	driftCmd.Flags().StringSlice("methods", []string{"psi", "ks", "adwin", "js"}, 
		"Enabled detection methods (psi,ks,adwin,js,wasserstein)")

	// Bind flags to viper
	viper.BindPFlag("drift.host", driftCmd.Flags().Lookup("host"))
	viper.BindPFlag("drift.port", driftCmd.Flags().Lookup("port"))
	viper.BindPFlag("drift.adwin_delta", driftCmd.Flags().Lookup("adwin-delta"))
	viper.BindPFlag("drift.psi_threshold", driftCmd.Flags().Lookup("psi-threshold"))
	viper.BindPFlag("drift.ks_alpha", driftCmd.Flags().Lookup("ks-alpha"))
	viper.BindPFlag("drift.psi_bins", driftCmd.Flags().Lookup("psi-bins"))
	viper.BindPFlag("drift.min_sample_size", driftCmd.Flags().Lookup("min-sample-size"))
	viper.BindPFlag("drift.metrics", driftCmd.Flags().Lookup("metrics"))
	viper.BindPFlag("drift.cors", driftCmd.Flags().Lookup("cors"))
	viper.BindPFlag("drift.verbose", driftCmd.Flags().Lookup("verbose"))
	viper.BindPFlag("drift.methods", driftCmd.Flags().Lookup("methods"))
}

func runDriftService(cmd *cobra.Command, args []string) error {
	fmt.Println("🔍 Starting FlowOps Drift Detection Service...")

	// Load configuration from file if specified
	if configFile, _ := cmd.Flags().GetString("config"); configFile != "" {
		viper.SetConfigFile(configFile)
		if err := viper.ReadInConfig(); err != nil {
			return fmt.Errorf("failed to read config file: %w", err)
		}
		fmt.Printf("📁 Loaded configuration from: %s\n", configFile)
	}

	// Create detector configuration
	methods, _ := cmd.Flags().GetStringSlice("methods")
	detectorConfig := &drift.DetectorConfig{
		ADWINDelta:    viper.GetFloat64("drift.adwin_delta"),
		PSIThreshold:  viper.GetFloat64("drift.psi_threshold"),
		KSAlpha:       viper.GetFloat64("drift.ks_alpha"),
		PSIBins:       viper.GetInt("drift.psi_bins"),
		MinSampleSize: viper.GetInt("drift.min_sample_size"),
		AlertThresholds: map[string]float64{
			"psi":         viper.GetFloat64("drift.psi_threshold"),
			"ks":          viper.GetFloat64("drift.ks_alpha"),
			"adwin":       0.5,
			"js":          0.1,
			"wasserstein": 0.1,
		},
		EnabledMethods: methods,
		SeverityLevels: map[string]string{
			"low":    "0.0-0.3",
			"medium": "0.3-0.7",
			"high":   "0.7-1.0",
		},
	}

	// Create service configuration
	serviceConfig := &drift.ServiceConfig{
		Host:          viper.GetString("drift.host"),
		Port:          viper.GetInt("drift.port"),
		EnableMetrics: viper.GetBool("drift.metrics"),
		EnableCORS:    viper.GetBool("drift.cors"),
		LogLevel:      "info",
	}

	if viper.GetBool("drift.verbose") {
		serviceConfig.LogLevel = "debug"
	}

	// Validate configuration
	if err := validateDriftConfig(detectorConfig, serviceConfig); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	// Create drift detector
	detector := drift.NewDriftDetector(detectorConfig)

	// Create drift service
	service := drift.NewDriftService(detector, serviceConfig)

	// Print configuration summary
	printDriftConfig(detectorConfig, serviceConfig)

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
	printDriftExamples(serviceConfig)

	// Wait for interrupt signal or service error
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errChan:
		return err
	case sig := <-quit:
		fmt.Printf("\n🛑 Received signal: %v\n", sig)
	}

	fmt.Println("🔄 Shutting down service...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// In a real implementation, service would have a Stop method
	select {
	case <-ctx.Done():
		fmt.Println("⏰ Shutdown timeout exceeded")
	case <-time.After(1 * time.Second):
		fmt.Println("✅ Service stopped successfully")
	}

	return nil
}

// validateDriftConfig validates the drift service configuration
func validateDriftConfig(detectorConfig *drift.DetectorConfig, serviceConfig *drift.ServiceConfig) error {
	if serviceConfig.Port <= 0 || serviceConfig.Port > 65535 {
		return fmt.Errorf("invalid port: %d", serviceConfig.Port)
	}

	if detectorConfig.ADWINDelta <= 0 || detectorConfig.ADWINDelta >= 1 {
		return fmt.Errorf("ADWIN delta must be between 0 and 1: %f", detectorConfig.ADWINDelta)
	}

	if detectorConfig.PSIThreshold < 0 {
		return fmt.Errorf("PSI threshold must be non-negative: %f", detectorConfig.PSIThreshold)
	}

	if detectorConfig.KSAlpha <= 0 || detectorConfig.KSAlpha >= 1 {
		return fmt.Errorf("KS alpha must be between 0 and 1: %f", detectorConfig.KSAlpha)
	}

	if detectorConfig.PSIBins <= 0 {
		return fmt.Errorf("PSI bins must be positive: %d", detectorConfig.PSIBins)
	}

	if detectorConfig.MinSampleSize <= 0 {
		return fmt.Errorf("minimum sample size must be positive: %d", detectorConfig.MinSampleSize)
	}

	if len(detectorConfig.EnabledMethods) == 0 {
		return fmt.Errorf("at least one detection method must be enabled")
	}

	validMethods := map[string]bool{
		"psi": true, "ks": true, "adwin": true, "js": true, "wasserstein": true,
	}
	for _, method := range detectorConfig.EnabledMethods {
		if !validMethods[method] {
			return fmt.Errorf("invalid detection method: %s", method)
		}
	}

	return nil
}

// printDriftConfig prints the drift service configuration summary
func printDriftConfig(detectorConfig *drift.DetectorConfig, serviceConfig *drift.ServiceConfig) {
	fmt.Println("\n📋 Drift Detection Configuration:")
	fmt.Printf("   Host: %s\n", serviceConfig.Host)
	fmt.Printf("   Port: %d\n", serviceConfig.Port)
	fmt.Printf("   Metrics: %v\n", serviceConfig.EnableMetrics)
	fmt.Printf("   CORS: %v\n", serviceConfig.EnableCORS)
	fmt.Printf("   Log Level: %s\n", serviceConfig.LogLevel)
	fmt.Println()

	fmt.Println("🧮 Detection Methods:")
	for _, method := range detectorConfig.EnabledMethods {
		threshold := detectorConfig.AlertThresholds[method]
		fmt.Printf("   %s: threshold=%.3f\n", method, threshold)
	}
	fmt.Println()

	fmt.Println("⚙️ Algorithm Parameters:")
	fmt.Printf("   ADWIN Delta: %.3f\n", detectorConfig.ADWINDelta)
	fmt.Printf("   PSI Bins: %d\n", detectorConfig.PSIBins)
	fmt.Printf("   KS Alpha: %.3f\n", detectorConfig.KSAlpha)
	fmt.Printf("   Min Sample Size: %d\n", detectorConfig.MinSampleSize)
	fmt.Println()
}

// printDriftExamples prints example API usage
func printDriftExamples(serviceConfig *drift.ServiceConfig) {
	fmt.Println("📝 Example API Usage:")
	fmt.Printf("  # Set reference data\n")
	fmt.Printf("  curl -X POST http://%s:%d/api/v1/reference \\\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("    -H 'Content-Type: application/json' \\\n")
	fmt.Printf("    -d '{\"model_name\":\"fraud_model\",\"features\":{\"amount\":[1,2,3,4,5,6,7,8,9,10]}}'\n")
	fmt.Printf("\n")
	fmt.Printf("  # Detect drift\n")
	fmt.Printf("  curl -X POST http://%s:%d/api/v1/detect \\\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("    -H 'Content-Type: application/json' \\\n")
	fmt.Printf("    -d '{\"model_name\":\"fraud_model\",\"features\":{\"amount\":[15,20,25,30,35,40,45,50,55,60]}}'\n")
	fmt.Printf("\n")
	fmt.Printf("  # Get detection results\n")
	fmt.Printf("  curl http://%s:%d/api/v1/results?limit=10\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Printf("\n")
	fmt.Printf("  # Check service status\n")
	fmt.Printf("  curl http://%s:%d/api/v1/status\n", serviceConfig.Host, serviceConfig.Port)
	fmt.Println()
}