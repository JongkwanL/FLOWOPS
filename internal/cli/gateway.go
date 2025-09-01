package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/flowops/flowops/internal/gateway"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

// gatewayCmd represents the gateway command
var gatewayCmd = &cobra.Command{
	Use:   "gateway",
	Short: "Start the FlowOps API Gateway",
	Long: `Start the FlowOps API Gateway server that provides:

- Unified API endpoint for all MLOps services
- Load balancing and service discovery
- Authentication and authorization
- Rate limiting and security
- Request/response transformation
- Metrics and monitoring integration

The gateway acts as a single entry point for:
- MLflow experiment tracking
- Model serving endpoints
- Pipeline management
- Monitoring and metrics

Examples:
  flowops gateway                          # Start with default settings
  flowops gateway --port 8080              # Start on specific port
  flowops gateway --auth --jwt-secret xyz  # Start with authentication
  flowops gateway --config gateway.yaml   # Start with config file`,
	RunE: runGateway,
}

func init() {
	rootCmd.AddCommand(gatewayCmd)

	// Server configuration
	gatewayCmd.Flags().String("host", "0.0.0.0", "Server host address")
	gatewayCmd.Flags().Int("port", 8080, "Server port")
	gatewayCmd.Flags().String("config", "", "Configuration file path")

	// Security flags
	gatewayCmd.Flags().Bool("auth", false, "Enable authentication")
	gatewayCmd.Flags().String("jwt-secret", "", "JWT secret key (required if auth enabled)")
	gatewayCmd.Flags().Bool("cors", true, "Enable CORS")

	// Rate limiting flags
	gatewayCmd.Flags().Bool("rate-limit", true, "Enable rate limiting")
	gatewayCmd.Flags().Int("rate-limit-rps", 100, "Rate limit requests per second")

	// TLS flags
	gatewayCmd.Flags().String("tls-cert", "", "TLS certificate file")
	gatewayCmd.Flags().String("tls-key", "", "TLS private key file")

	// Feature flags
	gatewayCmd.Flags().Bool("metrics", true, "Enable metrics collection")
	gatewayCmd.Flags().Bool("debug", false, "Enable debug logging")

	// Bind flags to viper
	viper.BindPFlag("gateway.host", gatewayCmd.Flags().Lookup("host"))
	viper.BindPFlag("gateway.port", gatewayCmd.Flags().Lookup("port"))
	viper.BindPFlag("gateway.auth", gatewayCmd.Flags().Lookup("auth"))
	viper.BindPFlag("gateway.jwt_secret", gatewayCmd.Flags().Lookup("jwt-secret"))
	viper.BindPFlag("gateway.cors", gatewayCmd.Flags().Lookup("cors"))
	viper.BindPFlag("gateway.rate_limit", gatewayCmd.Flags().Lookup("rate-limit"))
	viper.BindPFlag("gateway.rate_limit_rps", gatewayCmd.Flags().Lookup("rate-limit-rps"))
	viper.BindPFlag("gateway.tls_cert", gatewayCmd.Flags().Lookup("tls-cert"))
	viper.BindPFlag("gateway.tls_key", gatewayCmd.Flags().Lookup("tls-key"))
	viper.BindPFlag("gateway.metrics", gatewayCmd.Flags().Lookup("metrics"))
	viper.BindPFlag("gateway.debug", gatewayCmd.Flags().Lookup("debug"))
}

func runGateway(cmd *cobra.Command, args []string) error {
	fmt.Println("🚀 Starting FlowOps API Gateway...")

	// Load configuration from file if specified
	if configFile, _ := cmd.Flags().GetString("config"); configFile != "" {
		viper.SetConfigFile(configFile)
		if err := viper.ReadInConfig(); err != nil {
			return fmt.Errorf("failed to read config file: %w", err)
		}
		fmt.Printf("📁 Loaded configuration from: %s\n", configFile)
	}

	// Create gateway configuration
	config := &gateway.Config{
		Host:            viper.GetString("gateway.host"),
		Port:            viper.GetInt("gateway.port"),
		ReadTimeout:     15 * time.Second,
		WriteTimeout:    15 * time.Second,
		IdleTimeout:     60 * time.Second,
		EnableCORS:      viper.GetBool("gateway.cors"),
		EnableAuth:      viper.GetBool("gateway.auth"),
		EnableMetrics:   viper.GetBool("gateway.metrics"),
		EnableRateLimit: viper.GetBool("gateway.rate_limit"),
		RateLimitRPS:    viper.GetInt("gateway.rate_limit_rps"),
		JWTSecret:       viper.GetString("gateway.jwt_secret"),
		CertFile:        viper.GetString("gateway.tls_cert"),
		KeyFile:         viper.GetString("gateway.tls_key"),
	}

	// Validate configuration
	if err := validateGatewayConfig(config); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	// Create and start the gateway server
	server := gateway.NewServer(config)

	// Print configuration summary
	printGatewayConfig(config)

	// Start server in a goroutine
	errChan := make(chan error, 1)
	go func() {
		if err := server.Start(); err != nil {
			errChan <- fmt.Errorf("failed to start server: %w", err)
		}
	}()

	fmt.Printf("✅ FlowOps API Gateway started successfully\n")
	fmt.Printf("🌐 Server running on: http://%s:%d\n", config.Host, config.Port)
	fmt.Printf("📊 Health check: http://%s:%d/health\n", config.Host, config.Port)
	fmt.Printf("📈 Metrics: http://%s:%d/metrics\n", config.Host, config.Port)
	fmt.Printf("📚 API docs: http://%s:%d/api/v1\n", config.Host, config.Port)

	// Wait for interrupt signal or server error
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errChan:
		return err
	case sig := <-quit:
		fmt.Printf("\n🛑 Received signal: %v\n", sig)
	}

	fmt.Println("🔄 Shutting down gateway...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Stop(ctx); err != nil {
		return fmt.Errorf("failed to stop server gracefully: %w", err)
	}

	fmt.Println("✅ Gateway stopped successfully")
	return nil
}

// validateGatewayConfig validates the gateway configuration
func validateGatewayConfig(config *gateway.Config) error {
	if config.Port <= 0 || config.Port > 65535 {
		return fmt.Errorf("invalid port: %d", config.Port)
	}

	if config.EnableAuth && config.JWTSecret == "" {
		return fmt.Errorf("JWT secret is required when authentication is enabled")
	}

	if config.RateLimitRPS <= 0 {
		return fmt.Errorf("rate limit RPS must be positive: %d", config.RateLimitRPS)
	}

	// Validate TLS configuration
	if (config.CertFile != "" && config.KeyFile == "") || (config.CertFile == "" && config.KeyFile != "") {
		return fmt.Errorf("both TLS cert and key files must be provided")
	}

	if config.CertFile != "" {
		if _, err := os.Stat(config.CertFile); os.IsNotExist(err) {
			return fmt.Errorf("TLS cert file not found: %s", config.CertFile)
		}
	}

	if config.KeyFile != "" {
		if _, err := os.Stat(config.KeyFile); os.IsNotExist(err) {
			return fmt.Errorf("TLS key file not found: %s", config.KeyFile)
		}
	}

	return nil
}

// printGatewayConfig prints the gateway configuration summary
func printGatewayConfig(config *gateway.Config) {
	fmt.Println("\n📋 Gateway Configuration:")
	fmt.Printf("   Host: %s\n", config.Host)
	fmt.Printf("   Port: %d\n", config.Port)
	fmt.Printf("   Authentication: %v\n", config.EnableAuth)
	fmt.Printf("   CORS: %v\n", config.EnableCORS)
	fmt.Printf("   Rate Limiting: %v", config.EnableRateLimit)
	if config.EnableRateLimit {
		fmt.Printf(" (%d RPS)", config.RateLimitRPS)
	}
	fmt.Printf("\n   Metrics: %v\n", config.EnableMetrics)
	fmt.Printf("   TLS: %v", config.CertFile != "" && config.KeyFile != "")
	if config.CertFile != "" {
		fmt.Printf(" (cert: %s)", config.CertFile)
	}
	fmt.Println("\n")

	fmt.Println("🔌 Available Endpoints:")
	fmt.Printf("   GET  /health                     - Health check\n")
	fmt.Printf("   GET  /ready                      - Readiness check\n")
	fmt.Printf("   GET  /metrics                    - Prometheus metrics\n")
	fmt.Printf("   ANY  /api/v1/mlflow/*            - MLflow proxy\n")
	fmt.Printf("   ANY  /api/v1/models/*            - Model serving proxy\n")
	fmt.Printf("   GET  /api/v1/monitoring/services - Service status\n")
	fmt.Printf("   GET  /api/v1/pipelines           - List pipelines\n")
	fmt.Printf("   POST /api/v1/pipelines           - Create pipeline\n")
	fmt.Printf("   GET  /api/v1/model-management/models - List models\n")
	fmt.Println()
}