package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/flowops/flowops/internal/gateway"
)

func main() {
	// Command line flags
	var (
		host           = flag.String("host", "0.0.0.0", "Server host")
		port           = flag.Int("port", 8080, "Server port")
		enableAuth     = flag.Bool("auth", false, "Enable authentication")
		enableMetrics  = flag.Bool("metrics", true, "Enable metrics collection")
		enableCORS     = flag.Bool("cors", true, "Enable CORS")
		enableRateLimit = flag.Bool("ratelimit", true, "Enable rate limiting")
		rateLimitRPS   = flag.Int("ratelimit-rps", 100, "Rate limit requests per second")
		jwtSecret      = flag.String("jwt-secret", "", "JWT secret key")
		certFile       = flag.String("cert", "", "TLS certificate file")
		keyFile        = flag.String("key", "", "TLS private key file")
	)
	flag.Parse()

	// Load configuration
	config := &gateway.Config{
		Host:            *host,
		Port:            *port,
		ReadTimeout:     15 * time.Second,
		WriteTimeout:    15 * time.Second,
		IdleTimeout:     60 * time.Second,
		EnableCORS:      *enableCORS,
		EnableAuth:      *enableAuth,
		EnableMetrics:   *enableMetrics,
		EnableRateLimit: *enableRateLimit,
		RateLimitRPS:    *rateLimitRPS,
		JWTSecret:       *jwtSecret,
		CertFile:        *certFile,
		KeyFile:         *keyFile,
	}

	// Override with environment variables if present
	if envHost := os.Getenv("GATEWAY_HOST"); envHost != "" {
		config.Host = envHost
	}
	if envJWTSecret := os.Getenv("JWT_SECRET"); envJWTSecret != "" {
		config.JWTSecret = envJWTSecret
	}
	if envCertFile := os.Getenv("TLS_CERT_FILE"); envCertFile != "" {
		config.CertFile = envCertFile
	}
	if envKeyFile := os.Getenv("TLS_KEY_FILE"); envKeyFile != "" {
		config.KeyFile = envKeyFile
	}

	// Create server
	server := gateway.NewServer(config)

	// Start server in a goroutine
	go func() {
		if err := server.Start(); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	log.Printf("FlowOps API Gateway started on %s:%d", config.Host, config.Port)
	
	// Print feature status
	log.Printf("Features enabled:")
	log.Printf("  - Authentication: %v", config.EnableAuth)
	log.Printf("  - Metrics: %v", config.EnableMetrics)
	log.Printf("  - CORS: %v", config.EnableCORS)
	log.Printf("  - Rate Limiting: %v (RPS: %d)", config.EnableRateLimit, config.RateLimitRPS)
	log.Printf("  - TLS: %v", config.CertFile != "" && config.KeyFile != "")

	// Wait for interrupt signal to gracefully shutdown the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Stop(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Println("Server exited")
}