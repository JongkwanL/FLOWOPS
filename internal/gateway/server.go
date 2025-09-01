package gateway

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Server represents the API gateway server
type Server struct {
	router     *gin.Engine
	httpServer *http.Server
	config     *Config
	middleware *MiddlewareManager
	metrics    *Metrics
}

// Config holds the server configuration
type Config struct {
	Host           string
	Port           int
	ReadTimeout    time.Duration
	WriteTimeout   time.Duration
	IdleTimeout    time.Duration
	EnableCORS     bool
	EnableAuth     bool
	EnableMetrics  bool
	EnableRateLimit bool
	RateLimitRPS   int
	JWTSecret      string
	CertFile       string
	KeyFile        string
}

// NewServer creates a new API gateway server
func NewServer(config *Config) *Server {
	// Set Gin mode based on environment
	gin.SetMode(gin.ReleaseMode)
	
	router := gin.New()
	
	// Create server instance
	server := &Server{
		router: router,
		config: config,
		metrics: NewMetrics(),
	}
	
	// Initialize middleware manager
	server.middleware = NewMiddlewareManager(config, server.metrics)
	
	// Setup middleware
	server.setupMiddleware()
	
	// Setup routes
	server.setupRoutes()
	
	// Create HTTP server
	server.httpServer = &http.Server{
		Addr:         fmt.Sprintf("%s:%d", config.Host, config.Port),
		Handler:      router,
		ReadTimeout:  config.ReadTimeout,
		WriteTimeout: config.WriteTimeout,
		IdleTimeout:  config.IdleTimeout,
	}
	
	return server
}

// setupMiddleware configures all middleware
func (s *Server) setupMiddleware() {
	// Recovery middleware
	s.router.Use(gin.Recovery())
	
	// Logging middleware
	s.router.Use(s.middleware.Logger())
	
	// CORS middleware
	if s.config.EnableCORS {
		s.router.Use(s.middleware.CORS())
	}
	
	// Rate limiting middleware
	if s.config.EnableRateLimit {
		s.router.Use(s.middleware.RateLimit())
	}
	
	// Metrics middleware
	if s.config.EnableMetrics {
		s.router.Use(s.middleware.Metrics())
	}
	
	// Security headers middleware
	s.router.Use(s.middleware.Security())
}

// setupRoutes configures all API routes
func (s *Server) setupRoutes() {
	// Health check endpoints
	s.router.GET("/health", s.healthCheck)
	s.router.GET("/ready", s.readinessCheck)
	
	// Metrics endpoint
	if s.config.EnableMetrics {
		s.router.GET("/metrics", gin.WrapH(promhttp.Handler()))
	}
	
	// API version 1
	v1 := s.router.Group("/api/v1")
	{
		// MLflow proxy routes
		mlflow := v1.Group("/mlflow")
		{
			mlflow.Any("/*path", s.proxyMLflow)
		}
		
		// Model serving proxy routes
		models := v1.Group("/models")
		{
			models.Any("/*path", s.proxyModelServing)
		}
		
		// Monitoring endpoints
		monitoring := v1.Group("/monitoring")
		{
			monitoring.GET("/services", s.getServiceStatus)
			monitoring.GET("/metrics", s.getMetrics)
		}
		
		// Pipeline management
		pipelines := v1.Group("/pipelines")
		if s.config.EnableAuth {
			pipelines.Use(s.middleware.Auth())
		}
		{
			pipelines.GET("/", s.listPipelines)
			pipelines.POST("/", s.createPipeline)
			pipelines.GET("/:id", s.getPipeline)
			pipelines.PUT("/:id", s.updatePipeline)
			pipelines.DELETE("/:id", s.deletePipeline)
			pipelines.POST("/:id/run", s.runPipeline)
		}
		
		// Model management
		modelMgmt := v1.Group("/model-management")
		if s.config.EnableAuth {
			modelMgmt.Use(s.middleware.Auth())
		}
		{
			modelMgmt.GET("/models", s.listModels)
			modelMgmt.GET("/models/:name/versions", s.getModelVersions)
			modelMgmt.POST("/models/:name/versions/:version/deploy", s.deployModel)
			modelMgmt.POST("/models/:name/versions/:version/rollback", s.rollbackModel)
		}
	}
}

// Start starts the API gateway server
func (s *Server) Start() error {
	log.Printf("Starting API Gateway on %s", s.httpServer.Addr)
	
	// Register metrics
	if s.config.EnableMetrics {
		prometheus.MustRegister(s.metrics.collectors...)
	}
	
	// Start server with TLS if certificates are provided
	if s.config.CertFile != "" && s.config.KeyFile != "" {
		return s.httpServer.ListenAndServeTLS(s.config.CertFile, s.config.KeyFile)
	}
	
	return s.httpServer.ListenAndServe()
}

// Stop gracefully stops the server
func (s *Server) Stop(ctx context.Context) error {
	log.Println("Shutting down API Gateway...")
	return s.httpServer.Shutdown(ctx)
}

// healthCheck returns the health status of the gateway
func (s *Server) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"version":   "1.0.0",
	})
}

// readinessCheck returns the readiness status of the gateway
func (s *Server) readinessCheck(c *gin.Context) {
	// Check if dependent services are ready
	ready := true
	services := map[string]bool{
		"mlflow": s.checkMLflowReady(),
		"models": s.checkModelServingReady(),
	}
	
	for _, status := range services {
		if !status {
			ready = false
			break
		}
	}
	
	status := http.StatusOK
	if !ready {
		status = http.StatusServiceUnavailable
	}
	
	c.JSON(status, gin.H{
		"ready":    ready,
		"services": services,
		"timestamp": time.Now().Unix(),
	})
}

// checkMLflowReady checks if MLflow is ready
func (s *Server) checkMLflowReady() bool {
	// Simple health check to MLflow
	resp, err := http.Get("http://localhost:5000/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// checkModelServingReady checks if model serving is ready
func (s *Server) checkModelServingReady() bool {
	// Simple health check to model serving
	resp, err := http.Get("http://localhost:8000/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}