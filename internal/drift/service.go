package drift

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// DriftService provides HTTP API for drift detection
type DriftService struct {
	detector *DriftDetector
	metrics  *ServiceMetrics
	config   *ServiceConfig
}

// ServiceConfig holds service configuration
type ServiceConfig struct {
	Host         string `json:"host"`
	Port         int    `json:"port"`
	EnableMetrics bool   `json:"enable_metrics"`
	EnableCORS   bool   `json:"enable_cors"`
	LogLevel     string `json:"log_level"`
}

// ServiceMetrics holds Prometheus metrics
type ServiceMetrics struct {
	RequestsTotal   *prometheus.CounterVec
	RequestDuration *prometheus.HistogramVec
	DriftDetected   *prometheus.CounterVec
	ActiveModels    prometheus.Gauge
}

// DriftRequest represents a drift detection request
type DriftRequest struct {
	ModelName   string             `json:"model_name" binding:"required"`
	Features    map[string][]float64 `json:"features" binding:"required"`
	Timestamp   *time.Time         `json:"timestamp,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ReferenceDataRequest represents reference data setup request
type ReferenceDataRequest struct {
	ModelName string             `json:"model_name" binding:"required"`
	Features  map[string][]float64 `json:"features" binding:"required"`
}

// NewDriftService creates a new drift detection service
func NewDriftService(detector *DriftDetector, config *ServiceConfig) *DriftService {
	if config == nil {
		config = &ServiceConfig{
			Host:          "0.0.0.0",
			Port:          8090,
			EnableMetrics: true,
			EnableCORS:    true,
			LogLevel:      "info",
		}
	}
	
	return &DriftService{
		detector: detector,
		metrics:  NewServiceMetrics(),
		config:   config,
	}
}

// NewServiceMetrics creates Prometheus metrics
func NewServiceMetrics() *ServiceMetrics {
	requestsTotal := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "drift_service_requests_total",
			Help: "Total number of drift detection requests",
		},
		[]string{"method", "endpoint", "status"},
	)
	
	requestDuration := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "drift_service_request_duration_seconds",
			Help:    "Time spent processing drift detection requests",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "endpoint"},
	)
	
	driftDetected := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "drift_detected_total",
			Help: "Total number of drift detections",
		},
		[]string{"model", "feature", "method", "severity"},
	)
	
	activeModels := prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "drift_active_models",
			Help: "Number of models being monitored for drift",
		},
	)
	
	// Register metrics
	prometheus.MustRegister(requestsTotal, requestDuration, driftDetected, activeModels)
	
	return &ServiceMetrics{
		RequestsTotal:   requestsTotal,
		RequestDuration: requestDuration,
		DriftDetected:   driftDetected,
		ActiveModels:    activeModels,
	}
}

// SetupRoutes configures HTTP routes
func (s *DriftService) SetupRoutes() *gin.Engine {
	router := gin.Default()
	
	// CORS middleware
	if s.config.EnableCORS {
		router.Use(func(c *gin.Context) {
			c.Header("Access-Control-Allow-Origin", "*")
			c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			c.Header("Access-Control-Allow-Headers", "Content-Type, Authorization")
			
			if c.Request.Method == "OPTIONS" {
				c.AbortWithStatus(204)
				return
			}
			
			c.Next()
		})
	}
	
	// Metrics middleware
	router.Use(s.metricsMiddleware())
	
	// Health check
	router.GET("/health", s.healthCheck)
	router.GET("/ready", s.readinessCheck)
	
	// Metrics endpoint
	if s.config.EnableMetrics {
		router.GET("/metrics", gin.WrapH(promhttp.Handler()))
	}
	
	// API routes
	api := router.Group("/api/v1")
	{
		// Reference data management
		api.POST("/reference", s.setReferenceData)
		api.GET("/reference/:model", s.getReferenceData)
		api.DELETE("/reference/:model", s.deleteReferenceData)
		
		// Drift detection
		api.POST("/detect", s.detectDrift)
		api.POST("/detect/batch", s.detectBatchDrift)
		
		// Results and monitoring
		api.GET("/results", s.getResults)
		api.GET("/results/:model", s.getModelResults)
		api.GET("/status", s.getStatus)
		api.GET("/features/:model", s.getFeatureStatus)
		
		// Configuration
		api.GET("/config", s.getConfig)
		api.PUT("/config", s.updateConfig)
	}
	
	return router
}

// metricsMiddleware records request metrics
func (s *DriftService) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		c.Next()
		
		duration := time.Since(start)
		status := strconv.Itoa(c.Writer.Status())
		
		s.metrics.RequestsTotal.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
			status,
		).Inc()
		
		s.metrics.RequestDuration.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
		).Observe(duration.Seconds())
	}
}

// healthCheck returns service health status
func (s *DriftService) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "healthy",
		"service":   "drift-detection",
		"timestamp": time.Now().Unix(),
		"version":   "1.0.0",
	})
}

// readinessCheck returns service readiness status
func (s *DriftService) readinessCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"ready":     true,
		"timestamp": time.Now().Unix(),
	})
}

// setReferenceData sets reference data for drift detection
func (s *DriftService) setReferenceData(c *gin.Context) {
	var req ReferenceDataRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
			"details": err.Error(),
		})
		return
	}
	
	// Set reference data for each feature
	errors := make(map[string]string)
	success := make(map[string]int)
	
	for featureName, data := range req.Features {
		fullFeatureName := fmt.Sprintf("%s_%s", req.ModelName, featureName)
		err := s.detector.SetReferenceData(fullFeatureName, data)
		if err != nil {
			errors[featureName] = err.Error()
		} else {
			success[featureName] = len(data)
		}
	}
	
	if len(errors) > 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "failed to set some reference data",
			"success": success,
			"errors":  errors,
		})
		return
	}
	
	c.JSON(http.StatusOK, gin.H{
		"message": "reference data set successfully",
		"model":   req.ModelName,
		"features": success,
	})
}

// detectDrift performs drift detection
func (s *DriftService) detectDrift(c *gin.Context) {
	var req DriftRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
			"details": err.Error(),
		})
		return
	}
	
	// Detect drift for each feature
	results := make(map[string]*DriftResult)
	alerts := make([]string, 0)
	
	for featureName, data := range req.Features {
		fullFeatureName := fmt.Sprintf("%s_%s", req.ModelName, featureName)
		result, err := s.detector.DetectDrift(req.ModelName, fullFeatureName, data)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf("failed to detect drift for feature %s", featureName),
				"details": err.Error(),
			})
			return
		}
		
		results[featureName] = result
		
		// Record metrics
		if result.Alert {
			alerts = append(alerts, featureName)
			s.metrics.DriftDetected.WithLabelValues(
				req.ModelName,
				featureName,
				result.Method,
				result.Severity,
			).Inc()
		}
	}
	
	response := gin.H{
		"model":      req.ModelName,
		"timestamp":  time.Now().Unix(),
		"results":    results,
		"alert":      len(alerts) > 0,
		"alert_features": alerts,
	}
	
	if req.Metadata != nil {
		response["metadata"] = req.Metadata
	}
	
	c.JSON(http.StatusOK, response)
}

// detectBatchDrift performs batch drift detection
func (s *DriftService) detectBatchDrift(c *gin.Context) {
	var req DriftRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
			"details": err.Error(),
		})
		return
	}
	
	// Create feature map with full names
	featureData := make(map[string][]float64)
	for featureName, data := range req.Features {
		fullFeatureName := fmt.Sprintf("%s_%s", req.ModelName, featureName)
		featureData[fullFeatureName] = data
	}
	
	results, err := s.detector.DetectBatchDrift(req.ModelName, featureData)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "failed to perform batch drift detection",
			"details": err.Error(),
		})
		return
	}
	
	// Convert back to original feature names
	cleanResults := make(map[string]*DriftResult)
	alerts := make([]string, 0)
	
	for fullName, result := range results {
		// Extract original feature name
		featureName := fullName[len(req.ModelName)+1:]
		cleanResults[featureName] = result
		
		if result.Alert {
			alerts = append(alerts, featureName)
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"model":          req.ModelName,
		"timestamp":      time.Now().Unix(),
		"results":        cleanResults,
		"alert":          len(alerts) > 0,
		"alert_features": alerts,
		"summary": gin.H{
			"total_features":   len(cleanResults),
			"drift_detected":   len(alerts),
			"drift_percentage": float64(len(alerts)) / float64(len(cleanResults)) * 100,
		},
	})
}

// getResults returns recent drift detection results
func (s *DriftService) getResults(c *gin.Context) {
	limitStr := c.DefaultQuery("limit", "50")
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		limit = 50
	}
	
	results := s.detector.GetRecentResults(limit)
	
	c.JSON(http.StatusOK, gin.H{
		"results":   results,
		"count":     len(results),
		"timestamp": time.Now().Unix(),
	})
}

// getModelResults returns results for a specific model
func (s *DriftService) getModelResults(c *gin.Context) {
	modelName := c.Param("model")
	limitStr := c.DefaultQuery("limit", "50")
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		limit = 50
	}
	
	allResults := s.detector.GetRecentResults(limit * 2) // Get more to filter
	modelResults := make([]DriftResult, 0)
	
	for _, result := range allResults {
		if result.ModelName == modelName {
			modelResults = append(modelResults, result)
			if len(modelResults) >= limit {
				break
			}
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"model":     modelName,
		"results":   modelResults,
		"count":     len(modelResults),
		"timestamp": time.Now().Unix(),
	})
}

// getStatus returns overall service status
func (s *DriftService) getStatus(c *gin.Context) {
	status := s.detector.GetFeatureStatus()
	
	c.JSON(http.StatusOK, gin.H{
		"status":       "running",
		"features":     status,
		"total_features": len(status),
		"timestamp":    time.Now().Unix(),
	})
}

// getFeatureStatus returns status for features of a specific model
func (s *DriftService) getFeatureStatus(c *gin.Context) {
	modelName := c.Param("model")
	allStatus := s.detector.GetFeatureStatus()
	
	modelStatus := make(map[string]interface{})
	for featureName, status := range allStatus {
		if len(featureName) > len(modelName) && 
		   featureName[:len(modelName)] == modelName {
			originalName := featureName[len(modelName)+1:]
			modelStatus[originalName] = status
		}
	}
	
	c.JSON(http.StatusOK, gin.H{
		"model":     modelName,
		"features":  modelStatus,
		"count":     len(modelStatus),
		"timestamp": time.Now().Unix(),
	})
}

// getConfig returns current detector configuration
func (s *DriftService) getConfig(c *gin.Context) {
	c.JSON(http.StatusOK, s.detector.config)
}

// updateConfig updates detector configuration
func (s *DriftService) updateConfig(c *gin.Context) {
	var newConfig DetectorConfig
	if err := c.ShouldBindJSON(&newConfig); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid configuration",
			"details": err.Error(),
		})
		return
	}
	
	s.detector.config = &newConfig
	
	c.JSON(http.StatusOK, gin.H{
		"message": "configuration updated successfully",
		"config":  newConfig,
	})
}

// getReferenceData placeholder - not implemented for security
func (s *DriftService) getReferenceData(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{
		"error": "reference data retrieval not implemented for security reasons",
	})
}

// deleteReferenceData placeholder - not implemented for security
func (s *DriftService) deleteReferenceData(c *gin.Context) {
	c.JSON(http.StatusNotImplemented, gin.H{
		"error": "reference data deletion not implemented for security reasons",
	})
}

// Start starts the drift detection service
func (s *DriftService) Start() error {
	router := s.SetupRoutes()
	address := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
	
	fmt.Printf("Starting Drift Detection Service on %s\n", address)
	return router.Run(address)
}