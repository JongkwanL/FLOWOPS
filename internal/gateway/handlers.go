package gateway

import (
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
)

// ServiceStatus represents the status of a service
type ServiceStatus struct {
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	Healthy   bool      `json:"healthy"`
	LastCheck time.Time `json:"last_check"`
	Latency   float64   `json:"latency_ms"`
	Error     string    `json:"error,omitempty"`
}

// Pipeline represents a ML pipeline
type Pipeline struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Status      string                 `json:"status"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Config      map[string]interface{} `json:"config"`
}

// Model represents a ML model
type Model struct {
	Name        string    `json:"name"`
	Version     string    `json:"version"`
	Status      string    `json:"status"`
	Accuracy    float64   `json:"accuracy"`
	LastTrained time.Time `json:"last_trained"`
	DeployedAt  time.Time `json:"deployed_at"`
}

// getServiceStatus returns the status of all monitored services
func (s *Server) getServiceStatus(c *gin.Context) {
	services := []ServiceStatus{
		s.checkServiceHealth("mlflow", "http://localhost:5000/health"),
		s.checkServiceHealth("model-serving", "http://localhost:8000/health"),
		s.checkServiceHealth("prometheus", "http://localhost:9090/-/healthy"),
		s.checkServiceHealth("grafana", "http://localhost:3000/api/health"),
	}
	
	c.JSON(http.StatusOK, gin.H{
		"services":  services,
		"timestamp": time.Now().Unix(),
	})
}

// checkServiceHealth checks the health of a specific service
func (s *Server) checkServiceHealth(name, url string) ServiceStatus {
	start := time.Now()
	
	client := &http.Client{
		Timeout: 5 * time.Second,
	}
	
	resp, err := client.Get(url)
	latency := time.Since(start).Seconds() * 1000 // Convert to milliseconds
	
	status := ServiceStatus{
		Name:      name,
		LastCheck: time.Now(),
		Latency:   latency,
	}
	
	if err != nil {
		status.Status = "error"
		status.Healthy = false
		status.Error = err.Error()
		return status
	}
	defer resp.Body.Close()
	
	if resp.StatusCode == http.StatusOK {
		status.Status = "healthy"
		status.Healthy = true
	} else {
		status.Status = "unhealthy"
		status.Healthy = false
		status.Error = fmt.Sprintf("HTTP %d", resp.StatusCode)
	}
	
	return status
}

// getMetrics returns gateway metrics
func (s *Server) getMetrics(c *gin.Context) {
	// This would typically return aggregated metrics
	// For now, redirect to Prometheus metrics endpoint
	c.Redirect(http.StatusFound, "/metrics")
}

// listPipelines returns a list of all pipelines
func (s *Server) listPipelines(c *gin.Context) {
	// Mock data - in real implementation, this would query a database
	pipelines := []Pipeline{
		{
			ID:          "pipeline-001",
			Name:        "Model Training Pipeline",
			Description: "Automated model training and validation",
			Status:      "active",
			CreatedAt:   time.Now().Add(-24 * time.Hour),
			UpdatedAt:   time.Now().Add(-1 * time.Hour),
			Config: map[string]interface{}{
				"model_type": "xgboost",
				"features":   []string{"feature1", "feature2", "feature3"},
				"target":     "target_variable",
			},
		},
		{
			ID:          "pipeline-002",
			Name:        "Data Preprocessing Pipeline",
			Description: "Data cleaning and feature engineering",
			Status:      "inactive",
			CreatedAt:   time.Now().Add(-48 * time.Hour),
			UpdatedAt:   time.Now().Add(-2 * time.Hour),
			Config: map[string]interface{}{
				"source": "s3://data-bucket/raw",
				"output": "s3://data-bucket/processed",
			},
		},
	}
	
	c.JSON(http.StatusOK, gin.H{
		"pipelines": pipelines,
		"total":     len(pipelines),
	})
}

// createPipeline creates a new pipeline
func (s *Server) createPipeline(c *gin.Context) {
	var pipeline Pipeline
	
	if err := c.ShouldBindJSON(&pipeline); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
		})
		return
	}
	
	// Generate ID and timestamps
	pipeline.ID = fmt.Sprintf("pipeline-%d", time.Now().Unix())
	pipeline.CreatedAt = time.Now()
	pipeline.UpdatedAt = time.Now()
	pipeline.Status = "created"
	
	// In real implementation, save to database
	
	c.JSON(http.StatusCreated, pipeline)
}

// getPipeline returns a specific pipeline
func (s *Server) getPipeline(c *gin.Context) {
	id := c.Param("id")
	
	// Mock data - in real implementation, query database
	pipeline := Pipeline{
		ID:          id,
		Name:        "Model Training Pipeline",
		Description: "Automated model training and validation",
		Status:      "running",
		CreatedAt:   time.Now().Add(-24 * time.Hour),
		UpdatedAt:   time.Now().Add(-1 * time.Hour),
		Config: map[string]interface{}{
			"model_type": "xgboost",
			"features":   []string{"feature1", "feature2", "feature3"},
			"target":     "target_variable",
		},
	}
	
	c.JSON(http.StatusOK, pipeline)
}

// updatePipeline updates an existing pipeline
func (s *Server) updatePipeline(c *gin.Context) {
	id := c.Param("id")
	
	var updates map[string]interface{}
	if err := c.ShouldBindJSON(&updates); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "invalid request body",
		})
		return
	}
	
	// In real implementation, update database record
	
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"updated": updates,
		"message": "pipeline updated successfully",
	})
}

// deletePipeline deletes a pipeline
func (s *Server) deletePipeline(c *gin.Context) {
	id := c.Param("id")
	
	// In real implementation, delete from database
	
	c.JSON(http.StatusOK, gin.H{
		"id":      id,
		"message": "pipeline deleted successfully",
	})
}

// runPipeline triggers a pipeline execution
func (s *Server) runPipeline(c *gin.Context) {
	id := c.Param("id")
	
	// Parse optional parameters
	var params map[string]interface{}
	c.ShouldBindJSON(&params)
	
	// In real implementation, trigger pipeline execution
	runID := fmt.Sprintf("run-%d", time.Now().Unix())
	
	c.JSON(http.StatusAccepted, gin.H{
		"pipeline_id": id,
		"run_id":      runID,
		"status":      "started",
		"parameters":  params,
		"started_at":  time.Now().Unix(),
	})
}

// listModels returns a list of all models
func (s *Server) listModels(c *gin.Context) {
	// Parse query parameters
	limit := 10
	if l := c.Query("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil {
			limit = parsed
		}
	}
	
	status := c.Query("status")
	
	// Mock data - in real implementation, query model registry
	models := []Model{
		{
			Name:        "fraud-detection",
			Version:     "v1.2.3",
			Status:      "deployed",
			Accuracy:    0.95,
			LastTrained: time.Now().Add(-48 * time.Hour),
			DeployedAt:  time.Now().Add(-24 * time.Hour),
		},
		{
			Name:        "recommendation-engine",
			Version:     "v2.1.0",
			Status:      "staging",
			Accuracy:    0.87,
			LastTrained: time.Now().Add(-12 * time.Hour),
			DeployedAt:  time.Time{},
		},
	}
	
	// Filter by status if provided
	if status != "" {
		filtered := make([]Model, 0)
		for _, model := range models {
			if model.Status == status {
				filtered = append(filtered, model)
			}
		}
		models = filtered
	}
	
	// Apply limit
	if len(models) > limit {
		models = models[:limit]
	}
	
	c.JSON(http.StatusOK, gin.H{
		"models": models,
		"total":  len(models),
	})
}

// getModelVersions returns all versions of a specific model
func (s *Server) getModelVersions(c *gin.Context) {
	name := c.Param("name")
	
	// Mock data - in real implementation, query model registry
	versions := []Model{
		{
			Name:        name,
			Version:     "v1.2.3",
			Status:      "deployed",
			Accuracy:    0.95,
			LastTrained: time.Now().Add(-48 * time.Hour),
			DeployedAt:  time.Now().Add(-24 * time.Hour),
		},
		{
			Name:        name,
			Version:     "v1.2.2",
			Status:      "archived",
			Accuracy:    0.93,
			LastTrained: time.Now().Add(-120 * time.Hour),
			DeployedAt:  time.Now().Add(-96 * time.Hour),
		},
		{
			Name:        name,
			Version:     "v1.2.1",
			Status:      "archived",
			Accuracy:    0.91,
			LastTrained: time.Now().Add(-240 * time.Hour),
			DeployedAt:  time.Now().Add(-216 * time.Hour),
		},
	}
	
	c.JSON(http.StatusOK, gin.H{
		"model":    name,
		"versions": versions,
		"total":    len(versions),
	})
}

// deployModel deploys a specific model version
func (s *Server) deployModel(c *gin.Context) {
	name := c.Param("name")
	version := c.Param("version")
	
	var deployConfig map[string]interface{}
	c.ShouldBindJSON(&deployConfig)
	
	// In real implementation, trigger deployment pipeline
	deploymentID := fmt.Sprintf("deploy-%d", time.Now().Unix())
	
	c.JSON(http.StatusAccepted, gin.H{
		"model":         name,
		"version":       version,
		"deployment_id": deploymentID,
		"status":        "deploying",
		"config":        deployConfig,
		"started_at":    time.Now().Unix(),
	})
}

// rollbackModel rolls back to a previous model version
func (s *Server) rollbackModel(c *gin.Context) {
	name := c.Param("name")
	version := c.Param("version")
	
	// In real implementation, trigger rollback pipeline
	rollbackID := fmt.Sprintf("rollback-%d", time.Now().Unix())
	
	c.JSON(http.StatusAccepted, gin.H{
		"model":       name,
		"version":     version,
		"rollback_id": rollbackID,
		"status":      "rolling_back",
		"started_at":  time.Now().Unix(),
	})
}