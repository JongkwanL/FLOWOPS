package monitor

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// ServiceStatus represents the health status of a service
type ServiceStatus struct {
	Name         string                 `json:"name"`
	URL          string                 `json:"url"`
	Status       string                 `json:"status"`       // healthy, unhealthy, unknown
	ResponseTime time.Duration          `json:"response_time"`
	LastCheck    time.Time              `json:"last_check"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Error        string                 `json:"error,omitempty"`
}

// Agent monitors the health of FlowOps services
type Agent struct {
	services map[string]*ServiceConfig
	statuses map[string]*ServiceStatus
	mutex    sync.RWMutex
	
	// Prometheus metrics
	serviceUpGauge     *prometheus.GaugeVec
	responseTimeGauge  *prometheus.GaugeVec
	checkCounterVec    *prometheus.CounterVec
	
	// Configuration
	checkInterval time.Duration
	timeout       time.Duration
	
	// Control
	ctx    context.Context
	cancel context.CancelFunc
}

// ServiceConfig defines how to monitor a service
type ServiceConfig struct {
	Name         string            `json:"name"`
	URL          string            `json:"url"`
	HealthPath   string            `json:"health_path"`
	CheckType    string            `json:"check_type"`    // http, tcp, command
	Timeout      time.Duration     `json:"timeout"`
	Interval     time.Duration     `json:"interval"`
	Headers      map[string]string `json:"headers,omitempty"`
	ExpectedCode int               `json:"expected_code"`
	Command      string            `json:"command,omitempty"`
}

// NewAgent creates a new monitoring agent
func NewAgent(checkInterval, timeout time.Duration) *Agent {
	ctx, cancel := context.WithCancel(context.Background())
	
	agent := &Agent{
		services:      make(map[string]*ServiceConfig),
		statuses:      make(map[string]*ServiceStatus),
		checkInterval: checkInterval,
		timeout:       timeout,
		ctx:           ctx,
		cancel:        cancel,
	}
	
	agent.initMetrics()
	return agent
}

// initMetrics initializes Prometheus metrics
func (a *Agent) initMetrics() {
	a.serviceUpGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "flowops_service_up",
			Help: "Whether a FlowOps service is up (1) or down (0)",
		},
		[]string{"service", "url"},
	)
	
	a.responseTimeGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "flowops_service_response_time_seconds",
			Help: "Response time of FlowOps services in seconds",
		},
		[]string{"service", "url"},
	)
	
	a.checkCounterVec = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "flowops_service_checks_total",
			Help: "Total number of health checks performed",
		},
		[]string{"service", "status"},
	)
}

// AddService adds a service to monitor
func (a *Agent) AddService(config *ServiceConfig) {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	
	// Set defaults
	if config.HealthPath == "" {
		config.HealthPath = "/health"
	}
	if config.CheckType == "" {
		config.CheckType = "http"
	}
	if config.Timeout == 0 {
		config.Timeout = a.timeout
	}
	if config.Interval == 0 {
		config.Interval = a.checkInterval
	}
	if config.ExpectedCode == 0 {
		config.ExpectedCode = 200
	}
	
	a.services[config.Name] = config
	
	// Initialize status
	a.statuses[config.Name] = &ServiceStatus{
		Name:      config.Name,
		URL:       config.URL,
		Status:    "unknown",
		LastCheck: time.Time{},
	}
}

// RemoveService removes a service from monitoring
func (a *Agent) RemoveService(name string) {
	a.mutex.Lock()
	defer a.mutex.Unlock()
	
	delete(a.services, name)
	delete(a.statuses, name)
}

// GetStatus returns the current status of a service
func (a *Agent) GetStatus(name string) (*ServiceStatus, bool) {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	
	status, exists := a.statuses[name]
	if !exists {
		return nil, false
	}
	
	// Return a copy to avoid data races
	statusCopy := *status
	return &statusCopy, true
}

// GetAllStatuses returns the status of all monitored services
func (a *Agent) GetAllStatuses() map[string]*ServiceStatus {
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	
	result := make(map[string]*ServiceStatus)
	for name, status := range a.statuses {
		statusCopy := *status
		result[name] = &statusCopy
	}
	return result
}

// Start begins monitoring all configured services
func (a *Agent) Start() error {
	a.mutex.RLock()
	serviceConfigs := make([]*ServiceConfig, 0, len(a.services))
	for _, config := range a.services {
		serviceConfigs = append(serviceConfigs, config)
	}
	a.mutex.RUnlock()
	
	// Start monitoring each service in its own goroutine
	for _, config := range serviceConfigs {
		go a.monitorService(config)
	}
	
	return nil
}

// Stop stops the monitoring agent
func (a *Agent) Stop() {
	a.cancel()
}

// monitorService monitors a single service
func (a *Agent) monitorService(config *ServiceConfig) {
	ticker := time.NewTicker(config.Interval)
	defer ticker.Stop()
	
	// Perform initial check
	a.checkService(config)
	
	for {
		select {
		case <-a.ctx.Done():
			return
		case <-ticker.C:
			a.checkService(config)
		}
	}
}

// checkService performs a health check on a service
func (a *Agent) checkService(config *ServiceConfig) {
	start := time.Now()
	
	var status *ServiceStatus
	switch config.CheckType {
	case "http":
		status = a.checkHTTPService(config)
	case "tcp":
		status = a.checkTCPService(config)
	case "command":
		status = a.checkCommandService(config)
	default:
		status = &ServiceStatus{
			Name:         config.Name,
			URL:          config.URL,
			Status:       "unknown",
			ResponseTime: 0,
			LastCheck:    time.Now(),
			Error:        fmt.Sprintf("unknown check type: %s", config.CheckType),
		}
	}
	
	status.ResponseTime = time.Since(start)
	status.LastCheck = time.Now()
	
	// Update status
	a.mutex.Lock()
	a.statuses[config.Name] = status
	a.mutex.Unlock()
	
	// Update Prometheus metrics
	a.updateMetrics(status)
}

// updateMetrics updates Prometheus metrics
func (a *Agent) updateMetrics(status *ServiceStatus) {
	serviceUpValue := 0.0
	if status.Status == "healthy" {
		serviceUpValue = 1.0
	}
	
	a.serviceUpGauge.WithLabelValues(status.Name, status.URL).Set(serviceUpValue)
	a.responseTimeGauge.WithLabelValues(status.Name, status.URL).Set(status.ResponseTime.Seconds())
	a.checkCounterVec.WithLabelValues(status.Name, status.Status).Inc()
}

// StartMetricsServer starts the Prometheus metrics HTTP server
func (a *Agent) StartMetricsServer(addr string) error {
	http.Handle("/metrics", promhttp.Handler())
	http.HandleFunc("/health", a.handleHealth)
	http.HandleFunc("/status", a.handleStatus)
	
	return http.ListenAndServe(addr, nil)
}

// handleHealth handles health check requests for the agent itself
func (a *Agent) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	response := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().Unix(),
		"agent":     "flowops-monitor",
		"version":   "1.0.0",
	}
	
	json.NewEncoder(w).Encode(response)
}

// handleStatus handles status requests for all monitored services
func (a *Agent) handleStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	
	statuses := a.GetAllStatuses()
	
	response := map[string]interface{}{
		"timestamp": time.Now().Unix(),
		"services":  statuses,
		"count":     len(statuses),
	}
	
	json.NewEncoder(w).Encode(response)
}