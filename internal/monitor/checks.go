package monitor

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

// checkHTTPService performs HTTP health check
func (a *Agent) checkHTTPService(config *ServiceConfig) *ServiceStatus {
	ctx, cancel := context.WithTimeout(a.ctx, config.Timeout)
	defer cancel()
	
	url := config.URL + config.HealthPath
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return &ServiceStatus{
			Name:   config.Name,
			URL:    config.URL,
			Status: "unhealthy",
			Error:  fmt.Sprintf("failed to create request: %v", err),
		}
	}
	
	// Add custom headers
	for key, value := range config.Headers {
		req.Header.Set(key, value)
	}
	
	client := &http.Client{
		Timeout: config.Timeout,
	}
	
	resp, err := client.Do(req)
	if err != nil {
		return &ServiceStatus{
			Name:   config.Name,
			URL:    config.URL,
			Status: "unhealthy",
			Error:  fmt.Sprintf("request failed: %v", err),
		}
	}
	defer resp.Body.Close()
	
	// Check status code
	if resp.StatusCode != config.ExpectedCode {
		return &ServiceStatus{
			Name:   config.Name,
			URL:    config.URL,
			Status: "unhealthy",
			Error:  fmt.Sprintf("unexpected status code: %d", resp.StatusCode),
		}
	}
	
	// Try to parse response for additional metadata
	metadata := make(map[string]interface{})
	if body, err := io.ReadAll(resp.Body); err == nil {
		var jsonData map[string]interface{}
		if json.Unmarshal(body, &jsonData) == nil {
			metadata = jsonData
		}
	}
	
	return &ServiceStatus{
		Name:     config.Name,
		URL:      config.URL,
		Status:   "healthy",
		Metadata: metadata,
	}
}

// checkTCPService performs TCP connection check
func (a *Agent) checkTCPService(config *ServiceConfig) *ServiceStatus {
	ctx, cancel := context.WithTimeout(a.ctx, config.Timeout)
	defer cancel()
	
	var d net.Dialer
	conn, err := d.DialContext(ctx, "tcp", config.URL)
	if err != nil {
		return &ServiceStatus{
			Name:   config.Name,
			URL:    config.URL,
			Status: "unhealthy",
			Error:  fmt.Sprintf("connection failed: %v", err),
		}
	}
	defer conn.Close()
	
	return &ServiceStatus{
		Name:   config.Name,
		URL:    config.URL,
		Status: "healthy",
	}
}

// checkCommandService performs command-based check
func (a *Agent) checkCommandService(config *ServiceConfig) *ServiceStatus {
	ctx, cancel := context.WithTimeout(a.ctx, config.Timeout)
	defer cancel()
	
	parts := strings.Fields(config.Command)
	if len(parts) == 0 {
		return &ServiceStatus{
			Name:   config.Name,
			URL:    config.URL,
			Status: "unhealthy",
			Error:  "empty command",
		}
	}
	
	cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		return &ServiceStatus{
			Name:   config.Name,
			URL:    config.URL,
			Status: "unhealthy",
			Error:  fmt.Sprintf("command failed: %v, output: %s", err, string(output)),
		}
	}
	
	metadata := map[string]interface{}{
		"command_output": string(output),
		"exit_code":      0,
	}
	
	return &ServiceStatus{
		Name:     config.Name,
		URL:      config.URL,
		Status:   "healthy",
		Metadata: metadata,
	}
}

// MLflowChecker provides specialized checks for MLflow
type MLflowChecker struct {
	baseURL string
	timeout time.Duration
}

// NewMLflowChecker creates a new MLflow checker
func NewMLflowChecker(baseURL string, timeout time.Duration) *MLflowChecker {
	return &MLflowChecker{
		baseURL: baseURL,
		timeout: timeout,
	}
}

// CheckExperiments checks MLflow experiments
func (m *MLflowChecker) CheckExperiments() (map[string]interface{}, error) {
	client := &http.Client{Timeout: m.timeout}
	
	resp, err := client.Get(m.baseURL + "/api/2.0/mlflow/experiments/list")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("MLflow API returned %d", resp.StatusCode)
	}
	
	var data map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	
	metadata := map[string]interface{}{
		"experiments_count": len(data["experiments"].([]interface{})),
		"api_status":        "healthy",
	}
	
	return metadata, nil
}

// CheckModels checks registered models
func (m *MLflowChecker) CheckModels() (map[string]interface{}, error) {
	client := &http.Client{Timeout: m.timeout}
	
	resp, err := client.Get(m.baseURL + "/api/2.0/mlflow/registered-models/list")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("MLflow Models API returned %d", resp.StatusCode)
	}
	
	var data map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	
	models := data["registered_models"].([]interface{})
	metadata := map[string]interface{}{
		"registered_models_count": len(models),
		"api_status":              "healthy",
	}
	
	return metadata, nil
}

// DVCChecker provides specialized checks for DVC
type DVCChecker struct {
	repoPath string
	timeout  time.Duration
}

// NewDVCChecker creates a new DVC checker
func NewDVCChecker(repoPath string, timeout time.Duration) *DVCChecker {
	return &DVCChecker{
		repoPath: repoPath,
		timeout:  timeout,
	}
}

// CheckStatus checks DVC repository status
func (d *DVCChecker) CheckStatus() (map[string]interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), d.timeout)
	defer cancel()
	
	cmd := exec.CommandContext(ctx, "dvc", "status", "--json")
	cmd.Dir = d.repoPath
	
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("dvc status failed: %v", err)
	}
	
	var status map[string]interface{}
	if err := json.Unmarshal(output, &status); err != nil {
		return nil, err
	}
	
	metadata := map[string]interface{}{
		"dvc_status": status,
		"repo_path":  d.repoPath,
	}
	
	return metadata, nil
}

// CheckRemotes checks DVC remote status
func (d *DVCChecker) CheckRemotes() (map[string]interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), d.timeout)
	defer cancel()
	
	cmd := exec.CommandContext(ctx, "dvc", "remote", "list")
	cmd.Dir = d.repoPath
	
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("dvc remote list failed: %v", err)
	}
	
	remotes := strings.Split(strings.TrimSpace(string(output)), "\n")
	metadata := map[string]interface{}{
		"remotes_count": len(remotes),
		"remotes":       remotes,
	}
	
	return metadata, nil
}

// ModelAPIChecker provides specialized checks for model serving API
type ModelAPIChecker struct {
	baseURL string
	timeout time.Duration
}

// NewModelAPIChecker creates a new model API checker
func NewModelAPIChecker(baseURL string, timeout time.Duration) *ModelAPIChecker {
	return &ModelAPIChecker{
		baseURL: baseURL,
		timeout: timeout,
	}
}

// CheckHealth checks API health
func (m *ModelAPIChecker) CheckHealth() (map[string]interface{}, error) {
	client := &http.Client{Timeout: m.timeout}
	
	resp, err := client.Get(m.baseURL + "/health")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API health check returned %d", resp.StatusCode)
	}
	
	var health map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		return nil, err
	}
	
	return health, nil
}

// CheckMetrics checks API metrics
func (m *ModelAPIChecker) CheckMetrics() (map[string]interface{}, error) {
	client := &http.Client{Timeout: m.timeout}
	
	resp, err := client.Get(m.baseURL + "/metrics")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("API metrics returned %d", resp.StatusCode)
	}
	
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	
	// Parse Prometheus metrics
	metrics := parsePrometheusMetrics(string(body))
	
	return map[string]interface{}{
		"metrics_available": true,
		"metrics_count":     len(metrics),
		"key_metrics":       metrics,
	}, nil
}

// parsePrometheusMetrics parses basic Prometheus metrics
func parsePrometheusMetrics(content string) map[string]string {
	metrics := make(map[string]string)
	lines := strings.Split(content, "\n")
	
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		
		parts := strings.Fields(line)
		if len(parts) >= 2 {
			// Extract metric name and value
			metricName := strings.Split(parts[0], "{")[0]
			metricValue := parts[len(parts)-1]
			
			// Keep only key metrics
			if strings.Contains(metricName, "request") ||
				strings.Contains(metricName, "prediction") ||
				strings.Contains(metricName, "duration") ||
				strings.Contains(metricName, "error") {
				metrics[metricName] = metricValue
			}
		}
	}
	
	return metrics
}

// KubernetesChecker provides checks for Kubernetes deployments
type KubernetesChecker struct {
	namespace string
	timeout   time.Duration
}

// NewKubernetesChecker creates a new Kubernetes checker
func NewKubernetesChecker(namespace string, timeout time.Duration) *KubernetesChecker {
	return &KubernetesChecker{
		namespace: namespace,
		timeout:   timeout,
	}
}

// CheckDeployments checks deployment status
func (k *KubernetesChecker) CheckDeployments() (map[string]interface{}, error) {
	ctx, cancel := context.WithTimeout(context.Background(), k.timeout)
	defer cancel()
	
	cmd := exec.CommandContext(ctx, "kubectl", "get", "deployments", 
		"-n", k.namespace, "-o", "json")
	
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("kubectl get deployments failed: %v", err)
	}
	
	var deployments map[string]interface{}
	if err := json.Unmarshal(output, &deployments); err != nil {
		return nil, err
	}
	
	items := deployments["items"].([]interface{})
	healthy := 0
	
	for _, item := range items {
		deployment := item.(map[string]interface{})
		status := deployment["status"].(map[string]interface{})
		
		replicas, _ := status["replicas"].(float64)
		readyReplicas, _ := status["readyReplicas"].(float64)
		
		if replicas == readyReplicas {
			healthy++
		}
	}
	
	metadata := map[string]interface{}{
		"total_deployments":   len(items),
		"healthy_deployments": healthy,
		"namespace":           k.namespace,
	}
	
	return metadata, nil
}