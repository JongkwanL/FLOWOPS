package gateway

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"github.com/gin-gonic/gin"
)

// ProxyTarget represents a proxy target configuration
type ProxyTarget struct {
	Name    string
	URL     *url.URL
	Healthy bool
	Weight  int
}

// LoadBalancer handles load balancing across multiple targets
type LoadBalancer struct {
	targets   []*ProxyTarget
	current   int
	algorithm string // "round-robin", "weighted", "least-connections"
}

// NewLoadBalancer creates a new load balancer
func NewLoadBalancer(algorithm string) *LoadBalancer {
	return &LoadBalancer{
		targets:   make([]*ProxyTarget, 0),
		current:   0,
		algorithm: algorithm,
	}
}

// AddTarget adds a new proxy target
func (lb *LoadBalancer) AddTarget(name, targetURL string, weight int) error {
	u, err := url.Parse(targetURL)
	if err != nil {
		return err
	}
	
	target := &ProxyTarget{
		Name:    name,
		URL:     u,
		Healthy: true,
		Weight:  weight,
	}
	
	lb.targets = append(lb.targets, target)
	return nil
}

// GetNextTarget returns the next available target based on the load balancing algorithm
func (lb *LoadBalancer) GetNextTarget() *ProxyTarget {
	if len(lb.targets) == 0 {
		return nil
	}
	
	// Filter healthy targets
	healthyTargets := make([]*ProxyTarget, 0)
	for _, target := range lb.targets {
		if target.Healthy {
			healthyTargets = append(healthyTargets, target)
		}
	}
	
	if len(healthyTargets) == 0 {
		return nil
	}
	
	switch lb.algorithm {
	case "weighted":
		return lb.getWeightedTarget(healthyTargets)
	case "round-robin":
		fallthrough
	default:
		return lb.getRoundRobinTarget(healthyTargets)
	}
}

// getRoundRobinTarget implements round-robin load balancing
func (lb *LoadBalancer) getRoundRobinTarget(targets []*ProxyTarget) *ProxyTarget {
	target := targets[lb.current%len(targets)]
	lb.current++
	return target
}

// getWeightedTarget implements weighted load balancing
func (lb *LoadBalancer) getWeightedTarget(targets []*ProxyTarget) *ProxyTarget {
	totalWeight := 0
	for _, target := range targets {
		totalWeight += target.Weight
	}
	
	if totalWeight == 0 {
		return lb.getRoundRobinTarget(targets)
	}
	
	// Simple weighted selection (can be improved with better algorithms)
	current := lb.current % totalWeight
	lb.current++
	
	for _, target := range targets {
		current -= target.Weight
		if current < 0 {
			return target
		}
	}
	
	return targets[0]
}

// proxyMLflow proxies requests to MLflow server
func (s *Server) proxyMLflow(c *gin.Context) {
	target := &url.URL{
		Scheme: "http",
		Host:   "localhost:5000",
	}
	
	s.proxy(c, target, "mlflow")
}

// proxyModelServing proxies requests to model serving endpoints
func (s *Server) proxyModelServing(c *gin.Context) {
	target := &url.URL{
		Scheme: "http",
		Host:   "localhost:8000",
	}
	
	s.proxy(c, target, "model-serving")
}

// proxy handles the actual proxying logic
func (s *Server) proxy(c *gin.Context, target *url.URL, serviceName string) {
	start := time.Now()
	
	// Create reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(target)
	
	// Custom director to modify the request
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		
		// Modify the request as needed
		req.Header.Set("X-Forwarded-Host", req.Header.Get("Host"))
		req.Header.Set("X-Origin-Host", target.Host)
		req.Header.Set("X-Gateway", "FlowOps-Gateway")
		
		// Add request ID if available
		if requestID, exists := c.Get("request_id"); exists {
			req.Header.Set("X-Request-ID", requestID.(string))
		}
		
		// Remove the API prefix from the path
		path := c.Param("path")
		if path != "" {
			req.URL.Path = path
		}
	}
	
	// Custom error handler
	proxy.ErrorHandler = func(w http.ResponseWriter, req *http.Request, err error) {
		// Record upstream error metrics
		if s.metrics != nil {
			s.metrics.ErrorsTotal.WithLabelValues(
				req.Method,
				req.URL.Path,
				"502",
			).Inc()
		}
		
		w.WriteHeader(http.StatusBadGateway)
		w.Write([]byte(fmt.Sprintf("Upstream service unavailable: %v", err)))
	}
	
	// Custom response modifier
	proxy.ModifyResponse = func(resp *http.Response) error {
		// Record upstream latency
		if s.metrics != nil {
			duration := time.Since(start)
			s.metrics.UpstreamLatency.WithLabelValues(
				serviceName,
				resp.Request.URL.Path,
			).Observe(duration.Seconds())
		}
		
		// Add response headers
		resp.Header.Set("X-Gateway", "FlowOps-Gateway")
		resp.Header.Set("X-Upstream-Service", serviceName)
		
		return nil
	}
	
	// Serve the proxy request
	proxy.ServeHTTP(c.Writer, c.Request)
}

// CircuitBreaker implements circuit breaker pattern for upstream services
type CircuitBreaker struct {
	maxFailures     int
	resetTimeout    time.Duration
	failureCount    int
	lastFailureTime time.Time
	state           string // "closed", "open", "half-open"
}

// NewCircuitBreaker creates a new circuit breaker
func NewCircuitBreaker(maxFailures int, resetTimeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		maxFailures:  maxFailures,
		resetTimeout: resetTimeout,
		state:        "closed",
	}
}

// Call executes a function with circuit breaker protection
func (cb *CircuitBreaker) Call(fn func() error) error {
	if cb.state == "open" {
		if time.Since(cb.lastFailureTime) > cb.resetTimeout {
			cb.state = "half-open"
			cb.failureCount = 0
		} else {
			return fmt.Errorf("circuit breaker is open")
		}
	}
	
	err := fn()
	if err != nil {
		cb.failureCount++
		cb.lastFailureTime = time.Now()
		
		if cb.failureCount >= cb.maxFailures {
			cb.state = "open"
		}
		return err
	}
	
	// Success - reset circuit breaker
	cb.failureCount = 0
	cb.state = "closed"
	return nil
}

// RetryableHTTPClient implements HTTP client with retry logic
type RetryableHTTPClient struct {
	client       *http.Client
	maxRetries   int
	retryDelay   time.Duration
	backoffFactor float64
}

// NewRetryableHTTPClient creates a new retryable HTTP client
func NewRetryableHTTPClient(maxRetries int, retryDelay time.Duration) *RetryableHTTPClient {
	return &RetryableHTTPClient{
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
		maxRetries:    maxRetries,
		retryDelay:    retryDelay,
		backoffFactor: 2.0,
	}
}

// Do executes an HTTP request with retry logic
func (rc *RetryableHTTPClient) Do(req *http.Request) (*http.Response, error) {
	var lastErr error
	delay := rc.retryDelay
	
	for attempt := 0; attempt <= rc.maxRetries; attempt++ {
		if attempt > 0 {
			time.Sleep(delay)
			delay = time.Duration(float64(delay) * rc.backoffFactor)
		}
		
		// Clone request for retry
		reqClone := rc.cloneRequest(req)
		
		resp, err := rc.client.Do(reqClone)
		if err == nil && resp.StatusCode < 500 {
			return resp, nil
		}
		
		if resp != nil {
			resp.Body.Close()
		}
		lastErr = err
	}
	
	return nil, lastErr
}

// cloneRequest creates a copy of the HTTP request
func (rc *RetryableHTTPClient) cloneRequest(req *http.Request) *http.Request {
	clone := req.Clone(req.Context())
	
	// Clone body if present
	if req.Body != nil {
		body, _ := io.ReadAll(req.Body)
		req.Body = io.NopCloser(bytes.NewReader(body))
		clone.Body = io.NopCloser(bytes.NewReader(body))
	}
	
	return clone
}