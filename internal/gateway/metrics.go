package gateway

import (
	"github.com/prometheus/client_golang/prometheus"
)

// Metrics holds all Prometheus metrics for the gateway
type Metrics struct {
	RequestsTotal     *prometheus.CounterVec
	RequestDuration   *prometheus.HistogramVec
	ErrorsTotal       *prometheus.CounterVec
	ActiveConnections prometheus.Gauge
	UpstreamLatency   *prometheus.HistogramVec
	collectors        []prometheus.Collector
}

// NewMetrics creates and registers all metrics
func NewMetrics() *Metrics {
	requestsTotal := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_requests_total",
			Help: "Total number of HTTP requests processed by the gateway",
		},
		[]string{"method", "path", "status"},
	)
	
	requestDuration := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gateway_request_duration_seconds",
			Help:    "Time spent processing HTTP requests",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method", "path"},
	)
	
	errorsTotal := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "gateway_errors_total",
			Help: "Total number of HTTP errors",
		},
		[]string{"method", "path", "status"},
	)
	
	activeConnections := prometheus.NewGauge(
		prometheus.GaugeOpts{
			Name: "gateway_active_connections",
			Help: "Number of active connections to the gateway",
		},
	)
	
	upstreamLatency := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "gateway_upstream_duration_seconds",
			Help:    "Time spent waiting for upstream services",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"service", "endpoint"},
	)
	
	metrics := &Metrics{
		RequestsTotal:     requestsTotal,
		RequestDuration:   requestDuration,
		ErrorsTotal:       errorsTotal,
		ActiveConnections: activeConnections,
		UpstreamLatency:   upstreamLatency,
		collectors: []prometheus.Collector{
			requestsTotal,
			requestDuration,
			errorsTotal,
			activeConnections,
			upstreamLatency,
		},
	}
	
	return metrics
}

// GetCollectors returns all collectors for registration
func (m *Metrics) GetCollectors() []prometheus.Collector {
	return m.collectors
}