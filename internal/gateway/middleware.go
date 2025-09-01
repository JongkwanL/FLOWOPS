package gateway

import (
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/time/rate"
)

// MiddlewareManager manages all middleware components
type MiddlewareManager struct {
	config  *Config
	metrics *Metrics
	limiter *rate.Limiter
}

// NewMiddlewareManager creates a new middleware manager
func NewMiddlewareManager(config *Config, metrics *Metrics) *MiddlewareManager {
	var limiter *rate.Limiter
	if config.EnableRateLimit {
		limiter = rate.NewLimiter(rate.Limit(config.RateLimitRPS), config.RateLimitRPS*2)
	}
	
	return &MiddlewareManager{
		config:  config,
		metrics: metrics,
		limiter: limiter,
	}
}

// Logger returns a logging middleware
func (m *MiddlewareManager) Logger() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
			param.ClientIP,
			param.TimeStamp.Format(time.RFC1123),
			param.Method,
			param.Path,
			param.Request.Proto,
			param.StatusCode,
			param.Latency,
			param.Request.UserAgent(),
			param.ErrorMessage,
		)
	})
}

// CORS returns a CORS middleware
func (m *MiddlewareManager) CORS() gin.HandlerFunc {
	config := cors.DefaultConfig()
	config.AllowOrigins = []string{"*"}
	config.AllowMethods = []string{"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
	config.AllowHeaders = []string{"Origin", "Content-Length", "Content-Type", "Authorization"}
	config.AllowCredentials = true
	
	return cors.New(config)
}

// RateLimit returns a rate limiting middleware
func (m *MiddlewareManager) RateLimit() gin.HandlerFunc {
	return func(c *gin.Context) {
		if m.limiter != nil && !m.limiter.Allow() {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "rate limit exceeded",
			})
			c.Abort()
			return
		}
		c.Next()
	}
}

// Metrics returns a metrics collection middleware
func (m *MiddlewareManager) Metrics() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		c.Next()
		
		duration := time.Since(start)
		status := c.Writer.Status()
		method := c.Request.Method
		path := c.FullPath()
		
		// Record metrics
		m.metrics.RequestDuration.WithLabelValues(method, path).Observe(duration.Seconds())
		m.metrics.RequestsTotal.WithLabelValues(method, path, fmt.Sprintf("%d", status)).Inc()
		
		// Record error if status is >= 400
		if status >= 400 {
			m.metrics.ErrorsTotal.WithLabelValues(method, path, fmt.Sprintf("%d", status)).Inc()
		}
	}
}

// Security returns a security headers middleware
func (m *MiddlewareManager) Security() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Security headers
		c.Header("X-Content-Type-Options", "nosniff")
		c.Header("X-Frame-Options", "DENY")
		c.Header("X-XSS-Protection", "1; mode=block")
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		c.Header("Content-Security-Policy", "default-src 'self'")
		c.Header("Referrer-Policy", "strict-origin-when-cross-origin")
		
		c.Next()
	}
}

// Auth returns an authentication middleware
func (m *MiddlewareManager) Auth() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Skip authentication for health checks
		if c.Request.URL.Path == "/health" || c.Request.URL.Path == "/ready" {
			c.Next()
			return
		}
		
		// Get authorization header
		authHeader := c.GetHeader("Authorization")
		if authHeader == "" {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "missing authorization header",
			})
			c.Abort()
			return
		}
		
		// Check if it's a Bearer token
		if !strings.HasPrefix(authHeader, "Bearer ") {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "invalid authorization header format",
			})
			c.Abort()
			return
		}
		
		// Extract token
		tokenString := strings.TrimPrefix(authHeader, "Bearer ")
		
		// Validate JWT token
		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			// Validate signing method
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
			}
			return []byte(m.config.JWTSecret), nil
		})
		
		if err != nil || !token.Valid {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "invalid token",
			})
			c.Abort()
			return
		}
		
		// Extract claims
		if claims, ok := token.Claims.(jwt.MapClaims); ok && token.Valid {
			c.Set("user", claims["sub"])
			c.Set("roles", claims["roles"])
		}
		
		c.Next()
	}
}

// RequestID returns a request ID middleware
func (m *MiddlewareManager) RequestID() gin.HandlerFunc {
	return func(c *gin.Context) {
		requestID := c.GetHeader("X-Request-ID")
		if requestID == "" {
			requestID = generateRequestID()
		}
		
		c.Header("X-Request-ID", requestID)
		c.Set("request_id", requestID)
		c.Next()
	}
}

// generateRequestID generates a unique request ID
func generateRequestID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// Recovery returns a custom recovery middleware
func (m *MiddlewareManager) Recovery() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		log.Printf("Panic recovered: %v", recovered)
		
		if m.metrics != nil {
			m.metrics.ErrorsTotal.WithLabelValues(
				c.Request.Method,
				c.FullPath(),
				"500",
			).Inc()
		}
		
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "internal server error",
		})
	})
}