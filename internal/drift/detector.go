package drift

import (
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// DriftResult represents the result of drift detection
type DriftResult struct {
	Timestamp    time.Time                  `json:"timestamp"`
	ModelName    string                     `json:"model_name"`
	FeatureName  string                     `json:"feature_name"`
	DriftType    string                     `json:"drift_type"`
	Severity     string                     `json:"severity"`
	Confidence   float64                    `json:"confidence"`
	Method       string                     `json:"method"`
	Statistics   map[string]float64         `json:"statistics"`
	Tests        map[string]StatisticalTest `json:"tests"`
	Alert        bool                       `json:"alert"`
	Message      string                     `json:"message"`
	Metadata     map[string]interface{}     `json:"metadata"`
}

// DriftDetector manages multiple drift detection methods
type DriftDetector struct {
	mu             sync.RWMutex
	referenceData  map[string][]float64 // Reference data for each feature
	adwinDetectors map[string]*ADWIN    // ADWIN detectors for each feature
	config         *DetectorConfig
	results        []DriftResult // Recent drift results
	maxResults     int           // Maximum number of results to keep
}

// DetectorConfig holds configuration for drift detection
type DetectorConfig struct {
	ADWINDelta       float64           `json:"adwin_delta"`
	PSIThreshold     float64           `json:"psi_threshold"`
	KSAlpha          float64           `json:"ks_alpha"`
	PSIBins          int               `json:"psi_bins"`
	MinSampleSize    int               `json:"min_sample_size"`
	AlertThresholds  map[string]float64 `json:"alert_thresholds"`
	EnabledMethods   []string          `json:"enabled_methods"`
	SeverityLevels   map[string]string `json:"severity_levels"`
}

// NewDriftDetector creates a new drift detector
func NewDriftDetector(config *DetectorConfig) *DriftDetector {
	if config == nil {
		config = DefaultDetectorConfig()
	}
	
	return &DriftDetector{
		referenceData:  make(map[string][]float64),
		adwinDetectors: make(map[string]*ADWIN),
		config:         config,
		results:        make([]DriftResult, 0),
		maxResults:     1000,
	}
}

// DefaultDetectorConfig returns default configuration
func DefaultDetectorConfig() *DetectorConfig {
	return &DetectorConfig{
		ADWINDelta:    0.002,
		PSIThreshold:  0.1,
		KSAlpha:       0.05,
		PSIBins:       10,
		MinSampleSize: 30,
		AlertThresholds: map[string]float64{
			"psi":        0.1,
			"ks":         0.05,
			"adwin":      0.5,
			"js":         0.1,
			"wasserstein": 0.1,
		},
		EnabledMethods: []string{"psi", "ks", "adwin", "js"},
		SeverityLevels: map[string]string{
			"low":    "0.0-0.3",
			"medium": "0.3-0.7", 
			"high":   "0.7-1.0",
		},
	}
}

// SetReferenceData sets reference data for a feature
func (d *DriftDetector) SetReferenceData(featureName string, data []float64) error {
	if len(data) < d.config.MinSampleSize {
		return fmt.Errorf("reference data too small: %d < %d", len(data), d.config.MinSampleSize)
	}
	
	d.mu.Lock()
	defer d.mu.Unlock()
	
	// Store reference data
	refCopy := make([]float64, len(data))
	copy(refCopy, data)
	d.referenceData[featureName] = refCopy
	
	// Initialize ADWIN detector
	d.adwinDetectors[featureName] = NewADWIN(d.config.ADWINDelta)
	
	return nil
}

// DetectDrift detects drift for a single feature
func (d *DriftDetector) DetectDrift(modelName, featureName string, currentData []float64) (*DriftResult, error) {
	if len(currentData) < d.config.MinSampleSize {
		return nil, fmt.Errorf("current data too small: %d < %d", len(currentData), d.config.MinSampleSize)
	}
	
	d.mu.RLock()
	referenceData, hasRef := d.referenceData[featureName]
	adwinDetector, hasAdwin := d.adwinDetectors[featureName]
	d.mu.RUnlock()
	
	if !hasRef {
		return nil, fmt.Errorf("no reference data for feature: %s", featureName)
	}
	
	result := &DriftResult{
		Timestamp:   time.Now(),
		ModelName:   modelName,
		FeatureName: featureName,
		Statistics:  make(map[string]float64),
		Tests:       make(map[string]StatisticalTest),
		Metadata:    make(map[string]interface{}),
	}
	
	// Run enabled detection methods
	var maxConfidence float64
	var primaryMethod string
	var alerts []string
	
	for _, method := range d.config.EnabledMethods {
		switch method {
		case "psi":
			psi := PSI(referenceData, currentData, d.config.PSIBins)
			result.Statistics["psi"] = psi
			
			if psi > d.config.AlertThresholds["psi"] {
				alerts = append(alerts, "PSI")
				confidence := math.Min(psi/d.config.AlertThresholds["psi"], 1.0)
				if confidence > maxConfidence {
					maxConfidence = confidence
					primaryMethod = "PSI"
				}
			}
			
		case "ks":
			ksTest := KolmogorovSmirnov(referenceData, currentData, d.config.KSAlpha)
			result.Tests["ks"] = ksTest
			result.Statistics["ks_statistic"] = ksTest.Statistic
			
			if ksTest.Drift {
				alerts = append(alerts, "KS")
				confidence := ksTest.Statistic / ksTest.Critical
				if confidence > maxConfidence {
					maxConfidence = confidence
					primaryMethod = "KS"
				}
			}
			
		case "adwin":
			if hasAdwin {
				// Add current data points to ADWIN
				driftDetected := false
				for _, value := range currentData {
					if adwinDetector.AddElement(value) {
						driftDetected = true
					}
				}
				
				if driftDetected {
					alerts = append(alerts, "ADWIN")
					confidence := adwinDetector.EstimateDistributionChange()
					if confidence > maxConfidence {
						maxConfidence = confidence
						primaryMethod = "ADWIN"
					}
				}
				
				result.Statistics["adwin_width"] = float64(adwinDetector.GetWidth())
				result.Statistics["adwin_mean"] = adwinDetector.GetMean()
				result.Statistics["adwin_variance"] = adwinDetector.GetVariance()
			}
			
		case "js":
			// Create distributions for JS divergence
			refDist := calculateDistribution(referenceData, 
				min(referenceData), max(referenceData), d.config.PSIBins,
				(max(referenceData)-min(referenceData))/float64(d.config.PSIBins))
			curDist := calculateDistribution(currentData, 
				min(referenceData), max(referenceData), d.config.PSIBins,
				(max(referenceData)-min(referenceData))/float64(d.config.PSIBins))
			
			js := JensenShannonDivergence(refDist, curDist)
			result.Statistics["js_divergence"] = js
			
			if js > d.config.AlertThresholds["js"] {
				alerts = append(alerts, "JS")
				confidence := js / d.config.AlertThresholds["js"]
				if confidence > maxConfidence {
					maxConfidence = confidence
					primaryMethod = "JS"
				}
			}
			
		case "wasserstein":
			wd := Wasserstein1Distance(referenceData, currentData)
			result.Statistics["wasserstein_distance"] = wd
			
			if wd > d.config.AlertThresholds["wasserstein"] {
				alerts = append(alerts, "Wasserstein")
				confidence := wd / d.config.AlertThresholds["wasserstein"]
				if confidence > maxConfidence {
					maxConfidence = confidence
					primaryMethod = "Wasserstein"
				}
			}
		}
	}
	
	// Additional statistics
	result.Statistics["mean_difference"] = MeanDifference(referenceData, currentData)
	result.Statistics["variance_difference"] = VarianceDifference(referenceData, currentData)
	result.Statistics["sample_size"] = float64(len(currentData))
	result.Statistics["reference_size"] = float64(len(referenceData))
	
	// Determine final result
	result.Alert = len(alerts) > 0
	result.Confidence = maxConfidence
	result.Method = primaryMethod
	
	if result.Alert {
		result.DriftType = "distribution_shift"
		result.Severity = d.determineSeverity(maxConfidence)
		result.Message = fmt.Sprintf("Drift detected using %s (confidence: %.3f)", 
			primaryMethod, maxConfidence)
		result.Metadata["alert_methods"] = alerts
	} else {
		result.DriftType = "stable"
		result.Severity = "none"
		result.Message = "No significant drift detected"
	}
	
	// Store result
	d.mu.Lock()
	d.results = append(d.results, *result)
	if len(d.results) > d.maxResults {
		d.results = d.results[1:]
	}
	d.mu.Unlock()
	
	return result, nil
}

// DetectBatchDrift detects drift for multiple features
func (d *DriftDetector) DetectBatchDrift(modelName string, currentData map[string][]float64) (map[string]*DriftResult, error) {
	results := make(map[string]*DriftResult)
	
	for featureName, data := range currentData {
		result, err := d.DetectDrift(modelName, featureName, data)
		if err != nil {
			return nil, fmt.Errorf("failed to detect drift for %s: %w", featureName, err)
		}
		results[featureName] = result
	}
	
	return results, nil
}

// GetRecentResults returns recent drift detection results
func (d *DriftDetector) GetRecentResults(limit int) []DriftResult {
	d.mu.RLock()
	defer d.mu.RUnlock()
	
	if limit <= 0 || limit > len(d.results) {
		limit = len(d.results)
	}
	
	results := make([]DriftResult, limit)
	startIdx := len(d.results) - limit
	copy(results, d.results[startIdx:])
	
	return results
}

// GetFeatureStatus returns status for all monitored features
func (d *DriftDetector) GetFeatureStatus() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()
	
	status := make(map[string]interface{})
	
	for featureName := range d.referenceData {
		featureStatus := map[string]interface{}{
			"has_reference_data": true,
			"reference_size":     len(d.referenceData[featureName]),
		}
		
		if adwin, exists := d.adwinDetectors[featureName]; exists {
			featureStatus["adwin_width"] = adwin.GetWidth()
			featureStatus["adwin_mean"] = adwin.GetMean()
			featureStatus["drift_detected"] = adwin.IsDriftDetected()
		}
		
		status[featureName] = featureStatus
	}
	
	return status
}

// determineSeverity determines severity level based on confidence
func (d *DriftDetector) determineSeverity(confidence float64) string {
	switch {
	case confidence >= 0.7:
		return "high"
	case confidence >= 0.3:
		return "medium"
	default:
		return "low"
	}
}

// Helper functions
func min(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	minVal := data[0]
	for _, v := range data[1:] {
		if v < minVal {
			minVal = v
		}
	}
	return minVal
}

func max(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	maxVal := data[0]
	for _, v := range data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

// ToJSON converts drift result to JSON string
func (dr *DriftResult) ToJSON() (string, error) {
	data, err := json.MarshalIndent(dr, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}