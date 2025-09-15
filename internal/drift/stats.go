package drift

import (
	"math"
	"sort"
)

// StatisticalTest represents a statistical test result
type StatisticalTest struct {
	Statistic float64 `json:"statistic"`
	PValue    float64 `json:"p_value"`
	Critical  float64 `json:"critical_value"`
	Drift     bool    `json:"drift_detected"`
	Method    string  `json:"method"`
}

// PSI calculates Population Stability Index between reference and current data
func PSI(reference, current []float64, bins int) float64 {
	if len(reference) == 0 || len(current) == 0 {
		return 0.0
	}
	
	// Create bins based on reference data
	refSorted := make([]float64, len(reference))
	copy(refSorted, reference)
	sort.Float64s(refSorted)
	
	min := refSorted[0]
	max := refSorted[len(refSorted)-1]
	
	// Handle edge case where all values are the same
	if min == max {
		return 0.0
	}
	
	binWidth := (max - min) / float64(bins)
	
	// Calculate distributions
	refDist := calculateDistribution(reference, min, max, bins, binWidth)
	curDist := calculateDistribution(current, min, max, bins, binWidth)
	
	// Calculate PSI
	psi := 0.0
	for i := 0; i < bins; i++ {
		if refDist[i] > 0 && curDist[i] > 0 {
			psi += (curDist[i] - refDist[i]) * math.Log(curDist[i]/refDist[i])
		}
	}
	
	return psi
}

// calculateDistribution creates histogram distribution
func calculateDistribution(data []float64, min, max float64, bins int, binWidth float64) []float64 {
	dist := make([]float64, bins)
	total := float64(len(data))
	
	for _, value := range data {
		binIndex := int((value - min) / binWidth)
		if binIndex >= bins {
			binIndex = bins - 1
		}
		if binIndex < 0 {
			binIndex = 0
		}
		dist[binIndex]++
	}
	
	// Convert to probabilities
	for i := range dist {
		dist[i] = (dist[i] + 1e-8) / total // Add small epsilon to avoid log(0)
	}
	
	return dist
}

// KolmogorovSmirnov performs Kolmogorov-Smirnov test
func KolmogorovSmirnov(reference, current []float64, alpha float64) StatisticalTest {
	if len(reference) == 0 || len(current) == 0 {
		return StatisticalTest{Method: "KS"}
	}
	
	n1, n2 := float64(len(reference)), float64(len(current))
	
	// Sort both samples
	ref := make([]float64, len(reference))
	cur := make([]float64, len(current))
	copy(ref, reference)
	copy(cur, current)
	sort.Float64s(ref)
	sort.Float64s(cur)
	
	// Calculate empirical CDFs and find maximum difference
	maxDiff := 0.0
	i, j := 0, 0
	
	for i < len(ref) && j < len(cur) {
		if ref[i] <= cur[j] {
			cdf1 := float64(i+1) / n1
			cdf2 := float64(j) / n2
			diff := math.Abs(cdf1 - cdf2)
			if diff > maxDiff {
				maxDiff = diff
			}
			i++
		} else {
			cdf1 := float64(i) / n1
			cdf2 := float64(j+1) / n2
			diff := math.Abs(cdf1 - cdf2)
			if diff > maxDiff {
				maxDiff = diff
			}
			j++
		}
	}
	
	// Handle remaining elements
	for i < len(ref) {
		cdf1 := float64(i+1) / n1
		cdf2 := 1.0
		diff := math.Abs(cdf1 - cdf2)
		if diff > maxDiff {
			maxDiff = diff
		}
		i++
	}
	
	for j < len(cur) {
		cdf1 := 1.0
		cdf2 := float64(j+1) / n2
		diff := math.Abs(cdf1 - cdf2)
		if diff > maxDiff {
			maxDiff = diff
		}
		j++
	}
	
	// Calculate critical value
	critical := ksCriticalValue(n1, n2, alpha)
	
	return StatisticalTest{
		Statistic: maxDiff,
		PValue:    ksPValue(maxDiff, n1, n2),
		Critical:  critical,
		Drift:     maxDiff > critical,
		Method:    "KS",
	}
}

// ksCriticalValue calculates critical value for KS test
func ksCriticalValue(n1, n2, alpha float64) float64 {
	// Approximation for large samples
	c := math.Sqrt(-0.5 * math.Log(alpha/2))
	return c * math.Sqrt((n1+n2)/(n1*n2))
}

// ksPValue calculates p-value for KS test (approximation)
func ksPValue(d, n1, n2 float64) float64 {
	// Simplified p-value calculation
	lambda := d * math.Sqrt((n1*n2)/(n1+n2))
	return 2 * math.Exp(-2*lambda*lambda)
}

// JensenShannonDivergence calculates JS divergence between two distributions
func JensenShannonDivergence(p, q []float64) float64 {
	if len(p) != len(q) {
		return 0.0
	}
	
	js := 0.0
	for i := range p {
		if p[i] > 0 && q[i] > 0 {
			m := (p[i] + q[i]) / 2
			if m > 0 {
				js += 0.5 * (p[i]*math.Log(p[i]/m) + q[i]*math.Log(q[i]/m))
			}
		}
	}
	
	return js / math.Log(2) // Convert to base 2
}

// Wasserstein1Distance calculates 1-Wasserstein distance (Earth Mover's Distance)
func Wasserstein1Distance(reference, current []float64) float64 {
	if len(reference) == 0 || len(current) == 0 {
		return 0.0
	}
	
	// Sort both samples
	ref := make([]float64, len(reference))
	cur := make([]float64, len(current))
	copy(ref, reference)
	copy(cur, current)
	sort.Float64s(ref)
	sort.Float64s(cur)
	
	// Calculate empirical CDFs
	distance := 0.0
	i, j := 0, 0
	cdf1, cdf2 := 0.0, 0.0
	n1, n2 := float64(len(ref)), float64(len(cur))
	
	// Merge and calculate
	for i < len(ref) && j < len(cur) {
		if ref[i] <= cur[j] {
			if i > 0 {
				distance += math.Abs(cdf1-cdf2) * (ref[i] - ref[i-1])
			}
			cdf1 = float64(i+1) / n1
			i++
		} else {
			if j > 0 {
				distance += math.Abs(cdf1-cdf2) * (cur[j] - cur[j-1])
			}
			cdf2 = float64(j+1) / n2
			j++
		}
	}
	
	return distance
}

// ChiSquareTest performs chi-square goodness of fit test
func ChiSquareTest(observed, expected []float64, alpha float64) StatisticalTest {
	if len(observed) != len(expected) || len(observed) == 0 {
		return StatisticalTest{Method: "ChiSquare"}
	}
	
	chiSq := 0.0
	for i := range observed {
		if expected[i] > 0 {
			diff := observed[i] - expected[i]
			chiSq += (diff * diff) / expected[i]
		}
	}
	
	// Degrees of freedom
	df := len(observed) - 1
	critical := chiSquareCritical(df, alpha)
	
	return StatisticalTest{
		Statistic: chiSq,
		PValue:    chiSquarePValue(chiSq, df),
		Critical:  critical,
		Drift:     chiSq > critical,
		Method:    "ChiSquare",
	}
}

// chiSquareCritical returns critical value for chi-square test (approximation)
func chiSquareCritical(df int, alpha float64) float64 {
	// Simplified critical value calculation
	if df <= 0 {
		return 0.0
	}
	
	// Approximate critical values for common alpha levels
	switch {
	case alpha >= 0.1:
		return float64(df) + 1.64*math.Sqrt(2*float64(df))
	case alpha >= 0.05:
		return float64(df) + 1.96*math.Sqrt(2*float64(df))
	case alpha >= 0.01:
		return float64(df) + 2.58*math.Sqrt(2*float64(df))
	default:
		return float64(df) + 3.29*math.Sqrt(2*float64(df))
	}
}

// chiSquarePValue calculates p-value for chi-square test (approximation)
func chiSquarePValue(chiSq float64, df int) float64 {
	// Simplified p-value calculation using normal approximation
	if df <= 0 {
		return 1.0
	}
	
	z := (chiSq - float64(df)) / math.Sqrt(2*float64(df))
	return 1.0 - normalCDF(z)
}

// normalCDF calculates normal cumulative distribution function (approximation)
func normalCDF(x float64) float64 {
	return 0.5 * (1.0 + math.Erf(x/math.Sqrt2))
}

// MeanDifference calculates normalized difference in means
func MeanDifference(reference, current []float64) float64 {
	if len(reference) == 0 || len(current) == 0 {
		return 0.0
	}
	
	refMean := mean(reference)
	curMean := mean(current)
	refStd := standardDeviation(reference)
	
	if refStd == 0 {
		return 0.0
	}
	
	return math.Abs(curMean-refMean) / refStd
}

// VarianceDifference calculates normalized difference in variances
func VarianceDifference(reference, current []float64) float64 {
	if len(reference) == 0 || len(current) == 0 {
		return 0.0
	}
	
	refVar := variance(reference)
	curVar := variance(current)
	
	if refVar == 0 {
		return 0.0
	}
	
	return math.Abs(curVar-refVar) / refVar
}

// Helper functions
func mean(data []float64) float64 {
	if len(data) == 0 {
		return 0.0
	}
	
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}

func variance(data []float64) float64 {
	if len(data) <= 1 {
		return 0.0
	}
	
	m := mean(data)
	sum := 0.0
	for _, v := range data {
		diff := v - m
		sum += diff * diff
	}
	return sum / float64(len(data)-1)
}

func standardDeviation(data []float64) float64 {
	return math.Sqrt(variance(data))
}