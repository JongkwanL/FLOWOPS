package drift

import (
	"container/list"
	"math"
)

// ADWIN implements Adaptive Windowing for drift detection
type ADWIN struct {
	delta         float64     // Confidence parameter (0 < delta < 1)
	window        *list.List  // Sliding window of data points
	buckets       []Bucket    // Exponential histogram buckets
	total         float64     // Sum of all elements in window
	variance      float64     // Variance of elements in window
	width         int         // Current window width
	lastBucketRow int         // Last bucket row index
	driftDetected bool        // Whether drift was detected
	changePoint   int         // Position where change was detected
}

// Bucket represents a bucket in the exponential histogram
type Bucket struct {
	size     int     // Number of elements in bucket
	sum      float64 // Sum of elements in bucket
	variance float64 // Variance of elements in bucket
}

// NewADWIN creates a new ADWIN detector
func NewADWIN(delta float64) *ADWIN {
	if delta <= 0 || delta >= 1 {
		delta = 0.002 // Default value
	}
	
	return &ADWIN{
		delta:   delta,
		window:  list.New(),
		buckets: make([]Bucket, 0),
		width:   0,
	}
}

// AddElement adds a new element to the detector and checks for drift
func (a *ADWIN) AddElement(value float64) bool {
	// Add element to window
	a.window.PushBack(value)
	a.width++
	a.total += value
	
	// Update variance incrementally
	if a.width > 1 {
		oldMean := (a.total - value) / float64(a.width-1)
		newMean := a.total / float64(a.width)
		
		a.variance = ((float64(a.width-2) * a.variance) + 
			(value-oldMean)*(value-newMean)) / float64(a.width-1)
	}
	
	// Compress old elements into buckets
	a.compressBuckets()
	
	// Check for drift
	driftDetected := a.detectChange()
	
	if driftDetected {
		a.driftDetected = true
	}
	
	return driftDetected
}

// compressBuckets manages the exponential histogram
func (a *ADWIN) compressBuckets() {
	// This is a simplified version of the exponential histogram
	// In practice, this would be more complex with proper bucket management
	
	if a.width%100 == 0 && a.width > 100 {
		// Compress older elements into a bucket
		oldElements := 0
		oldSum := 0.0
		
		for i := 0; i < 50 && a.window.Len() > 0; i++ {
			elem := a.window.Remove(a.window.Front()).(float64)
			oldSum += elem
			oldElements++
			a.width--
		}
		
		if oldElements > 0 {
			bucket := Bucket{
				size: oldElements,
				sum:  oldSum,
			}
			a.buckets = append(a.buckets, bucket)
		}
	}
}

// detectChange implements the ADWIN change detection algorithm
func (a *ADWIN) detectChange() bool {
	if a.width < 10 {
		return false
	}
	
	// Convert window to slice for easier processing
	windowData := make([]float64, 0, a.window.Len())
	for e := a.window.Front(); e != nil; e = e.Next() {
		windowData = append(windowData, e.Value.(float64))
	}
	
	n := len(windowData)
	
	// Check all possible splits
	for i := 5; i < n-5; i++ {
		left := windowData[:i]
		right := windowData[i:]
		
		if a.hasSignificantChange(left, right) {
			// Remove elements from the beginning up to the change point
			for j := 0; j < i; j++ {
				if a.window.Len() > 0 {
					removed := a.window.Remove(a.window.Front()).(float64)
					a.total -= removed
					a.width--
				}
			}
			
			a.changePoint = i
			return true
		}
	}
	
	return false
}

// hasSignificantChange checks if there's a significant change between two sub-windows
func (a *ADWIN) hasSignificantChange(left, right []float64) bool {
	if len(left) < 2 || len(right) < 2 {
		return false
	}
	
	n0, n1 := float64(len(left)), float64(len(right))
	mean0, mean1 := mean(left), mean(right)
	
	// Calculate the bound for significant change
	m := 1.0 / (1.0/n0 + 1.0/n1)
	
	// Hoeffding bound calculation
	delta := a.delta / float64(a.width)
	bound := math.Sqrt((2.0 / m) * math.Log(2.0/delta))
	
	// Check if difference in means exceeds the bound
	meanDiff := math.Abs(mean0 - mean1)
	
	return meanDiff > bound
}

// GetWidth returns the current window width
func (a *ADWIN) GetWidth() int {
	return a.width
}

// GetTotal returns the sum of elements in the current window
func (a *ADWIN) GetTotal() float64 {
	return a.total
}

// GetMean returns the mean of elements in the current window
func (a *ADWIN) GetMean() float64 {
	if a.width == 0 {
		return 0.0
	}
	return a.total / float64(a.width)
}

// GetVariance returns the variance of elements in the current window
func (a *ADWIN) GetVariance() float64 {
	return a.variance
}

// IsDriftDetected returns whether drift was detected
func (a *ADWIN) IsDriftDetected() bool {
	return a.driftDetected
}

// GetChangePoint returns the position where change was detected
func (a *ADWIN) GetChangePoint() int {
	return a.changePoint
}

// Reset resets the detector to initial state
func (a *ADWIN) Reset() {
	a.window.Init()
	a.buckets = a.buckets[:0]
	a.total = 0
	a.variance = 0
	a.width = 0
	a.driftDetected = false
	a.changePoint = 0
}

// GetWindowData returns a copy of the current window data
func (a *ADWIN) GetWindowData() []float64 {
	data := make([]float64, 0, a.window.Len())
	for e := a.window.Front(); e != nil; e = e.Next() {
		data = append(data, e.Value.(float64))
	}
	return data
}

// EstimateDistributionChange estimates the magnitude of distribution change
func (a *ADWIN) EstimateDistributionChange() float64 {
	if a.width < 20 {
		return 0.0
	}
	
	// Split window into two halves and compare
	windowData := a.GetWindowData()
	mid := len(windowData) / 2
	left := windowData[:mid]
	right := windowData[mid:]
	
	// Calculate KS statistic between the two halves
	ks := KolmogorovSmirnov(left, right, 0.05)
	return ks.Statistic
}

// GetBucketInfo returns information about the current buckets
func (a *ADWIN) GetBucketInfo() []Bucket {
	bucketsCopy := make([]Bucket, len(a.buckets))
	copy(bucketsCopy, a.buckets)
	return bucketsCopy
}