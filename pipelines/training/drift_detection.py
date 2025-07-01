"""Model drift detection using statistical methods and MLflow tracking."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy import stats
from scipy.spatial.distance import jensenshannon
import json
import mlflow
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Implements multiple drift detection algorithms based on MLOps best practices."""
    
    def __init__(self, reference_data: Optional[np.ndarray] = None):
        """Initialize drift detector with reference data."""
        self.reference_data = reference_data
        self.reference_stats = self._compute_stats(reference_data) if reference_data is not None else None
        self.drift_history = []
        
        # Thresholds based on codex recommendations
        self.thresholds = {
            'psi': {'warn': 0.1, 'alert': 0.25},
            'ks': {'warn': 0.1, 'alert': 0.2},
            'js_divergence': {'warn': 0.05, 'alert': 0.1},
            'wasserstein': {'warn': 0.1, 'alert': 0.2},
            'performance': {'accuracy_drop': 0.03, 'f1_drop': 0.05},
            'feature_importance': {'spearman_threshold': 0.75}
        }
        
        # ADWIN parameters for online drift detection
        self.adwin_delta = 0.002  # Confidence parameter
        self.adwin_window = []
        self.adwin_variance = 0
        
    def _compute_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute statistical properties of data."""
        if data is None or len(data) == 0:
            return {}
            
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'median': np.median(data, axis=0),
            'quantiles': {
                'q25': np.percentile(data, 25, axis=0),
                'q75': np.percentile(data, 75, axis=0),
                'q95': np.percentile(data, 95, axis=0)
            },
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI) for drift detection."""
        # Create bins based on reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        bins_edges[0] = -np.inf
        bins_edges[-1] = np.inf
        
        # Calculate frequencies
        ref_freq = np.histogram(reference, bins=bins_edges)[0]
        curr_freq = np.histogram(current, bins=bins_edges)[0]
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ref_freq = ref_freq + epsilon
        curr_freq = curr_freq + epsilon
        
        # Normalize to get probabilities
        ref_prob = ref_freq / ref_freq.sum()
        curr_prob = curr_freq / curr_freq.sum()
        
        # Calculate PSI
        psi = np.sum((curr_prob - ref_prob) * np.log(curr_prob / ref_prob))
        
        return float(psi)
    
    def calculate_ks_statistic(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic for continuous distributions."""
        ks_statistic, p_value = stats.ks_2samp(reference, current)
        return float(ks_statistic)
    
    def calculate_js_divergence(self, reference: np.ndarray, current: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence for prediction drift."""
        # Create probability distributions
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        bins_edges = np.linspace(min_val, max_val, bins + 1)
        
        ref_hist = np.histogram(reference, bins=bins_edges, density=True)[0]
        curr_hist = np.histogram(current, bins=bins_edges, density=True)[0]
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        # Calculate JS divergence
        js_div = jensenshannon(ref_hist, curr_hist) ** 2  # Square for JS divergence
        
        return float(js_div)
    
    def calculate_wasserstein_distance(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Wasserstein distance (Earth Mover's Distance)."""
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(reference, current))
    
    def detect_data_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using multiple statistical tests."""
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return {'drift_detected': False, 'reason': 'No reference data'}
        
        drift_results = {
            'drift_detected': False,
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'alerts': []
        }
        
        # Calculate drift metrics for each feature
        n_features = current_data.shape[1] if len(current_data.shape) > 1 else 1
        
        for feature_idx in range(n_features):
            if len(current_data.shape) > 1:
                ref_feature = self.reference_data[:, feature_idx]
                curr_feature = current_data[:, feature_idx]
            else:
                ref_feature = self.reference_data
                curr_feature = current_data
            
            feature_metrics = {
                'psi': self.calculate_psi(ref_feature, curr_feature),
                'ks_statistic': self.calculate_ks_statistic(ref_feature, curr_feature),
                'js_divergence': self.calculate_js_divergence(ref_feature, curr_feature),
                'wasserstein': self.calculate_wasserstein_distance(ref_feature, curr_feature)
            }
            
            drift_results['metrics'][f'feature_{feature_idx}'] = feature_metrics
            
            # Check thresholds
            if feature_metrics['psi'] > self.thresholds['psi']['alert']:
                drift_results['alerts'].append(f"Feature {feature_idx}: PSI alert ({feature_metrics['psi']:.3f})")
                drift_results['drift_detected'] = True
            elif feature_metrics['psi'] > self.thresholds['psi']['warn']:
                drift_results['alerts'].append(f"Feature {feature_idx}: PSI warning ({feature_metrics['psi']:.3f})")
            
            if feature_metrics['ks_statistic'] > self.thresholds['ks']['alert']:
                drift_results['alerts'].append(f"Feature {feature_idx}: KS alert ({feature_metrics['ks_statistic']:.3f})")
                drift_results['drift_detected'] = True
        
        # Log to MLflow
        if mlflow.active_run():
            for feature_key, metrics in drift_results['metrics'].items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"drift_{feature_key}_{metric_name}", value)
            
            if drift_results['drift_detected']:
                mlflow.set_tag("data_drift_detected", "true")
                mlflow.log_text(json.dumps(drift_results['alerts']), "drift_alerts.json")
        
        return drift_results
    
    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Detect drift in model predictions distribution."""
        drift_results = {
            'drift_detected': False,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Calculate JS divergence for predictions
        js_div = self.calculate_js_divergence(reference_predictions, current_predictions)
        drift_results['metrics']['js_divergence'] = js_div
        
        # Calculate prediction statistics shift
        ref_mean = np.mean(reference_predictions)
        curr_mean = np.mean(current_predictions)
        mean_shift = abs(curr_mean - ref_mean)
        drift_results['metrics']['mean_shift'] = float(mean_shift)
        
        # Check calibration (if probabilities)
        if reference_predictions.max() <= 1.0 and reference_predictions.min() >= 0:
            # Expected Calibration Error (ECE)
            ece = self._calculate_ece(current_predictions)
            drift_results['metrics']['ece'] = ece
        
        # Check thresholds
        if js_div > self.thresholds['js_divergence']['alert']:
            drift_results['drift_detected'] = True
            drift_results['alert'] = f"Prediction drift detected: JS divergence = {js_div:.3f}"
        
        # Log to MLflow
        if mlflow.active_run():
            for metric_name, value in drift_results['metrics'].items():
                mlflow.log_metric(f"prediction_drift_{metric_name}", value)
        
        return drift_results
    
    def _calculate_ece(self, predictions: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        # Simplified ECE calculation for binary predictions
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if bin_mask.sum() > 0:
                bin_confidence = predictions[bin_mask].mean()
                # In real implementation, we'd need true labels here
                bin_accuracy = bin_confidence  # Placeholder
                bin_weight = bin_mask.sum() / len(predictions)
                ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return float(ece)
    
    def detect_performance_drift(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Detect performance metric drift."""
        drift_results = {
            'drift_detected': False,
            'timestamp': datetime.now().isoformat(),
            'degradations': []
        }
        
        # Check accuracy drop
        if 'accuracy' in baseline_metrics and 'accuracy' in current_metrics:
            acc_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
            if acc_drop > self.thresholds['performance']['accuracy_drop']:
                drift_results['drift_detected'] = True
                drift_results['degradations'].append(f"Accuracy dropped by {acc_drop:.3f}")
        
        # Check F1 score drop
        if 'f1_score' in baseline_metrics and 'f1_score' in current_metrics:
            f1_drop = baseline_metrics['f1_score'] - current_metrics['f1_score']
            if f1_drop > self.thresholds['performance']['f1_drop']:
                drift_results['drift_detected'] = True
                drift_results['degradations'].append(f"F1 score dropped by {f1_drop:.3f}")
        
        # Log to MLflow
        if mlflow.active_run() and drift_results['drift_detected']:
            mlflow.set_tag("performance_drift_detected", "true")
            for degradation in drift_results['degradations']:
                logger.warning(degradation)
        
        return drift_results
    
    def detect_feature_importance_drift(
        self,
        baseline_importance: Dict[str, float],
        current_importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Detect drift in feature importance rankings."""
        from scipy.stats import spearmanr
        
        drift_results = {
            'drift_detected': False,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Get common features
        common_features = set(baseline_importance.keys()) & set(current_importance.keys())
        
        if len(common_features) < 5:
            logger.warning("Not enough common features for importance drift detection")
            return drift_results
        
        # Calculate rank correlation
        baseline_ranks = [baseline_importance.get(f, 0) for f in common_features]
        current_ranks = [current_importance.get(f, 0) for f in common_features]
        
        correlation, p_value = spearmanr(baseline_ranks, current_ranks)
        drift_results['metrics']['spearman_correlation'] = float(correlation)
        drift_results['metrics']['p_value'] = float(p_value)
        
        # Check threshold
        if correlation < self.thresholds['feature_importance']['spearman_threshold']:
            drift_results['drift_detected'] = True
            drift_results['alert'] = f"Feature importance drift: correlation = {correlation:.3f}"
        
        # Find features with biggest changes
        importance_changes = {}
        for feature in common_features:
            change = abs(current_importance[feature] - baseline_importance[feature])
            importance_changes[feature] = change
        
        # Get top changed features
        top_changes = sorted(importance_changes.items(), key=lambda x: x[1], reverse=True)[:5]
        drift_results['top_changed_features'] = dict(top_changes)
        
        return drift_results
    
    def adwin_add_element(self, value: float) -> bool:
        """Add element to ADWIN window and check for drift."""
        self.adwin_window.append(value)
        
        if len(self.adwin_window) < 2:
            return False
        
        # ADWIN algorithm simplified implementation
        n = len(self.adwin_window)
        
        for split_point in range(1, n):
            # Split window
            w1 = self.adwin_window[:split_point]
            w2 = self.adwin_window[split_point:]
            
            # Calculate means
            mean1 = np.mean(w1)
            mean2 = np.mean(w2)
            
            # Calculate variance
            var1 = np.var(w1)
            var2 = np.var(w2)
            
            # ADWIN test statistic
            m = 1 / (1/len(w1) + 1/len(w2))
            delta_prime = self.adwin_delta / n
            
            epsilon = np.sqrt((var1/len(w1) + var2/len(w2)) * 2 * np.log(2/delta_prime))
            
            if abs(mean1 - mean2) > epsilon:
                # Drift detected - remove old data
                self.adwin_window = self.adwin_window[split_point:]
                logger.info(f"ADWIN drift detected at split point {split_point}")
                return True
        
        # Limit window size
        if len(self.adwin_window) > 1000:
            self.adwin_window = self.adwin_window[-1000:]
        
        return False
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Generate comprehensive drift detection report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'thresholds': self.thresholds,
            'history': self.drift_history[-10:],  # Last 10 drift events
            'recommendations': []
        }
        
        # Add recommendations based on drift history
        if len(self.drift_history) > 0:
            recent_drifts = [d for d in self.drift_history if 
                            datetime.fromisoformat(d['timestamp']) > datetime.now() - timedelta(hours=24)]
            
            if len(recent_drifts) > 3:
                report['recommendations'].append("Frequent drift detected - consider retraining")
            
            if any(d.get('drift_detected') for d in recent_drifts):
                report['recommendations'].append("Recent drift detected - monitor model performance closely")
        
        return report


def main():
    """Example usage of drift detection."""
    # Generate sample data
    np.random.seed(42)
    reference_data = np.random.randn(1000, 10)
    current_data = np.random.randn(500, 10) + 0.5  # Shifted distribution
    
    # Initialize detector
    detector = DriftDetector(reference_data)
    
    # Detect data drift
    data_drift = detector.detect_data_drift(current_data)
    print(f"Data drift detected: {data_drift['drift_detected']}")
    
    # Detect prediction drift
    ref_predictions = np.random.beta(2, 5, 1000)
    curr_predictions = np.random.beta(3, 4, 500)
    pred_drift = detector.detect_prediction_drift(ref_predictions, curr_predictions)
    print(f"Prediction drift detected: {pred_drift['drift_detected']}")
    
    # Generate report
    report = detector.get_drift_report()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()