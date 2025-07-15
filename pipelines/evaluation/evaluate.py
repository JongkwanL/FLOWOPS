"""Model evaluation pipeline with comprehensive metrics and MLflow tracking."""

import os
import json
import argparse
import joblib
import mlflow
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    log_loss, matthews_corrcoef, cohen_kappa_score
)
import shap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with MLflow integration."""
    
    def __init__(self, model_path: Optional[str] = None, run_id: Optional[str] = None):
        """Initialize evaluator with model from file or MLflow run."""
        self.model = None
        self.run_id = run_id
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        elif run_id:
            self.model = self._load_model_from_mlflow(run_id)
            logger.info(f"Loaded model from MLflow run {run_id}")
    
    def _load_model_from_mlflow(self, run_id: str) -> Any:
        """Load model from MLflow run."""
        client = mlflow.tracking.MlflowClient()
        
        # Try different model flavors
        try:
            model_uri = f"runs:/{run_id}/model"
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model from run {run_id}: {e}")
            raise
    
    def load_test_data(self, test_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data for evaluation."""
        if os.path.exists(f"{test_data_path}/X_test.npy"):
            X_test = np.load(f"{test_data_path}/X_test.npy")
            y_test = np.load(f"{test_data_path}/y_test.npy")
        else:
            # Placeholder for testing
            logger.warning(f"Test data not found at {test_data_path}, using dummy data")
            X_test = np.random.randn(300, 20)
            y_test = np.random.randint(0, 2, 300)
        
        return X_test, y_test
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            metrics[f'precision_class_{i}'] = float(p)
            metrics[f'recall_class_{i}'] = float(r)
            metrics[f'f1_class_{i}'] = float(f)
        
        # Additional metrics
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))
        metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                metrics['log_loss'] = float(log_loss(y_true, y_prob))
                
                # Calibration metrics
                metrics['brier_score'] = float(np.mean((y_prob - y_true) ** 2))
                
                # Expected Calibration Error
                metrics['ece'] = self._calculate_ece(y_true, y_prob)
            except Exception as e:
                logger.warning(f"Error calculating probability metrics: {e}")
        
        return metrics
    
    def _calculate_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            
            if bin_mask.sum() > 0:
                bin_confidence = y_prob[bin_mask].mean()
                bin_accuracy = y_true[bin_mask].mean()
                bin_weight = bin_mask.sum() / len(y_prob)
                ece += bin_weight * abs(bin_confidence - bin_accuracy)
        
        return float(ece)
    
    def create_confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "reports/confusion_matrix.png"
    ):
        """Create and save confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        logger.info(f"Confusion matrix saved to {save_path}")
    
    def create_roc_curve_plot(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: str = "reports/roc_curve.png"
    ):
        """Create and save ROC curve visualization."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        logger.info(f"ROC curve saved to {save_path}")
    
    def create_precision_recall_curve_plot(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: str = "reports/precision_recall_curve.png"
    ):
        """Create and save precision-recall curve visualization."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        logger.info(f"Precision-recall curve saved to {save_path}")
    
    def calculate_feature_importance_shap(
        self,
        X_test: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """Calculate SHAP values for feature importance."""
        if not hasattr(self.model, 'predict'):
            logger.warning("Model does not support SHAP analysis")
            return {}
        
        try:
            # Use subset for SHAP calculation
            X_sample = X_test[:max_samples]
            
            # Create SHAP explainer
            if hasattr(self.model, 'tree_method'):  # XGBoost
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.KernelExplainer(self.model.predict, X_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Get feature importance
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            feature_importance = np.abs(shap_values).mean(axis=0)
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            importance_dict = dict(zip(feature_names, feature_importance.tolist()))
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            
            shap_plot_path = "reports/shap_summary.png"
            os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
            plt.savefig(shap_plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            if mlflow.active_run():
                mlflow.log_artifact(shap_plot_path)
            
            return {
                'feature_importance': importance_dict,
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values
            }
            
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {}
    
    def generate_evaluation_report(
        self,
        metrics: Dict[str, float],
        save_path: str = "reports/evaluation_report.html"
    ) -> str:
        """Generate HTML evaluation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric-value {{ font-weight: bold; }}
                .timestamp {{ color: #888; font-size: 14px; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .bad {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <p class="timestamp">Generated: {datetime.now().isoformat()}</p>
            
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
        """
        
        # Add metrics to table
        for metric_name, value in metrics.items():
            if 'class' not in metric_name:  # Skip per-class metrics for main table
                status_class = self._get_metric_status(metric_name, value)
                html_content += f"""
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td class="metric-value">{value:.4f}</td>
                    <td class="{status_class}">{status_class.upper()}</td>
                </tr>
                """
        
        html_content += """
            </table>
            
            <h2>Visualizations</h2>
            <p>
                <img src="confusion_matrix.png" alt="Confusion Matrix" style="max-width: 500px; margin: 10px;">
                <img src="roc_curve.png" alt="ROC Curve" style="max-width: 500px; margin: 10px;">
            </p>
            
            <h2>Feature Importance</h2>
            <p><img src="shap_summary.png" alt="SHAP Summary" style="max-width: 800px;"></p>
            
        </body>
        </html>
        """
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        if mlflow.active_run():
            mlflow.log_artifact(save_path)
        
        logger.info(f"Evaluation report saved to {save_path}")
        return save_path
    
    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Get status classification for metric value."""
        thresholds = {
            'accuracy': {'good': 0.85, 'warning': 0.75},
            'precision': {'good': 0.80, 'warning': 0.70},
            'recall': {'good': 0.80, 'warning': 0.70},
            'f1_score': {'good': 0.80, 'warning': 0.70},
            'roc_auc': {'good': 0.85, 'warning': 0.75},
            'log_loss': {'good': 0.3, 'warning': 0.5, 'reverse': True},
            'ece': {'good': 0.05, 'warning': 0.1, 'reverse': True}
        }
        
        if metric_name in thresholds:
            thresh = thresholds[metric_name]
            reverse = thresh.get('reverse', False)
            
            if reverse:
                if value <= thresh['good']:
                    return 'good'
                elif value <= thresh['warning']:
                    return 'warning'
                else:
                    return 'bad'
            else:
                if value >= thresh['good']:
                    return 'good'
                elif value >= thresh['warning']:
                    return 'warning'
                else:
                    return 'bad'
        
        return 'neutral'
    
    def should_register_model(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Optional[Dict[str, float]] = None,
        min_improvement: float = 0.01
    ) -> bool:
        """Determine if model should be registered based on performance."""
        # Default baseline if not provided
        if baseline_metrics is None:
            baseline_metrics = {
                'accuracy': 0.80,
                'f1_score': 0.75,
                'roc_auc': 0.80
            }
        
        # Check if current model is better
        key_metrics = ['accuracy', 'f1_score', 'roc_auc']
        improvements = []
        
        for metric in key_metrics:
            if metric in current_metrics and metric in baseline_metrics:
                improvement = current_metrics[metric] - baseline_metrics[metric]
                improvements.append(improvement)
                logger.info(f"{metric}: {current_metrics[metric]:.4f} (baseline: {baseline_metrics[metric]:.4f}, "
                          f"improvement: {improvement:+.4f})")
        
        # Register if average improvement exceeds threshold
        avg_improvement = np.mean(improvements) if improvements else 0
        should_register = avg_improvement >= min_improvement
        
        logger.info(f"Average improvement: {avg_improvement:+.4f}, Should register: {should_register}")
        return should_register
    
    def run(
        self,
        test_data_path: str = "data/test",
        register_if_better: bool = False,
        metrics_output: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute complete evaluation pipeline."""
        if self.model is None:
            raise ValueError("No model loaded for evaluation")
        
        # Load test data
        X_test, y_test = self.load_test_data(test_data_path)
        logger.info(f"Loaded test data: {X_test.shape}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = None
        
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                y_prob = y_prob[:, 1]  # Binary classification
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob)
        logger.info(f"Evaluation metrics: {metrics}")
        
        # Log metrics to MLflow
        if mlflow.active_run():
            for name, value in metrics.items():
                mlflow.log_metric(f"test_{name}", value)
        
        # Create visualizations
        self.create_confusion_matrix_plot(y_test, y_pred)
        
        if y_prob is not None:
            self.create_roc_curve_plot(y_test, y_prob)
            self.create_precision_recall_curve_plot(y_test, y_prob)
        
        # Calculate feature importance
        shap_analysis = self.calculate_feature_importance_shap(X_test)
        
        if shap_analysis:
            # Save feature importance
            importance_path = "reports/feature_importance.json"
            os.makedirs(os.path.dirname(importance_path), exist_ok=True)
            with open(importance_path, 'w') as f:
                json.dump(shap_analysis.get('feature_importance', {}), f, indent=2)
            
            if mlflow.active_run():
                mlflow.log_artifact(importance_path)
        
        # Generate evaluation report
        self.generate_evaluation_report(metrics)
        
        # Save metrics to file
        if metrics_output:
            os.makedirs(os.path.dirname(metrics_output) if os.path.dirname(metrics_output) else '.', exist_ok=True)
            with open(metrics_output, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Save to DVC metrics
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/eval_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Check if model should be registered
        if register_if_better:
            should_register = self.should_register_model(metrics)
            
            if should_register and self.run_id:
                logger.info(f"Model meets registration criteria, registering from run {self.run_id}")
                # Registration would be handled by MLflow model registry
                mlflow.set_tag("register_model", "true")
        
        return metrics


def main():
    """Main entry point for evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate ML model")
    parser.add_argument("--run-id", help="MLflow run ID to load model from")
    parser.add_argument("--model-path", default="models/model.pkl", help="Path to model file")
    parser.add_argument("--test-data-path", default="data/test", help="Path to test data")
    parser.add_argument("--register-if-better", action="store_true", help="Register model if better than baseline")
    parser.add_argument("--metrics-output", help="Path to save metrics JSON")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path=args.model_path, run_id=args.run_id)
    
    # Run evaluation
    metrics = evaluator.run(
        test_data_path=args.test_data_path,
        register_if_better=args.register_if_better,
        metrics_output=args.metrics_output
    )
    
    # Output key metrics
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"F1 Score: {metrics.get('f1_score', 0):.4f}")
    print(f"ROC AUC: {metrics.get('roc_auc', 0):.4f}")


if __name__ == "__main__":
    main()