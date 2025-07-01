"""Main training pipeline with MLflow integration and drift detection."""

import os
import json
import yaml
import argparse
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training with MLflow tracking and experiment management."""
    
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize trainer with configuration parameters."""
        self.params = self._load_params(params_path)
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
    def _load_params(self, params_path: str) -> Dict[str, Any]:
        """Load training parameters from YAML file."""
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed features and labels."""
        # In production, this would load from DVC-tracked files
        logger.info("Loading features from data/features")
        
        # Placeholder - replace with actual data loading
        X_train = np.load("data/features/X_train.npy") if os.path.exists("data/features/X_train.npy") else np.random.randn(1000, 20)
        y_train = np.load("data/features/y_train.npy") if os.path.exists("data/features/y_train.npy") else np.random.randint(0, 2, 1000)
        X_val = np.load("data/features/X_val.npy") if os.path.exists("data/features/X_val.npy") else np.random.randn(200, 20)
        y_val = np.load("data/features/y_val.npy") if os.path.exists("data/features/y_val.npy") else np.random.randint(0, 2, 200)
        
        return X_train, y_train, X_val, y_val
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> xgb.XGBClassifier:
        """Train XGBoost model with early stopping and cross-validation."""
        # Use provided hyperparams or defaults from config
        if hyperparams is None:
            hyperparams = self.params['train']['hyperparameters']
        
        logger.info(f"Training XGBoost with params: {hyperparams}")
        
        # Initialize model
        model = xgb.XGBClassifier(
            learning_rate=hyperparams.get('learning_rate', 0.1),
            max_depth=hyperparams.get('max_depth', 6),
            n_estimators=hyperparams.get('n_estimators', 100),
            subsample=hyperparams.get('subsample', 0.8),
            colsample_bytree=hyperparams.get('colsample_bytree', 0.8),
            objective=hyperparams.get('objective', 'binary:logistic'),
            eval_metric=hyperparams.get('eval_metric', 'auc'),
            random_state=42,
            use_label_encoder=False
        )
        
        # Setup early stopping if enabled
        early_stopping = self.params['train'].get('early_stopping', {})
        if early_stopping.get('enabled', True):
            eval_set = [(X_val, y_val)]
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping.get('patience', 10),
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        # Cross-validation if enabled
        cv_config = self.params['train'].get('cross_validation', {})
        if cv_config.get('enabled', True):
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_config.get('folds', 5)),
                scoring=cv_config.get('scoring', 'roc_auc')
            )
            logger.info(f"Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            mlflow.log_metric("cv_score_mean", cv_scores.mean())
            mlflow.log_metric("cv_score_std", cv_scores.std())
        
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        prefix: str = ""
    ) -> Dict[str, float]:
        """Evaluate model performance and return metrics."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
        
        try:
            auc_roc = roc_auc_score(y, y_prob)
        except:
            auc_roc = 0.0
        
        metrics = {
            f"{prefix}accuracy": float(accuracy),
            f"{prefix}precision": float(precision),
            f"{prefix}recall": float(recall),
            f"{prefix}f1_score": float(f1),
            f"{prefix}auc_roc": float(auc_roc)
        }
        
        # Log to MLflow
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        return metrics
    
    def calculate_feature_importance(self, model: Any, feature_names: Optional[list] = None) -> Dict[str, float]:
        """Calculate and log feature importance."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
            
            importance_dict = dict(zip(feature_names, importances.tolist()))
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            # Log top features to MLflow
            for i, (feature, importance) in enumerate(list(importance_dict.items())[:10]):
                mlflow.log_metric(f"feature_importance_{i}_{feature}", importance)
            
            return importance_dict
        
        return {}
    
    def save_model(self, model: Any, path: str = "models/model.pkl"):
        """Save model to disk and log to MLflow."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save with joblib
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
        
        # Log to MLflow
        if isinstance(model, xgb.XGBClassifier):
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        # Save metadata
        metadata = {
            "model_type": type(model).__name__,
            "training_date": datetime.now().isoformat(),
            "params": self.params['train'],
            "framework_version": xgb.__version__ if isinstance(model, xgb.XGBClassifier) else "sklearn"
        }
        
        metadata_path = path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        mlflow.log_artifact(metadata_path)
    
    def detect_training_drift(self, current_metrics: Dict[str, float]) -> bool:
        """Detect if model performance has drifted from baseline."""
        # Get baseline metrics from previous production model
        try:
            # In production, fetch from model registry
            baseline_metrics = {
                "accuracy": 0.85,
                "f1_score": 0.80,
                "auc_roc": 0.88
            }
            
            # Check for significant degradation
            accuracy_drop = baseline_metrics["accuracy"] - current_metrics.get("val_accuracy", 0)
            f1_drop = baseline_metrics["f1_score"] - current_metrics.get("val_f1_score", 0)
            
            drift_detected = accuracy_drop > 0.05 or f1_drop > 0.05
            
            if drift_detected:
                logger.warning(f"Performance drift detected! Accuracy drop: {accuracy_drop:.4f}, F1 drop: {f1_drop:.4f}")
                mlflow.set_tag("drift_detected", "true")
                mlflow.log_metric("accuracy_drop", accuracy_drop)
                mlflow.log_metric("f1_drop", f1_drop)
            
            return drift_detected
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return False
    
    def run(
        self,
        experiment_name: str = "default-experiment",
        run_name: Optional[str] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        log_model: bool = True
    ) -> str:
        """Execute the complete training pipeline."""
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params(self.params['train']['hyperparameters'])
            mlflow.set_tag("training_framework", "xgboost")
            mlflow.set_tag("pipeline_version", "1.0.0")
            
            # Load data
            X_train, y_train, X_val, y_val = self.load_features()
            logger.info(f"Loaded training data: {X_train.shape}, validation data: {X_val.shape}")
            
            # Log data statistics
            mlflow.log_metric("n_train_samples", len(X_train))
            mlflow.log_metric("n_val_samples", len(X_val))
            mlflow.log_metric("n_features", X_train.shape[1])
            
            # Train model
            model = self.train_xgboost(X_train, y_train, X_val, y_val, hyperparams)
            
            # Evaluate on training and validation sets
            train_metrics = self.evaluate_model(model, X_train, y_train, prefix="train_")
            val_metrics = self.evaluate_model(model, X_val, y_val, prefix="val_")
            
            logger.info(f"Training metrics: {train_metrics}")
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Calculate feature importance
            feature_importance = self.calculate_feature_importance(model)
            
            # Save feature importance as artifact
            importance_path = "artifacts/feature_importance.json"
            os.makedirs(os.path.dirname(importance_path), exist_ok=True)
            with open(importance_path, 'w') as f:
                json.dump(feature_importance, f, indent=2)
            mlflow.log_artifact(importance_path)
            
            # Check for drift
            all_metrics = {**train_metrics, **val_metrics}
            drift_detected = self.detect_training_drift(all_metrics)
            
            # Save model
            if log_model:
                self.save_model(model)
            
            # Save metrics to file for DVC
            os.makedirs("metrics", exist_ok=True)
            with open("metrics/train_metrics.json", 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            logger.info(f"Training completed. Run ID: {run.info.run_id}")
            
            return run.info.run_id


def main():
    """Main entry point for training pipeline."""
    parser = argparse.ArgumentParser(description="Train ML model with MLflow tracking")
    parser.add_argument("--experiment-name", default="default-experiment", help="MLflow experiment name")
    parser.add_argument("--run-name", default=None, help="MLflow run name")
    parser.add_argument("--use-best-params", action="store_true", help="Use best hyperparameters from tuning")
    parser.add_argument("--log-model", action="store_true", help="Log model to MLflow")
    
    args = parser.parse_args()
    
    # Load best params if requested
    hyperparams = None
    if args.use_best_params and os.path.exists("artifacts/best_params.json"):
        with open("artifacts/best_params.json", 'r') as f:
            hyperparams = json.load(f)
        logger.info(f"Using best hyperparameters: {hyperparams}")
    
    # Run training
    trainer = ModelTrainer()
    run_id = trainer.run(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        hyperparams=hyperparams,
        log_model=args.log_model
    )
    
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()