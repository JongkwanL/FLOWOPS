"""Hyperparameter optimization using Optuna with MLflow integration."""

import os
import json
import yaml
import argparse
import optuna
import mlflow
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Performs hyperparameter optimization with Optuna and MLflow tracking."""
    
    def __init__(self, params_path: str = "params.yaml"):
        """Initialize tuner with configuration."""
        self.params = self._load_params(params_path)
        self.best_score = -float('inf')
        self.best_params = {}
        
    def _load_params(self, params_path: str) -> Dict[str, Any]:
        """Load configuration parameters."""
        with open(params_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training data for optimization."""
        # In production, load from DVC-tracked files
        if os.path.exists("data/features/X_train.npy"):
            X = np.load("data/features/X_train.npy")
            y = np.load("data/features/y_train.npy")
        else:
            # Placeholder data for testing
            X = np.random.randn(1000, 20)
            y = np.random.randint(0, 2, 1000)
        
        return X, y
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for XGBoost hyperparameter optimization."""
        # Start MLflow run for this trial
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            # Suggest hyperparameters
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            }
            
            # Additional fixed parameters
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'use_label_encoder': False,
                'verbosity': 0
            })
            
            # Log parameters to MLflow
            mlflow.log_params(params)
            
            # Load data
            X, y = self.load_data()
            
            # Create model
            model = xgb.XGBClassifier(**params)
            
            # Cross-validation
            cv_config = self.params['train'].get('cross_validation', {})
            scoring = self.params['train']['optimization'].get('metric', 'roc_auc')
            
            cv_scores = cross_val_score(
                model, X, y,
                cv=StratifiedKFold(n_splits=cv_config.get('folds', 5), shuffle=True, random_state=42),
                scoring=scoring,
                n_jobs=-1
            )
            
            score = cv_scores.mean()
            std = cv_scores.std()
            
            # Log metrics to MLflow
            mlflow.log_metric('cv_score_mean', score)
            mlflow.log_metric('cv_score_std', std)
            mlflow.log_metric('trial_number', trial.number)
            
            # Update best params if this is the best score
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            
            logger.info(f"Trial {trial.number}: Score = {score:.4f} (+/- {std:.4f})")
            
            return score
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: int = 3600,
        study_name: str = "xgboost_optimization"
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        # Create Optuna study
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        # Log study info to MLflow
        mlflow.set_tag("optuna_study_name", study_name)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("timeout", timeout)
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=1,  # Use 1 job to avoid MLflow conflicts
            show_progress_bar=True
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best score: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Log best results to MLflow
        mlflow.log_metric("best_cv_score", best_value)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        # Save optimization history
        history = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                history.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                    'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
                })
        
        # Save results
        results = {
            'best_params': best_params,
            'best_score': best_value,
            'n_trials': len(study.trials),
            'study_name': study_name,
            'optimization_history': history
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save optimization results to artifacts."""
        os.makedirs("artifacts", exist_ok=True)
        
        # Save best parameters
        best_params_path = "artifacts/best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump(results['best_params'], f, indent=2)
        mlflow.log_artifact(best_params_path)
        
        # Save full results
        results_path = "artifacts/optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(results_path)
        
        logger.info(f"Results saved to {results_path}")
    
    def create_importance_plot(self, study: optuna.Study):
        """Create and save parameter importance visualization."""
        try:
            import plotly.graph_objects as go
            from optuna.visualization import plot_param_importances
            
            # Create importance plot
            fig = plot_param_importances(study)
            
            # Save as HTML
            plot_path = "artifacts/param_importance.html"
            fig.write_html(plot_path)
            mlflow.log_artifact(plot_path)
            
            logger.info(f"Parameter importance plot saved to {plot_path}")
        except ImportError:
            logger.warning("Plotly not installed, skipping visualization")
    
    def run(
        self,
        experiment_name: str = "hyperparameter-tuning",
        n_trials: int = 50,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """Execute hyperparameter optimization pipeline."""
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="optuna_optimization"):
            mlflow.set_tag("optimization_framework", "optuna")
            mlflow.set_tag("model_type", "xgboost")
            
            # Run optimization
            results = self.optimize(n_trials=n_trials, timeout=timeout)
            
            # Save results
            self.save_results(results)
            
            # Log summary
            logger.info("=" * 50)
            logger.info("Hyperparameter Optimization Complete")
            logger.info(f"Best Score: {results['best_score']:.4f}")
            logger.info(f"Best Parameters: {results['best_params']}")
            logger.info("=" * 50)
            
            return results


def main():
    """Main entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--experiment-name", default="hyperparameter-tuning", help="MLflow experiment name")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Optimization timeout in seconds")
    
    args = parser.parse_args()
    
    # Run optimization
    tuner = HyperparameterTuner()
    results = tuner.run(
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        timeout=args.timeout
    )
    
    # Output best params for downstream use
    print(json.dumps(results['best_params'], indent=2))


if __name__ == "__main__":
    main()