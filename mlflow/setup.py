"""MLflow setup and configuration module."""

import os
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowManager:
    """Manages MLflow experiments and model registry."""
    
    def __init__(self, config_path: str = "mlflow/experiments/config.yaml"):
        """Initialize MLflow manager with configuration."""
        self.config = self._load_config(config_path)
        self.client = MlflowClient()
        self._setup_tracking()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load MLflow configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_tracking(self):
        """Setup MLflow tracking configuration."""
        tracking_config = self.config.get('tracking', {})
        
        # Enable autologging
        autolog = tracking_config.get('autolog', {})
        if autolog.get('sklearn'):
            try:
                import mlflow.sklearn
                mlflow.sklearn.autolog()
            except ImportError:
                logger.warning("scikit-learn not installed, skipping autolog")
        
        if autolog.get('tensorflow'):
            try:
                import mlflow.tensorflow
                mlflow.tensorflow.autolog()
            except ImportError:
                logger.warning("TensorFlow not installed, skipping autolog")
        
        if autolog.get('pytorch'):
            try:
                import mlflow.pytorch
                mlflow.pytorch.autolog()
            except ImportError:
                logger.warning("PyTorch not installed, skipping autolog")
        
        if autolog.get('xgboost'):
            try:
                import mlflow.xgboost
                mlflow.xgboost.autolog()
            except ImportError:
                logger.warning("XGBoost not installed, skipping autolog")
    
    def create_experiment(
        self,
        name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Create or get MLflow experiment."""
        try:
            experiment = self.client.get_experiment_by_name(name)
            if experiment:
                logger.info(f"Experiment '{name}' already exists with ID: {experiment.experiment_id}")
                return experiment.experiment_id
        except:
            pass
        
        experiment_id = self.client.create_experiment(
            name=name,
            artifact_location=artifact_location,
            tags=tags or {}
        )
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    def setup_experiments(self):
        """Setup all configured experiments."""
        experiments = self.config.get('experiments', {})
        
        for exp_key, exp_config in experiments.items():
            self.create_experiment(
                name=exp_config['name'],
                artifact_location=exp_config.get('artifact_location'),
                tags=exp_config.get('tags', {})
            )
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model",
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Register a model from a run to the model registry."""
        model_uri = f"runs:/{run_id}/{model_path}"
        
        # Register the model
        mv = mlflow.register_model(model_uri, model_name)
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=mv.version,
                    key=key,
                    value=value
                )
        
        logger.info(f"Registered model '{model_name}' version {mv.version}")
        return mv.version
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
        archive_existing: bool = True
    ):
        """Transition a model version to a new stage."""
        valid_stages = self.config['model_registry']['stages']
        
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}")
        
        # Check transition rules
        if stage == "Production":
            rules = self.config['model_registry']['transition_rules']['to_production']
            # Here you would implement checks for the rules
            logger.info(f"Checking production transition rules: {rules}")
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
        
        logger.info(f"Transitioned model '{model_name}' version {version} to stage '{stage}'")
    
    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None) -> Optional[int]:
        """Get the latest version of a model, optionally filtered by stage."""
        filter_string = f"name='{model_name}'"
        results = self.client.search_model_versions(filter_string)
        
        if not results:
            return None
        
        if stage:
            results = [mv for mv in results if mv.current_stage == stage]
        
        if results:
            return max([int(mv.version) for mv in results])
        
        return None
    
    def cleanup_old_models(self, model_name: str, keep_versions: int = 5):
        """Archive old model versions, keeping only the latest N versions."""
        filter_string = f"name='{model_name}'"
        versions = self.client.search_model_versions(filter_string)
        
        # Sort by version number
        versions.sort(key=lambda x: int(x.version), reverse=True)
        
        # Archive old versions
        for mv in versions[keep_versions:]:
            if mv.current_stage not in ["Production", "Staging"]:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived"
                )
                logger.info(f"Archived model '{model_name}' version {mv.version}")


def setup_mlflow_server():
    """Setup MLflow tracking server."""
    import subprocess
    import time
    
    # Start MLflow server
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "s3://flowops-mlflow/artifacts",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    process = subprocess.Popen(cmd)
    logger.info("Started MLflow server on http://0.0.0.0:5000")
    
    # Wait for server to start
    time.sleep(5)
    
    return process


if __name__ == "__main__":
    # Setup MLflow
    manager = MLflowManager()
    manager.setup_experiments()
    
    # Example: Create and register a model
    with mlflow.start_run(experiment_id="0") as run:
        # Log some params and metrics
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 0.95)
        
        # Log a dummy model
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        
        model = LogisticRegression()
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        mlflow.sklearn.log_model(model, "model")
        
        # Register the model
        manager.register_model(
            run_id=run.info.run_id,
            model_name="test-model",
            tags={"test": "true"}
        )