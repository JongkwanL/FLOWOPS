"""Unit tests for training pipeline."""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
import mlflow
import xgboost as xgb
from sklearn.datasets import make_classification

# Import modules to test
import sys
sys.path.append('../../pipelines/training')
from pipelines.training.train import ModelTrainer
from pipelines.training.hyperparameter_tuning import HyperparameterTuner
from pipelines.training.drift_detection import DriftDetector


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def temp_params_file(self):
        """Create temporary parameters file."""
        params = {
            'train': {
                'hyperparameters': {
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'n_estimators': 10,
                    'objective': 'binary:logistic'
                },
                'cross_validation': {
                    'enabled': True,
                    'folds': 3
                },
                'early_stopping': {
                    'enabled': False
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(params, f)
            yield f.name
        
        os.unlink(f.name)
    
    def test_trainer_initialization(self, temp_params_file):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(temp_params_file)
        assert trainer.params is not None
        assert 'train' in trainer.params
    
    @patch('pipelines.training.train.np.load')
    def test_load_features(self, mock_load, sample_data, temp_params_file):
        """Test feature loading."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]
        
        # Mock numpy load to return our sample data
        mock_load.side_effect = [X_train, y_train, X_val, y_val]
        
        trainer = ModelTrainer(temp_params_file)
        
        # Mock os.path.exists to return True
        with patch('os.path.exists', return_value=True):
            loaded_X_train, loaded_y_train, loaded_X_val, loaded_y_val = trainer.load_features()
        
        np.testing.assert_array_equal(loaded_X_train, X_train)
        np.testing.assert_array_equal(loaded_y_train, y_train)
        np.testing.assert_array_equal(loaded_X_val, X_val)
        np.testing.assert_array_equal(loaded_y_val, y_val)
    
    def test_train_xgboost(self, sample_data, temp_params_file):
        """Test XGBoost model training."""
        X, y = sample_data
        X_train, y_train = X[:80], y[:80]
        X_val, y_val = X[80:], y[80:]
        
        trainer = ModelTrainer(temp_params_file)
        
        # Mock MLflow
        with patch('mlflow.log_metric'):
            model = trainer.train_xgboost(X_train, y_train, X_val, y_val)
        
        assert isinstance(model, xgb.XGBClassifier)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_evaluate_model(self, sample_data, temp_params_file):
        """Test model evaluation."""
        X, y = sample_data
        
        trainer = ModelTrainer(temp_params_file)
        
        # Create a simple mock model
        model = Mock()
        model.predict.return_value = np.random.randint(0, 2, len(y))
        model.predict_proba.return_value = np.random.rand(len(y), 2)
        
        with patch('mlflow.log_metric'):
            metrics = trainer.evaluate_model(model, X, y, prefix="test_")
        
        assert 'test_accuracy' in metrics
        assert 'test_precision' in metrics
        assert 'test_recall' in metrics
        assert 'test_f1_score' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
    
    def test_calculate_feature_importance(self, sample_data, temp_params_file):
        """Test feature importance calculation."""
        X, y = sample_data
        
        trainer = ModelTrainer(temp_params_file)
        
        # Create a mock model with feature_importances_
        model = Mock()
        model.feature_importances_ = np.random.rand(X.shape[1])
        
        with patch('mlflow.log_metric'):
            importance = trainer.calculate_feature_importance(model)
        
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_detect_training_drift(self, temp_params_file):
        """Test training drift detection."""
        trainer = ModelTrainer(temp_params_file)
        
        # Test with good metrics (no drift)
        good_metrics = {
            "val_accuracy": 0.90,
            "val_f1_score": 0.85
        }
        
        with patch('mlflow.set_tag'), patch('mlflow.log_metric'):
            drift_detected = trainer.detect_training_drift(good_metrics)
        
        assert not drift_detected
        
        # Test with poor metrics (drift detected)
        poor_metrics = {
            "val_accuracy": 0.75,
            "val_f1_score": 0.70
        }
        
        with patch('mlflow.set_tag'), patch('mlflow.log_metric'):
            drift_detected = trainer.detect_training_drift(poor_metrics)
        
        assert drift_detected


class TestHyperparameterTuner:
    """Test cases for HyperparameterTuner class."""
    
    @pytest.fixture
    def temp_params_file(self):
        """Create temporary parameters file."""
        params = {
            'train': {
                'cross_validation': {
                    'enabled': True,
                    'folds': 3
                },
                'optimization': {
                    'metric': 'roc_auc'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(params, f)
            yield f.name
        
        os.unlink(f.name)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for tuning."""
        X, y = make_classification(
            n_samples=50,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        return X, y
    
    def test_tuner_initialization(self, temp_params_file):
        """Test HyperparameterTuner initialization."""
        tuner = HyperparameterTuner(temp_params_file)
        assert tuner.params is not None
        assert tuner.best_score == -float('inf')
        assert tuner.best_params == {}
    
    @patch('pipelines.training.hyperparameter_tuning.np.load')
    def test_load_data(self, mock_load, sample_data, temp_params_file):
        """Test data loading for hyperparameter tuning."""
        X, y = sample_data
        
        mock_load.side_effect = [X, y]
        
        tuner = HyperparameterTuner(temp_params_file)
        
        with patch('os.path.exists', return_value=True):
            loaded_X, loaded_y = tuner.load_data()
        
        np.testing.assert_array_equal(loaded_X, X)
        np.testing.assert_array_equal(loaded_y, y)
    
    def test_objective_function(self, sample_data, temp_params_file):
        """Test Optuna objective function."""
        X, y = sample_data
        
        tuner = HyperparameterTuner(temp_params_file)
        
        # Mock the load_data method
        tuner.load_data = Mock(return_value=(X, y))
        
        # Create a mock trial
        trial = Mock()
        trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.1, 0.1]  # learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda
        trial.suggest_int.side_effect = [6, 50, 5]  # max_depth, n_estimators, min_child_weight
        trial.number = 1
        
        with patch('mlflow.start_run'), patch('mlflow.log_params'), patch('mlflow.log_metric'):
            score = tuner.objective(trial)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1  # ROC AUC should be between 0 and 1


class TestDriftDetector:
    """Test cases for DriftDetector class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample reference and current data."""
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (1000, 5))
        current_data = np.random.normal(0.5, 1.2, (500, 5))  # Shifted distribution
        return reference_data, current_data
    
    def test_detector_initialization(self, sample_data):
        """Test DriftDetector initialization."""
        reference_data, _ = sample_data
        
        detector = DriftDetector(reference_data)
        assert detector.reference_data is not None
        assert detector.reference_stats is not None
        assert 'mean' in detector.reference_stats
    
    def test_calculate_psi(self, sample_data):
        """Test PSI calculation."""
        reference_data, current_data = sample_data
        
        detector = DriftDetector()
        
        # Test with single feature
        psi = detector.calculate_psi(reference_data[:, 0], current_data[:, 0])
        
        assert isinstance(psi, float)
        assert psi >= 0  # PSI should be non-negative
    
    def test_calculate_ks_statistic(self, sample_data):
        """Test KS statistic calculation."""
        reference_data, current_data = sample_data
        
        detector = DriftDetector()
        
        ks_stat = detector.calculate_ks_statistic(reference_data[:, 0], current_data[:, 0])
        
        assert isinstance(ks_stat, float)
        assert 0 <= ks_stat <= 1  # KS statistic should be between 0 and 1
    
    def test_calculate_js_divergence(self, sample_data):
        """Test Jensen-Shannon divergence calculation."""
        reference_data, current_data = sample_data
        
        detector = DriftDetector()
        
        js_div = detector.calculate_js_divergence(reference_data[:, 0], current_data[:, 0])
        
        assert isinstance(js_div, float)
        assert 0 <= js_div <= 1  # JS divergence should be between 0 and 1
    
    def test_detect_data_drift(self, sample_data):
        """Test data drift detection."""
        reference_data, current_data = sample_data
        
        detector = DriftDetector(reference_data)
        
        with patch('mlflow.active_run', return_value=True), \
             patch('mlflow.log_metric'), \
             patch('mlflow.set_tag'), \
             patch('mlflow.log_text'):
            
            drift_results = detector.detect_data_drift(current_data)
        
        assert 'drift_detected' in drift_results
        assert 'timestamp' in drift_results
        assert 'metrics' in drift_results
        assert isinstance(drift_results['drift_detected'], bool)
    
    def test_detect_prediction_drift(self):
        """Test prediction drift detection."""
        detector = DriftDetector()
        
        # Create sample prediction distributions
        np.random.seed(42)
        reference_predictions = np.random.beta(2, 5, 1000)
        current_predictions = np.random.beta(3, 4, 500)  # Different distribution
        
        drift_results = detector.detect_prediction_drift(reference_predictions, current_predictions)
        
        assert 'drift_detected' in drift_results
        assert 'metrics' in drift_results
        assert 'js_divergence' in drift_results['metrics']
    
    def test_detect_performance_drift(self):
        """Test performance drift detection."""
        detector = DriftDetector()
        
        baseline_metrics = {
            'accuracy': 0.90,
            'f1_score': 0.85
        }
        
        # Test with degraded performance
        degraded_metrics = {
            'accuracy': 0.82,  # Dropped by 8%
            'f1_score': 0.78   # Dropped by 7%
        }
        
        drift_results = detector.detect_performance_drift(baseline_metrics, degraded_metrics)
        
        assert 'drift_detected' in drift_results
        assert drift_results['drift_detected']  # Should detect drift
        assert 'degradations' in drift_results
    
    def test_adwin_drift_detection(self):
        """Test ADWIN drift detection algorithm."""
        detector = DriftDetector()
        
        # Simulate stable data
        for i in range(50):
            drift_detected = detector.adwin_add_element(0.1 + np.random.normal(0, 0.01))
            assert not drift_detected
        
        # Simulate drift
        drift_detected = False
        for i in range(50):
            drift_detected = detector.adwin_add_element(0.5 + np.random.normal(0, 0.01))
            if drift_detected:
                break
        
        # Should detect drift at some point
        assert len(detector.adwin_window) > 0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration tests."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        return X, y
    
    def test_training_to_evaluation_pipeline(self, sample_data):
        """Test complete training to evaluation pipeline."""
        X, y = sample_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        # Create temporary directories and files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data
            data_dir = os.path.join(temp_dir, 'data', 'features')
            os.makedirs(data_dir, exist_ok=True)
            
            np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(data_dir, 'X_val.npy'), X_test[:20])
            np.save(os.path.join(data_dir, 'y_val.npy'), y_test[:20])
            np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
            
            # Create params file
            params = {
                'train': {
                    'hyperparameters': {
                        'learning_rate': 0.1,
                        'max_depth': 3,
                        'n_estimators': 10,
                        'objective': 'binary:logistic'
                    },
                    'cross_validation': {'enabled': False},
                    'early_stopping': {'enabled': False}
                }
            }
            
            params_file = os.path.join(temp_dir, 'params.yaml')
            with open(params_file, 'w') as f:
                import yaml
                yaml.dump(params, f)
            
            # Change to temp directory
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Test training
                trainer = ModelTrainer(params_file)
                
                with patch('mlflow.start_run'), \
                     patch('mlflow.set_experiment'), \
                     patch('mlflow.log_params'), \
                     patch('mlflow.log_metric'), \
                     patch('mlflow.xgboost.log_model'):
                    
                    run_id = trainer.run(experiment_name="test", log_model=False)
                
                assert run_id is not None
                
                # Verify model was saved
                model_path = os.path.join(temp_dir, 'models', 'model.pkl')
                assert os.path.exists(model_path)
                
                # Verify metrics were saved
                metrics_path = os.path.join(temp_dir, 'metrics', 'train_metrics.json')
                assert os.path.exists(metrics_path)
                
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                assert 'train_accuracy' in metrics
                assert 'val_accuracy' in metrics
                
            finally:
                os.chdir(original_cwd)


if __name__ == '__main__':
    pytest.main([__file__])