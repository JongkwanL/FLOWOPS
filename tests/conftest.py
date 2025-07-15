"""
Pytest configuration and shared fixtures for FlowOps testing
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import mlflow
from unittest.mock import MagicMock
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="flowops_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def sample_dataset():
    """Generate sample dataset for testing"""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

@pytest.fixture(scope="session")
def train_test_split_data(sample_dataset):
    """Split sample dataset into train/test"""
    from sklearn.model_selection import train_test_split
    
    X = sample_dataset.drop('target', axis=1)
    y = sample_dataset['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

@pytest.fixture(scope="function")
def mock_mlflow_client():
    """Mock MLflow client for testing"""
    mock_client = MagicMock()
    
    # Mock experiment creation
    mock_client.create_experiment.return_value = "test_experiment_id"
    mock_client.get_experiment_by_name.return_value = None
    
    # Mock run creation
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_client.create_run.return_value = mock_run
    
    # Mock model registration
    mock_model_version = MagicMock()
    mock_model_version.version = "1"
    mock_client.create_model_version.return_value = mock_model_version
    
    return mock_client

@pytest.fixture(scope="function")
def mock_model():
    """Create mock model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create mock training data
    X_mock = np.random.rand(100, 10)
    y_mock = np.random.randint(0, 2, 100)
    
    # Fit the model
    model.fit(X_mock, y_mock)
    
    return model

@pytest.fixture(scope="function")
def temp_model_dir(test_data_dir):
    """Create temporary directory for model artifacts"""
    model_dir = test_data_dir / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir

@pytest.fixture(scope="function")
def sample_config():
    """Sample configuration for testing"""
    return {
        "model": {
            "type": "random_forest",
            "params": {
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42
            }
        },
        "training": {
            "test_size": 0.2,
            "validation_size": 0.1,
            "cv_folds": 3
        },
        "mlflow": {
            "experiment_name": "test_experiment",
            "tracking_uri": "sqlite:///test_mlflow.db"
        },
        "drift_detection": {
            "enabled": True,
            "methods": ["psi", "ks_test"],
            "thresholds": {
                "psi": 0.2,
                "ks_test": 0.05
            }
        }
    }

@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Cleanup MLflow after each test"""
    yield
    try:
        mlflow.end_run()
    except Exception:
        pass

@pytest.fixture(scope="function")
def api_test_client():
    """Create test client for FastAPI testing"""
    try:
        from fastapi.testclient import TestClient
        from pipelines.deployment.serve import app
        
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI test client not available")

@pytest.fixture(scope="function") 
def sample_prediction_data():
    """Sample data for prediction testing"""
    return {
        "features": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        ]
    }

# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "smoke: marks tests as smoke tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )

# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location"""
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
            
        # Mark slow tests
        if "slow" in item.name or "load_test" in item.name:
            item.add_marker(pytest.mark.slow)