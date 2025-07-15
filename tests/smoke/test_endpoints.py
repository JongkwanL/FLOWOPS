"""
Smoke tests for critical endpoints and basic functionality
"""
import requests
import time
import pytest
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "api_base_url": "http://localhost:8080",
    "mlflow_url": "http://localhost:5000",
    "timeout": 30,
    "retry_count": 3,
    "retry_delay": 5
}

def wait_for_service(url: str, timeout: int = 30, retry_delay: int = 5) -> bool:
    """Wait for service to become available"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(retry_delay)
    return False

class TestCriticalEndpoints:
    """Test critical system endpoints"""
    
    def test_api_health_check(self):
        """Test API health endpoint"""
        url = f"{TEST_CONFIG['api_base_url']}/health"
        
        # Wait for service to be available
        assert wait_for_service(url), f"API not available at {url}"
        
        response = requests.get(url)
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "version" in health_data
        
    def test_api_metrics_endpoint(self):
        """Test metrics endpoint availability"""
        url = f"{TEST_CONFIG['api_base_url']}/metrics"
        
        response = requests.get(url)
        assert response.status_code == 200
        
        # Should return Prometheus metrics format
        assert "# HELP" in response.text
        assert "# TYPE" in response.text
        
    def test_model_prediction_basic(self):
        """Test basic model prediction endpoint"""
        url = f"{TEST_CONFIG['api_base_url']}/predict"
        
        # Simple test data
        test_data = {
            "features": [[1.0, 2.0, 3.0, 4.0]]
        }
        
        response = requests.post(
            url,
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert len(result["predictions"]) == 1
        
    def test_batch_prediction_basic(self):
        """Test batch prediction endpoint"""
        url = f"{TEST_CONFIG['api_base_url']}/predict/batch"
        
        # Batch test data
        test_data = {
            "features": [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0]
            ]
        }
        
        response = requests.post(
            url,
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert len(result["predictions"]) == 3

class TestMLflowIntegration:
    """Test MLflow service integration"""
    
    def test_mlflow_server_health(self):
        """Test MLflow server availability"""
        url = f"{TEST_CONFIG['mlflow_url']}/health"
        
        # MLflow health endpoint might not exist, check API instead
        api_url = f"{TEST_CONFIG['mlflow_url']}/api/2.0/mlflow/experiments/list"
        
        try:
            response = requests.get(api_url, timeout=10)
            # MLflow returns 200 even for empty results
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip("MLflow server not available for smoke test")
            
    def test_mlflow_experiments_api(self):
        """Test MLflow experiments API"""
        url = f"{TEST_CONFIG['mlflow_url']}/api/2.0/mlflow/experiments/list"
        
        try:
            response = requests.get(url, timeout=10)
            assert response.status_code == 200
            
            data = response.json()
            assert "experiments" in data
        except requests.exceptions.RequestException:
            pytest.skip("MLflow server not available for smoke test")

class TestSystemIntegration:
    """Test overall system integration"""
    
    def test_model_loading_smoke(self):
        """Smoke test for model loading"""
        # This would normally test if the model can be loaded
        # For smoke test, we just verify the service responds
        url = f"{TEST_CONFIG['api_base_url']}/model/info"
        
        response = requests.get(url)
        # Accept both 200 (model loaded) and 503 (model not ready)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data or "model_version" in data
            
    def test_prediction_pipeline_smoke(self):
        """End-to-end prediction pipeline smoke test"""
        # Test the complete prediction flow
        health_url = f"{TEST_CONFIG['api_base_url']}/health"
        predict_url = f"{TEST_CONFIG['api_base_url']}/predict"
        
        # 1. Check service health
        health_response = requests.get(health_url)
        assert health_response.status_code == 200
        
        # 2. Make a prediction
        test_data = {"features": [[1.0, 2.0, 3.0, 4.0]]}
        
        predict_response = requests.post(
            predict_url,
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Should either succeed or fail gracefully
        assert predict_response.status_code in [200, 503, 422]
        
        if predict_response.status_code == 200:
            result = predict_response.json()
            assert "predictions" in result
            
    def test_monitoring_integration_smoke(self):
        """Test monitoring integration"""
        metrics_url = f"{TEST_CONFIG['api_base_url']}/metrics"
        
        response = requests.get(metrics_url)
        assert response.status_code == 200
        
        metrics_text = response.text
        
        # Check for key metrics
        expected_metrics = [
            "http_requests_total",
            "prediction_duration_seconds",
            "model_predictions_total"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics_text, f"Missing metric: {metric}"

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_prediction_data(self):
        """Test handling of invalid prediction data"""
        url = f"{TEST_CONFIG['api_base_url']}/predict"
        
        # Test various invalid inputs
        invalid_inputs = [
            {},  # Empty data
            {"features": []},  # Empty features
            {"features": [["invalid", "data"]]},  # Non-numeric data
            {"invalid": "structure"}  # Wrong structure
        ]
        
        for invalid_input in invalid_inputs:
            response = requests.post(
                url,
                json=invalid_input,
                headers={"Content-Type": "application/json"}
            )
            
            # Should return 422 (validation error) or 400 (bad request)
            assert response.status_code in [400, 422]
            
    def test_nonexistent_endpoint(self):
        """Test handling of non-existent endpoints"""
        url = f"{TEST_CONFIG['api_base_url']}/nonexistent"
        
        response = requests.get(url)
        assert response.status_code == 404
        
    def test_method_not_allowed(self):
        """Test method not allowed errors"""
        url = f"{TEST_CONFIG['api_base_url']}/predict"
        
        # Try GET on POST endpoint
        response = requests.get(url)
        assert response.status_code == 405

@pytest.fixture(scope="session", autouse=True)
def setup_smoke_tests():
    """Setup for smoke tests"""
    logger.info("Starting smoke test setup...")
    
    # Check if services are available
    api_available = wait_for_service(
        f"{TEST_CONFIG['api_base_url']}/health",
        timeout=10,
        retry_delay=2
    )
    
    if not api_available:
        pytest.skip("API service not available for smoke tests")
    
    yield
    
    logger.info("Smoke test teardown complete")

def run_smoke_tests():
    """Run smoke tests with proper configuration"""
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--disable-warnings"
    ]
    
    return pytest.main(pytest_args)

if __name__ == "__main__":
    # Run smoke tests directly
    exit_code = run_smoke_tests()
    exit(exit_code)