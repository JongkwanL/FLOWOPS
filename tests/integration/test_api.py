"""Integration tests for model serving API."""

import pytest
import requests
import json
import time
import numpy as np
from typing import Dict, Any
from fastapi.testclient import TestClient
import tempfile
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import the serving application
import sys
sys.path.append('../../pipelines/deployment')
from pipelines.deployment.serve import app, model_cache


class TestModelServingAPI:
    """Integration tests for model serving API."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client for FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture(scope="class")
    def sample_model(self):
        """Create and save a sample model for testing."""
        # Create sample data and model
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Save model to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
        joblib.dump(model, temp_file.name)
        
        yield temp_file.name, model
        
        # Cleanup
        import os
        os.unlink(temp_file.name)
    
    @pytest.fixture(autouse=True)
    def setup_model_cache(self, sample_model):
        """Setup model cache for testing."""
        model_path, model = sample_model
        
        # Mock model config
        from pipelines.deployment.serve import ModelConfig
        config = ModelConfig(
            model_name="test-model",
            model_version="1.0.0",
            model_stage="Testing",
            mlflow_tracking_uri="http://localhost:5000"
        )
        
        # Load model into cache
        model_cache['model'] = model
        model_cache['config'] = config
        model_cache['start_time'] = time.time()
        
        yield
        
        # Cleanup
        model_cache.clear()
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data['status'] in ['healthy', 'degraded']
        assert isinstance(data['model_loaded'], bool)
        assert isinstance(data['uptime_seconds'], float)
        
        if data['model_loaded']:
            assert data['model_name'] is not None
            assert data['model_version'] is not None
    
    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/ready")
        
        # Should be ready when model is loaded
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ready'
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'model_name' in data
        assert 'model_version' in data
        assert 'model_stage' in data
        assert 'cache_enabled' in data
        assert 'cache_ttl' in data
    
    def test_single_prediction(self, client):
        """Test single prediction endpoint."""
        # Valid prediction request
        prediction_data = {
            "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "request_id": "test-request-1"
        }
        
        response = client.post("/predict", json=prediction_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'prediction' in data
        assert 'model_version' in data
        assert 'request_id' in data
        assert 'latency_ms' in data
        assert 'timestamp' in data
        
        assert isinstance(data['prediction'], (int, float))
        assert data['request_id'] == prediction_data['request_id']
        assert data['latency_ms'] > 0
    
    def test_prediction_with_probabilities(self, client):
        """Test prediction endpoint returns probabilities."""
        prediction_data = {
            "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        
        response = client.post("/predict", json=prediction_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have probabilities for RandomForest
        assert 'probability' in data
        if data['probability']:
            assert isinstance(data['probability'], list)
            assert len(data['probability']) == 2  # Binary classification
            assert all(0 <= p <= 1 for p in data['probability'])
            assert abs(sum(data['probability']) - 1.0) < 1e-6  # Should sum to 1
    
    def test_batch_prediction(self, client):
        """Test batch prediction endpoint."""
        batch_data = [
            {"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "request_id": "batch-1"},
            {"features": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], "request_id": "batch-2"},
            {"features": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], "request_id": "batch-3"}
        ]
        
        response = client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list)
        assert len(data) == len(batch_data)
        
        for i, prediction in enumerate(data):
            assert 'prediction' in prediction
            assert 'request_id' in prediction
            assert prediction['request_id'] == batch_data[i]['request_id']
    
    def test_invalid_prediction_request(self, client):
        """Test prediction with invalid data."""
        # Empty features
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422  # Validation error
        
        # Wrong number of features (should be 10)
        response = client.post("/predict", json={"features": [1.0, 2.0]})
        assert response.status_code == 500  # Model prediction error
        
        # Non-numeric features
        response = client.post("/predict", json={"features": ["a", "b", "c"]})
        assert response.status_code == 422  # Validation error
        
        # Too many features
        too_many_features = list(range(2000))  # Exceeds max limit
        response = client.post("/predict", json={"features": too_many_features})
        assert response.status_code == 422  # Validation error
    
    def test_batch_size_limit(self, client):
        """Test batch prediction size limit."""
        # Create batch larger than limit (100)
        large_batch = [
            {"features": [0.1] * 10} for _ in range(101)
        ]
        
        response = client.post("/predict/batch", json=large_batch)
        assert response.status_code == 400
        assert "Batch size too large" in response.json()['detail']
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        # Make some predictions first to generate metrics
        for i in range(5):
            client.post("/predict", json={"features": [0.1] * 10})
        
        response = client.get("/metrics")
        assert response.status_code == 200
        
        metrics_text = response.text
        
        # Check for expected metrics
        assert "model_predictions_total" in metrics_text
        assert "model_prediction_duration_seconds" in metrics_text
        assert "model_active_requests" in metrics_text
    
    def test_prediction_performance(self, client):
        """Test prediction performance and latency."""
        prediction_data = {"features": [0.1] * 10}
        
        # Measure latency
        start_time = time.time()
        response = client.post("/predict", json=prediction_data)
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Check response latency
        response_latency = (end_time - start_time) * 1000  # Convert to ms
        reported_latency = response.json()['latency_ms']
        
        # Reported latency should be reasonable
        assert 0 < reported_latency < 5000  # Less than 5 seconds
        
        # Response time should be fast
        assert response_latency < 1000  # Less than 1 second for integration test
    
    def test_concurrent_predictions(self, client):
        """Test concurrent prediction requests."""
        import concurrent.futures
        import threading
        
        def make_prediction(request_id):
            prediction_data = {
                "features": [0.1] * 10,
                "request_id": f"concurrent-{request_id}"
            }
            response = client.post("/predict", json=prediction_data)
            return response.status_code, response.json()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        for status_code, data in results:
            assert status_code == 200
            assert 'prediction' in data
            assert 'request_id' in data
    
    def test_model_reload(self, client):
        """Test model reload endpoint."""
        # This test might need mock MLflow since we're not running a full MLflow server
        with pytest.raises(Exception):
            # Should fail because MLflow is not available in test
            response = client.post("/model/reload")
    
    def test_request_logging(self, client, caplog):
        """Test that requests are properly logged."""
        import logging
        caplog.set_level(logging.INFO)
        
        prediction_data = {"features": [0.1] * 10, "request_id": "log-test"}
        
        response = client.post("/predict", json=prediction_data)
        assert response.status_code == 200
        
        # Check that request was logged
        log_messages = [record.message for record in caplog.records]
        request_logs = [msg for msg in log_messages if "Request:" in msg]
        response_logs = [msg for msg in log_messages if "Response:" in msg]
        
        assert len(request_logs) > 0
        assert len(response_logs) > 0


class TestAPILoadTesting:
    """Load testing for API endpoints."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client for load testing."""
        return TestClient(app)
    
    @pytest.fixture(scope="class", autouse=True)
    def setup_model_cache(self):
        """Setup model cache for load testing."""
        # Create sample model
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Mock model config
        from pipelines.deployment.serve import ModelConfig
        config = ModelConfig(
            model_name="load-test-model",
            model_version="1.0.0",
            model_stage="Testing",
            mlflow_tracking_uri="http://localhost:5000"
        )
        
        model_cache['model'] = model
        model_cache['config'] = config
        model_cache['start_time'] = time.time()
        
        yield
        
        model_cache.clear()
    
    @pytest.mark.slow
    def test_sustained_load(self, client):
        """Test API under sustained load."""
        import concurrent.futures
        import statistics
        
        def make_prediction(request_id):
            start_time = time.time()
            prediction_data = {
                "features": np.random.rand(10).tolist(),
                "request_id": f"load-{request_id}"
            }
            response = client.post("/predict", json=prediction_data)
            latency = (time.time() - start_time) * 1000
            return response.status_code == 200, latency
        
        # Generate load: 100 requests with 20 concurrent workers
        num_requests = 100
        max_workers = 20
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        # Analyze results
        successful_requests = sum(1 for success, _ in results if success)
        latencies = [latency for success, latency in results if success]
        
        total_time = end_time - start_time
        throughput = successful_requests / total_time
        
        # Assertions
        assert successful_requests >= num_requests * 0.95  # At least 95% success rate
        assert statistics.mean(latencies) < 1000  # Average latency < 1 second
        assert statistics.quantile(latencies, 0.95) < 2000  # 95th percentile < 2 seconds
        assert throughput > 10  # At least 10 requests/second
        
        print(f"Load test results:")
        print(f"- Successful requests: {successful_requests}/{num_requests}")
        print(f"- Throughput: {throughput:.2f} requests/second")
        print(f"- Average latency: {statistics.mean(latencies):.2f} ms")
        print(f"- 95th percentile latency: {statistics.quantile(latencies, 0.95):.2f} ms")


class TestAPIErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_model_not_loaded(self, client):
        """Test API behavior when model is not loaded."""
        # Clear model cache to simulate model not loaded
        model_cache.clear()
        
        # Health check should show model not loaded
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'degraded'
        assert not data['model_loaded']
        
        # Readiness check should fail
        response = client.get("/ready")
        assert response.status_code == 503
        
        # Predictions should fail
        response = client.post("/predict", json={"features": [0.1] * 10})
        assert response.status_code == 503
        
        # Model info should fail
        response = client.get("/model/info")
        assert response.status_code == 503
    
    def test_malformed_requests(self, client):
        """Test handling of malformed requests."""
        # Setup model
        X, y = make_classification(n_samples=50, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        model_cache['model'] = model
        
        # Invalid JSON
        response = client.post("/predict", data="invalid json")
        assert response.status_code == 422
        
        # Missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        # Wrong data types
        response = client.post("/predict", json={"features": "not a list"})
        assert response.status_code == 422


if __name__ == '__main__':
    pytest.main([__file__, "-v"])