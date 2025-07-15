"""FastAPI model serving application with monitoring and health checks."""

import os
import json
import time
import joblib
import mlflow
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import asyncio
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
prediction_counter = Counter('model_predictions_total', 'Total number of predictions', ['model', 'version', 'status'])
prediction_latency = Histogram('model_prediction_duration_seconds', 'Prediction latency', ['model', 'version'])
active_requests = Gauge('model_active_requests', 'Number of active requests')
model_info_gauge = Gauge('model_info', 'Model information', ['model_name', 'version', 'stage'])

# Global model storage
model_cache = {}


class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[float] = Field(..., description="Input features for prediction")
    request_id: Optional[str] = Field(None, description="Unique request ID for tracking")
    
    @validator('features')
    def validate_features(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Features cannot be empty")
        if len(v) > 1000:  # Prevent overly large requests
            raise ValueError("Too many features (max 1000)")
        return v


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    prediction: float
    probability: Optional[List[float]] = None
    model_version: str
    request_id: Optional[str] = None
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    uptime_seconds: float


class ModelConfig(BaseModel):
    """Model configuration."""
    model_name: str
    model_version: str
    model_stage: str
    mlflow_tracking_uri: str
    cache_enabled: bool = True
    cache_ttl: int = 300


# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting model serving application...")
    
    # Load model
    try:
        config = load_model_config()
        model = load_model(config)
        model_cache['model'] = model
        model_cache['config'] = config
        model_cache['start_time'] = time.time()
        
        # Set model info metric
        model_info_gauge.labels(
            model_name=config.model_name,
            version=config.model_version,
            stage=config.model_stage
        ).set(1)
        
        logger.info(f"Model loaded: {config.model_name} v{config.model_version}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue anyway for health checks to work
    
    yield
    
    # Shutdown
    logger.info("Shutting down model serving application...")
    model_cache.clear()


# Create FastAPI app
app = FastAPI(
    title="FlowOps Model Serving",
    description="Production ML model serving with monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_config() -> ModelConfig:
    """Load model configuration from environment variables."""
    return ModelConfig(
        model_name=os.getenv("MODEL_NAME", "flowops-model"),
        model_version=os.getenv("MODEL_VERSION", "latest"),
        model_stage=os.getenv("MODEL_STAGE", "Production"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
        cache_ttl=int(os.getenv("CACHE_TTL", "300"))
    )


def load_model(config: ModelConfig) -> Any:
    """Load model from MLflow or local file."""
    try:
        # Try loading from MLflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        if config.model_version == "latest":
            # Get latest version from registry
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{config.model_name}'")
            
            # Filter by stage
            staged_versions = [v for v in versions if v.current_stage == config.model_stage]
            
            if staged_versions:
                latest = max(staged_versions, key=lambda x: int(x.version))
                model_uri = f"models:/{config.model_name}/{latest.version}"
            else:
                raise ValueError(f"No model found in {config.model_stage} stage")
        else:
            model_uri = f"models:/{config.model_name}/{config.model_version}"
        
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model from MLflow: {model_uri}")
        return model
        
    except Exception as e:
        logger.warning(f"Failed to load from MLflow: {e}")
        
        # Fallback to local model file
        local_path = os.getenv("LOCAL_MODEL_PATH", "/app/models/model.pkl")
        if os.path.exists(local_path):
            model = joblib.load(local_path)
            logger.info(f"Loaded model from local file: {local_path}")
            return model
        
        raise RuntimeError("No model available")


@lru_cache(maxsize=1000)
def cached_prediction(features_tuple: tuple) -> Dict[str, Any]:
    """Cached prediction function."""
    features = np.array(features_tuple).reshape(1, -1)
    model = model_cache.get('model')
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Get probabilities if available
    probability = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(features)[0].tolist()
    
    return {
        'prediction': float(prediction),
        'probability': probability
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a prediction using the loaded model."""
    start_time = time.time()
    active_requests.inc()
    
    try:
        # Get model and config
        model = model_cache.get('model')
        config = model_cache.get('config', load_model_config())
        
        if model is None:
            prediction_counter.labels(
                model=config.model_name,
                version=config.model_version,
                status='error'
            ).inc()
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Use cache if enabled
        if config.cache_enabled:
            result = cached_prediction(tuple(request.features))
        else:
            features = np.array(request.features).reshape(1, -1)
            prediction = model.predict(features)[0]
            probability = None
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(features)[0].tolist()
            
            result = {
                'prediction': float(prediction),
                'probability': probability
            }
        
        # Calculate latency
        latency = (time.time() - start_time) * 1000
        
        # Update metrics
        prediction_counter.labels(
            model=config.model_name,
            version=config.model_version,
            status='success'
        ).inc()
        
        prediction_latency.labels(
            model=config.model_name,
            version=config.model_version
        ).observe(latency / 1000)
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction,
            request_id=request.request_id,
            features=request.features,
            prediction=result['prediction'],
            latency=latency
        )
        
        return PredictionResponse(
            prediction=result['prediction'],
            probability=result.get('probability'),
            model_version=config.model_version,
            request_id=request.request_id,
            latency_ms=latency,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        prediction_counter.labels(
            model=config.model_name if 'config' in locals() else 'unknown',
            version=config.model_version if 'config' in locals() else 'unknown',
            status='error'
        ).inc()
        
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        active_requests.dec()


@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]) -> List[PredictionResponse]:
    """Make batch predictions."""
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    responses = []
    for request in requests:
        response = await predict(request, BackgroundTasks())
        responses.append(response)
    
    return responses


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = 'model' in model_cache
    config = model_cache.get('config')
    uptime = time.time() - model_cache.get('start_time', time.time())
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=config.model_name if config else None,
        model_version=config.model_version if config else None,
        uptime_seconds=uptime
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    if 'model' not in model_cache:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
async def model_info():
    """Get model information."""
    config = model_cache.get('config')
    
    if not config:
        raise HTTPException(status_code=503, detail="Model information not available")
    
    return {
        "model_name": config.model_name,
        "model_version": config.model_version,
        "model_stage": config.model_stage,
        "cache_enabled": config.cache_enabled,
        "cache_ttl": config.cache_ttl
    }


@app.post("/model/reload")
async def reload_model():
    """Reload the model (admin endpoint)."""
    try:
        config = load_model_config()
        model = load_model(config)
        
        # Clear cache
        cached_prediction.cache_clear()
        
        # Update model cache
        model_cache['model'] = model
        model_cache['config'] = config
        
        logger.info(f"Model reloaded: {config.model_name} v{config.model_version}")
        
        return {"status": "success", "message": "Model reloaded"}
    
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def log_prediction(
    request_id: Optional[str],
    features: List[float],
    prediction: float,
    latency: float
):
    """Log prediction for monitoring and analysis."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "features": features,
        "prediction": prediction,
        "latency_ms": latency
    }
    
    # In production, send to logging service or data lake
    logger.info(f"Prediction logged: {json.dumps(log_entry)}")
    
    # Here you could also:
    # - Send to Kafka for stream processing
    # - Store in database for drift detection
    # - Send to monitoring service


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__
        }
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8080,
        workers=2,
        log_level="info",
        access_log=True
    )