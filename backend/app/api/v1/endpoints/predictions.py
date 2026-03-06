"""
Price prediction endpoints for SteadyPrice Enterprise
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import structlog
import time

from app.models.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelType,
    ErrorResponse
)
from app.services.prediction_service import PredictionService
from app.core.security import verify_token, get_current_user
from app.core.rate_limit import rate_limit
from app.utils.metrics import track_prediction

logger = structlog.get_logger()
router = APIRouter()

# Initialize prediction service
prediction_service = PredictionService()

@router.post("/predict", response_model=PredictionResponse)
@rate_limit(calls=100, period=60)  # 100 calls per minute
async def predict_price(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
) -> PredictionResponse:
    """
    Predict product price from description
    
    - **title**: Product title (required)
    - **description**: Product description (optional)
    - **category**: Product category (required)
    - **model_type**: Model type to use (optional, defaults to ensemble)
    
    Returns predicted price with confidence score and processing metrics.
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.title.strip():
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        
        # Make prediction
        result = await prediction_service.predict_single(request)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        result.processing_time_ms = processing_time
        
        # Track metrics in background
        background_tasks.add_task(
            track_prediction,
            model_type=result.model_used,
            confidence=result.confidence_score,
            processing_time=processing_time,
            success=True
        )
        
        logger.info(
            "Price prediction completed",
            title=request.title[:50],
            category=request.category,
            predicted_price=result.predicted_price,
            confidence=result.confidence_score,
            model_type=result.model_used,
            processing_time_ms=processing_time
        )
        
        return result
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        
        # Track failed prediction
        background_tasks.add_task(
            track_prediction,
            model_type=request.model_type,
            confidence=0.0,
            processing_time=processing_time,
            success=False
        )
        
        logger.error(
            "Price prediction failed",
            title=request.title[:50],
            category=request.category,
            error=str(e),
            processing_time_ms=processing_time
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@router.post("/predict/batch", response_model=BatchPredictionResponse)
@rate_limit(calls=10, period=60)  # 10 batch calls per minute
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_token)
) -> BatchPredictionResponse:
    """
    Predict prices for multiple products
    
    - **products**: List of product prediction requests (1-100 items)
    - **model_type**: Model type to use (optional, defaults to ensemble)
    
    Returns predictions for all products with processing metrics.
    """
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.products) > 100:
            raise HTTPException(
                status_code=400, 
                detail="Batch size cannot exceed 100 items"
            )
        
        # Make batch predictions
        results = await prediction_service.predict_batch(request.products, request.model_type)
        
        # Calculate total processing time
        total_processing_time = (time.time() - start_time) * 1000
        
        # Count successful and failed predictions
        success_count = len([r for r in results if r is not None])
        error_count = len(results) - success_count
        
        # Filter out None results (failed predictions)
        successful_predictions = [r for r in results if r is not None]
        
        # Track batch metrics
        background_tasks.add_task(
            track_prediction,
            model_type=request.model_type,
            confidence=sum(r.confidence_score for r in successful_predictions) / len(successful_predictions) if successful_predictions else 0,
            processing_time=total_processing_time,
            success=success_count > 0
        )
        
        logger.info(
            "Batch prediction completed",
            total_items=len(request.products),
            success_count=success_count,
            error_count=error_count,
            total_processing_time_ms=total_processing_time
        )
        
        return BatchPredictionResponse(
            predictions=successful_predictions,
            total_processing_time_ms=total_processing_time,
            success_count=success_count,
            error_count=error_count
        )
        
    except Exception as e:
        total_processing_time = (time.time() - start_time) * 1000
        
        logger.error(
            "Batch prediction failed",
            batch_size=len(request.products),
            error=str(e),
            total_processing_time_ms=total_processing_time
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@router.get("/models/{model_type}/metrics")
async def get_model_metrics(
    model_type: ModelType,
    token: str = Depends(verify_token)
):
    """
    Get performance metrics for a specific model type
    
    - **model_type**: Type of model to get metrics for
    
    Returns accuracy, MAE, RMSE, MAPE, and R² scores.
    """
    try:
        metrics = await prediction_service.get_model_metrics(model_type)
        return metrics
        
    except Exception as e:
        logger.error(
            "Failed to get model metrics",
            model_type=model_type,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model metrics: {str(e)}"
        )

@router.get("/models/compare")
async def compare_models(
    token: str = Depends(verify_token)
):
    """
    Compare performance across all model types
    
    Returns metrics for all available models for comparison.
    """
    try:
        comparison = await prediction_service.compare_models()
        return comparison
        
    except Exception as e:
        logger.error(
            "Failed to compare models",
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare models: {str(e)}"
        )

@router.get("/health")
async def prediction_health_check():
    """
    Health check for prediction service
    
    Returns status of all prediction models and services.
    """
    try:
        health_status = await prediction_service.health_check()
        return health_status
        
    except Exception as e:
        logger.error(
            "Prediction health check failed",
            error=str(e)
        )
        
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )
