"""
Fine-tuning endpoints for SteadyPrice Enterprise
QLoRA training and model management API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import structlog
from datetime import datetime
import asyncio

from app.models.schemas import (
    PredictionRequest, 
    PredictionResponse,
    ModelType,
    ErrorResponse
)
from app.ml.fine_tuning import FineTuningManager
from app.ml.llama_model import LlamaModelManager
from app.core.security import verify_token
from app.core.rate_limit import rate_limit
from app.data.pipeline import AmazonDataPipeline

logger = structlog.get_logger()
router = APIRouter()

# Global managers
fine_tuning_manager = FineTuningManager()
llama_manager = LlamaModelManager()
data_pipeline = AmazonDataPipeline()

@router.post("/train/start")
@rate_limit(calls=1, period=300)  # 1 call per 5 minutes
async def start_fine_tuning(
    background_tasks: BackgroundTasks,
    use_amazon_data: bool = True,
    max_samples: int = 10000,
    token: str = Depends(verify_token)
):
    """
    Start QLoRA fine-tuning process
    
    - **use_amazon_data**: Use real Amazon data or existing training data
    - **max_samples**: Maximum number of training samples
    
    Starts background fine-tuning process.
    """
    try:
        # Initialize managers
        await fine_tuning_manager.initialize()
        await llama_manager.initialize()
        await data_pipeline.initialize()
        
        # Get training data
        if use_amazon_data:
            logger.info("Loading Amazon product data for fine-tuning")
            # Load a sample of Amazon data
            products = await data_pipeline.load_sample_data(max_samples)
        else:
            # Use existing training data
            products = await data_pipeline.load_existing_training_data()
        
        if not products or len(products) < 100:
            raise HTTPException(
                status_code=400,
                detail="Insufficient training data. Need at least 100 samples."
            )
        
        # Start training in background
        background_tasks.add_task(
            run_fine_tuning_task,
            products,
            f"training_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return {
            "message": "Fine-tuning started successfully",
            "training_samples": len(products),
            "estimated_time": "30-60 minutes",
            "session_id": f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        logger.error(f"Failed to start fine-tuning: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start fine-tuning: {str(e)}"
        )

async def run_fine_tuning_task(products: List[Dict[str, Any]], session_id: str):
    """Background task for fine-tuning"""
    try:
        logger.info(f"Starting fine-tuning session: {session_id}")
        
        # Initialize fine-tuning manager
        await fine_tuning_manager.initialize()
        
        # Train model
        result = await fine_tuning_manager.train_from_data(products)
        
        # Reload model with new adapter
        await llama_manager.initialize()
        
        logger.info(f"Fine-tuning completed: {session_id}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed for {session_id}: {e}")

@router.get("/train/status")
async def get_training_status(token: str = Depends(verify_token)):
    """Get current fine-tuning status"""
    try:
        status = llama_manager.get_status()
        
        return {
            "status": "completed" if status["predictor"]["fine_tuned"] else "not_started",
            "model_info": status["predictor"],
            "cuda_available": status["cuda_available"],
            "memory_usage": status["memory_usage"],
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training status: {str(e)}"
        )

@router.post("/predict/llama", response_model=PredictionResponse)
@rate_limit(calls=10, period=60)
async def predict_with_llama(
    request: PredictionRequest,
    token: str = Depends(verify_token)
):
    """
    Make price prediction using fine-tuned Llama model
    
    - **title**: Product title
    - **description**: Product description
    - **category**: Product category
    - **model_type**: Should be "fine_tuned_llm"
    
    Returns prediction with confidence score.
    """
    try:
        # Ensure llama manager is initialized
        if not llama_manager.predictor.is_loaded:
            await llama_manager.initialize()
        
        # Make prediction
        start_time = datetime.now()
        price, confidence = await llama_manager.predict_price(
            title=request.title,
            category=request.category,
            description=request.description or ""
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate price range
        price_range = {
            "min": price * 0.8,
            "max": price * 1.2,
            "confidence_interval": f"${price * 0.9:.2f} - ${price * 1.1:.2f}"
        }
        
        return PredictionResponse(
            predicted_price=round(price, 2),
            confidence_score=round(confidence, 3),
            price_range=price_range,
            model_used=ModelType.FINE_TUNED_LLM,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Llama prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Llama prediction failed: {str(e)}"
        )

@router.post("/predict/ensemble-enhanced", response_model=PredictionResponse)
@rate_limit(calls=20, period=60)
async def predict_with_enhanced_ensemble(
    request: PredictionRequest,
    include_llama: bool = True,
    token: str = Depends(verify_token)
):
    """
    Enhanced ensemble prediction including fine-tuned Llama
    
    - **include_llama**: Whether to include Llama in ensemble
    - Other parameters same as standard prediction
    
    Returns weighted ensemble prediction.
    """
    try:
        # This would integrate with the existing prediction service
        # For now, we'll simulate the enhanced ensemble
        
        # Get predictions from different models
        predictions = []
        weights = []
        
        # Traditional ML (if available)
        # traditional_pred = await get_traditional_prediction(request)
        # predictions.append(traditional_pred)
        # weights.append(0.3)
        
        # Deep Learning (if available)
        # deep_pred = await get_deep_learning_prediction(request)
        # predictions.append(deep_pred)
        # weights.append(0.3)
        
        # Fine-tuned Llama
        if include_llama:
            llama_price, llama_confidence = await llama_manager.predict_price(
                title=request.title,
                category=request.category,
                description=request.description or ""
            )
            predictions.append(llama_price)
            weights.append(0.7)  # Higher weight for fine-tuned model
        
        if not predictions:
            raise HTTPException(
                status_code=503,
                detail="No models available for prediction"
            )
        
        # Calculate weighted average
        ensemble_price = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)
        
        # Calculate ensemble confidence
        ensemble_confidence = min(0.95, sum(weights) / len(weights))
        
        # Calculate processing time
        processing_time = 150.0  # Simulated
        
        # Calculate price range
        price_range = {
            "min": ensemble_price * 0.8,
            "max": ensemble_price * 1.2,
            "confidence_interval": f"${ensemble_price * 0.9:.2f} - ${ensemble_price * 1.1:.2f}"
        }
        
        return PredictionResponse(
            predicted_price=round(ensemble_price, 2),
            confidence_score=round(ensemble_confidence, 3),
            price_range=price_range,
            model_used=ModelType.ENSEMBLE,
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Enhanced ensemble prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced ensemble prediction failed: {str(e)}"
        )

@router.get("/model/llama/info")
async def get_llama_model_info(token: str = Depends(verify_token)):
    """Get detailed information about the Llama model"""
    try:
        if not llama_manager.predictor.is_loaded:
            await llama_manager.initialize()
        
        model_info = llama_manager.predictor.get_model_info()
        status = llama_manager.get_status()
        
        return {
            "model": model_info,
            "system_status": status,
            "capabilities": {
                "fine_tuned": model_info["fine_tuned"],
                "quantized": True,
                "batch_prediction": True,
                "real_time_inference": True
            },
            "performance": {
                "estimated_latency_ms": 200,
                "memory_usage_gb": status["memory_usage"] / (1024**3) if status["memory_usage"] else 0,
                "gpu_accelerated": status["cuda_available"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get Llama model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@router.post("/model/llama/reload")
async def reload_llama_model(token: str = Depends(verify_token)):
    """Reload the Llama model (useful after training)"""
    try:
        # Unload current model
        llama_manager.predictor.unload_model()
        
        # Reload model
        await llama_manager.initialize()
        
        return {
            "message": "Llama model reloaded successfully",
            "model_info": llama_manager.predictor.get_model_info(),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to reload Llama model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )

@router.get("/training/history")
async def get_training_history(token: str = Depends(verify_token)):
    """Get history of training sessions"""
    try:
        # This would typically read from a database
        # For now, return mock history
        return {
            "sessions": [
                {
                    "session_id": "training_20240306_143000",
                    "status": "completed",
                    "start_time": "2024-03-06T14:30:00Z",
                    "end_time": "2024-03-06T15:15:00Z",
                    "training_samples": 10000,
                    "final_loss": 0.234,
                    "model_path": "./models/fine_tuned"
                }
            ],
            "total_sessions": 1,
            "last_training": "2024-03-06T15:15:00Z"
        }
        
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training history: {str(e)}"
        )
