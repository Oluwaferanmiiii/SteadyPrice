"""
Simplified test version of SteadyPrice API
For production testing without complex dependencies
"""

from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn
import os
import random

# Load environment
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="SteadyPrice Enterprise API - Test",
    description="Transformative AI-powered price prediction platform",
    version="2.0.0-test"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PredictionRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    category: str
    model_type: str = "ensemble"

class PriceRange(BaseModel):
    min: float
    max: float
    confidence_interval: str

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_score: float
    price_range: PriceRange
    model_used: str
    processing_time_ms: float
    timestamp: datetime

class BatchPredictionRequest(BaseModel):
    products: List[PredictionRequest]
    model_type: str = "ensemble"

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processing_time_ms: float
    success_count: int
    error_count: int

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class HealthCheck(BaseModel):
    status: str
    service: str
    version: str
    models_loaded: bool
    timestamp: datetime

# Demo data
DEMO_USERS = {
    "admin@steadyprice.ai": "demo123",
    "user@steadyprice.ai": "demo123"
}

# Mock prediction function
def mock_predict_price(title: str, category: str) -> tuple:
    """Mock price prediction for testing"""
    # Base price by category
    base_prices = {
        "Electronics": 299.99,
        "Appliances": 199.99,
        "Automotive": 499.99,
        "Office_Products": 89.99,
        "Tools_and_Home_Improvement": 149.99,
        "Cell_Phones_and_Accessories": 399.99,
        "Toys_and_Games": 49.99,
        "Musical_Instruments": 299.99
    }
    
    base = base_prices.get(category, 199.99)
    
    # Add some variation based on title length
    variation = len(title) * 0.5 + random.uniform(-50, 50)
    price = max(1.0, base + variation)
    
    # Mock confidence
    confidence = random.uniform(0.75, 0.95)
    
    return price, confidence

# Endpoints
@app.get("/")
async def root():
    return {
        "message": "SteadyPrice Enterprise API - Test Mode",
        "description": "Transformative AI-powered price prediction platform",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return HealthCheck(
        status="healthy",
        service="SteadyPrice Enterprise API",
        version="2.0.0-test",
        models_loaded=True,
        timestamp=datetime.utcnow()
    )

@app.post("/api/v1/auth/token", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    """Demo authentication"""
    if username in DEMO_USERS and DEMO_USERS[username] == password:
        return Token(
            access_token="demo-jwt-token-for-testing",
            token_type="bearer",
            expires_in=1800
        )
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/v1/predictions/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Price prediction endpoint"""
    start_time = datetime.now()
    
    # Mock prediction
    price, confidence = mock_predict_price(request.title, request.category)
    
    # Calculate price range
    price_range = PriceRange(
        min=price * 0.8,
        max=price * 1.2,
        confidence_interval=f"${price * 0.9:.2f} - ${price * 1.1:.2f}"
    )
    
    # Calculate processing time
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return PredictionResponse(
        predicted_price=round(price, 2),
        confidence_score=round(confidence, 3),
        price_range=price_range,
        model_used=request.model_type,
        processing_time_ms=processing_time,
        timestamp=datetime.utcnow()
    )

@app.post("/api/v1/predictions/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch price prediction endpoint"""
    start_time = datetime.now()
    predictions = []
    
    for product in request.products:
        try:
            # Mock prediction
            price, confidence = mock_predict_price(product.title, product.category)
            
            # Calculate price range
            price_range = PriceRange(
                min=price * 0.8,
                max=price * 1.2,
                confidence_interval=f"${price * 0.9:.2f} - ${price * 1.1:.2f}"
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            prediction = PredictionResponse(
                predicted_price=round(price, 2),
                confidence_score=round(confidence, 3),
                price_range=price_range,
                model_used=request.model_type,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            predictions.append(prediction)
        except Exception as e:
            # Could add error handling here
            pass
    
    total_processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_processing_time_ms=total_processing_time,
        success_count=len(predictions),
        error_count=len(request.products) - len(predictions)
    )

@app.get("/api/v1/predictions/models/ensemble/metrics")
async def get_model_metrics():
    """Mock model metrics"""
    return {
        "model_type": "ensemble",
        "accuracy": 0.89,
        "mae": 12.5,
        "rmse": 18.75,
        "mape": 14.2,
        "r2_score": 0.82,
        "last_updated": datetime.utcnow()
    }

@app.get("/api/v1/predictions/models/compare")
async def compare_models():
    """Mock model comparison"""
    return {
        "traditional_ml": {
            "accuracy": 0.85,
            "mae": 15.2,
            "rmse": 22.1,
            "mape": 16.8,
            "r2_score": 0.78
        },
        "deep_learning": {
            "accuracy": 0.87,
            "mae": 13.8,
            "rmse": 20.3,
            "mape": 15.1,
            "r2_score": 0.81
        },
        "fine_tuned_llm": {
            "accuracy": 0.88,
            "mae": 12.9,
            "rmse": 19.2,
            "mape": 14.6,
            "r2_score": 0.83
        },
        "ensemble": {
            "accuracy": 0.91,
            "mae": 11.5,
            "rmse": 17.1,
            "mape": 13.2,
            "r2_score": 0.86
        }
    }

if __name__ == "__main__":
    print("🚀 Starting SteadyPrice Enterprise API - Test Mode")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("🎯 Ready for production testing!")
    
    uvicorn.run(
        "test_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
