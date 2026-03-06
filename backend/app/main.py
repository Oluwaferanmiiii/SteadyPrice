"""
SteadyPrice Enterprise API
Transformative price prediction platform
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import uvicorn
import structlog

from app.core.config import settings
from app.core.security import verify_token
from app.api.v1.api import api_router
from app.database.connection import init_db
from app.ml.model_manager import ModelManager

# Structured logging
logger = structlog.get_logger()

# Global model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting SteadyPrice Enterprise API")
    await init_db()
    
    # Load ML models
    await model_manager.initialize_models()
    logger.info("All models loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SteadyPrice Enterprise API")

# Create FastAPI app
app = FastAPI(
    title="SteadyPrice Enterprise API",
    description="Transformative AI-powered price prediction platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "service": "SteadyPrice Enterprise API",
        "version": "2.0.0",
        "models_loaded": model_manager.is_ready()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SteadyPrice Enterprise API",
        "description": "Transformative AI-powered price prediction platform",
        "docs": "/docs",
        "health": "/health"
    }

# Protected endpoint example
@app.get("/api/v1/protected")
async def protected_endpoint(token: str = Depends(verify_token)):
    """Example of a protected endpoint"""
    return {"message": "This is a protected endpoint", "token_valid": True}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
