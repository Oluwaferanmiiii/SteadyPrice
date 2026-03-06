"""
Main API router for SteadyPrice Enterprise
"""

from fastapi import APIRouter
from app.api.v1.endpoints import predictions, auth

api_router = APIRouter()

# Include prediction endpoints
api_router.include_router(
    predictions.router, 
    prefix="/predictions", 
    tags=["predictions"]
)

# Include authentication endpoints
api_router.include_router(
    auth.router, 
    prefix="/auth", 
    tags=["authentication"]
)
