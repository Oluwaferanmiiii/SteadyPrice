"""
Configuration settings for SteadyPrice Enterprise
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "SteadyPrice Enterprise"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Database
    DATABASE_URL: str = "sqlite:///./steadyprice.db"  # Default to SQLite for security
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1 hour
    
    # Security
    SECRET_KEY: str = "dev-secret-key-change-in-production"  # Override with env var in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 1000
    RATE_LIMIT_BURST: int = 100
    
    # Machine Learning
    MODEL_CACHE_DIR: str = "./models"
    MAX_PREDICTION_BATCH_SIZE: int = 100
    MODEL_CONFIDENCE_THRESHOLD: float = 0.7
    
    # HuggingFace
    HF_TOKEN: Optional[str] = None
    HF_MODEL_NAME: str = "microsoft/DialoGPT-medium"
    
    # Monitoring
    PROMETHEUS_PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    
    # External APIs
    AMAZON_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # File Storage
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Environment-specific overrides
if settings.ENVIRONMENT == "production":
    settings.DEBUG = False
    settings.LOG_LEVEL = "WARNING"
elif settings.ENVIRONMENT == "development":
    settings.DEBUG = True
    settings.LOG_LEVEL = "DEBUG"
