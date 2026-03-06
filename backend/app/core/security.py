"""
Security utilities for SteadyPrice Enterprise
Authentication and authorization
"""

from datetime import datetime, timedelta
from typing import Optional, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token scheme
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        
        if email is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    return email

# Demo user for assignment
DEMO_USERS = {
    "admin@steadyprice.ai": {
        "email": "admin@steadyprice.ai",
        "name": "Admin User",
        "password": get_password_hash("SteadyPriceDemo2024!"),
        "is_active": True,
        "role": "admin"
    },
    "user@steadyprice.ai": {
        "email": "user@steadyprice.ai", 
        "name": "Demo User",
        "password": get_password_hash("SteadyPriceDemo2024!"),
        "is_active": True,
        "role": "user"
    }
}

def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate user credentials"""
    user = DEMO_USERS.get(email)
    
    if not user:
        return None
    
    if not verify_password(password, user["password"]):
        return None
    
    return user

def get_current_user(email: str = Depends(verify_token)) -> dict:
    """Get current authenticated user"""
    user = DEMO_USERS.get(email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user
