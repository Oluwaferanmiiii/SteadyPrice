"""
Authentication endpoints for SteadyPrice Enterprise
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta
import structlog

from app.core.config import settings
from app.core.security import authenticate_user, create_access_token, get_current_user
from app.models.schemas import Token, User

logger = structlog.get_logger()
router = APIRouter()

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and return access token
    
    - **username**: Email address
    - **password**: User password
    
    Returns JWT token for authenticated requests.
    """
    try:
        # Authenticate user
        user = authenticate_user(form_data.username, form_data.password)
        
        if not user:
            logger.warning(f"Authentication failed for {form_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["email"]}, 
            expires_delta=access_token_expires
        )
        
        logger.info(f"User authenticated: {user['email']}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )

@router.get("/me", response_model=User)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Get current user information
    
    Returns the authenticated user's profile information.
    """
    return User(
        id=current_user["email"],
        email=current_user["email"],
        name=current_user["name"],
        is_active=current_user["is_active"]
    )

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout user (token invalidation would be handled by client)
    
    For demo purposes - in production, you'd implement token blacklisting.
    """
    logger.info(f"User logged out: {current_user['email']}")
    return {"message": "Successfully logged out"}

@router.get("/verify")
async def verify_token_endpoint(current_user: dict = Depends(get_current_user)):
    """
    Verify token validity
    
    Returns user information if token is valid.
    """
    return {
        "valid": True,
        "user": {
            "email": current_user["email"],
            "name": current_user["name"],
            "role": current_user["role"]
        }
    }
