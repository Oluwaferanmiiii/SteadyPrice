"""
Rate limiting for SteadyPrice Enterprise API
"""

import time
import asyncio
from typing import Dict, Callable
from functools import wraps
from fastapi import HTTPException, status
import structlog

logger = structlog.get_logger()

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, key: str, limit: int, period: int) -> bool:
        """Check if request is allowed"""
        now = time.time()
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                req_time for req_time in self.requests[key] 
                if now - req_time < period
            ]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= limit:
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True

# Global rate limiter instance
rate_limiter = RateLimiter()

def rate_limit(calls: int, period: int):
    """Rate limiting decorator"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Simple key generation (in production, use client IP/user ID)
            key = f"{func.__name__}_{int(time.time() // period)}"
            
            if not rate_limiter.is_allowed(key, calls, period):
                logger.warning(f"Rate limit exceeded for {func.__name__}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {calls} calls per {period} seconds"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
