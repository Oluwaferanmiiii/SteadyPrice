"""
Run script for SteadyPrice Enterprise Backend
"""

import uvicorn
import asyncio
import os
from pathlib import Path

from app.main import app
from app.core.config import settings

# Add project root to Python path
project_root = Path(__file__).parent
os.chdir(project_root)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
        access_log=True
    )
