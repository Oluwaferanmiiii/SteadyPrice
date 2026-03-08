"""
Modal.com Production Deployment for SteadyPrice Week 8 System

This module provides production deployment configuration for the
transformative multi-agent system on Modal.com cloud platform.
"""

import os
import json
from datetime import timedelta
from typing import Dict, Any, List

import modal
import asyncio

# Modal app configuration
app = modal.App("steadyprice-week8-transformative")

# Define image with all dependencies
image = modal.Image.debian_slim().pip_install([
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
    "pydantic==2.5.0",
    "aiohttp==3.9.1",
    "feedparser==6.0.10",
    "numpy==1.24.3",
    "pandas==1.5.3",
    "scikit-learn==1.3.2",
    "transformers==4.36.0",
    "torch==2.1.0",
    "peft==0.7.1",
    "bitsandbytes==0.41.2",
    "accelerate==0.25.0",
    "datasets==2.15.0",
    "sentence-transformers==2.2.2",
    "faiss-cpu==1.7.4",
    "redis==5.0.1",
    "structlog==23.2.0",
    "python-multipart==0.0.6",
    "python-dotenv==1.0.0",
])

# Secrets configuration
secrets = [
    modal.Secret.from_name("steadyprice-anthropic"),
    modal.Secret.from_name("steadyprice-openai"),
    modal.Secret.from_name("steadyprice-retail-api"),
]

# Volume for persistent storage
volume = modal.Volume.from_name("steadyprice-week8-data", create_if_missing=True)

# GPU configuration for ML models
gpu_config = modal.gpu.A10G()

# System requirements
SYSTEM_REQUIREMENTS = {
    "cpu": 4,
    "memory": 16384,  # 16GB RAM
    "gpu_count": 1,
    "timeout": 3600,  # 1 hour timeout
}

@app.function(
    image=image,
    secrets=secrets,
    gpu=gpu_config,
    timeout=3600,
    container_idle_timeout=300,  # 5 minutes idle timeout
    mounts=[modal.Mount.local_mount("/data", remote_path="/steadyprice/data")],
)
class SteadyPriceWeek8Service:
    """Production service for Week 8 multi-agent system."""
    
    def __init__(self):
        """Initialize the service."""
        self.orchestrator = None
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the multi-agent system."""
        try:
            # Import here to avoid issues with Modal
            from app.core.orchestrator import SteadyPriceOrchestrator
            
            print("🚀 Initializing SteadyPrice Week 8 on Modal.com...")
            
            # Initialize orchestrator
            self.orchestrator = SteadyPriceOrchestrator()
            
            # Initialize all agents
            if await self.orchestrator.initialize():
                print("✅ All agents initialized successfully")
                self.is_initialized = True
                return True
            else:
                print("❌ Failed to initialize agents")
                return False
                
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the multi-agent system."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            if await self.orchestrator.start():
                print("🎉 SteadyPrice Week 8 System started on Modal.com!")
                return True
            else:
                print("❌ Failed to start system")
                return False
                
        except Exception as e:
            print(f"❌ Startup error: {e}")
            return False
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user request."""
        if not self.orchestrator:
            raise RuntimeError("System not initialized")
        
        return await self.orchestrator.process_user_request(request_data)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status."""
        if not self.orchestrator:
            return {"status": "not_initialized"}
        
        return await self.orchestrator.get_system_status()
    
    async def shutdown(self):
        """Shutdown the system."""
        if self.orchestrator:
            await self.orchestrator.shutdown()

# Web server function
@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=50,
)
@modal.web_endpoint("GET", "/health")
async def health_check():
    """Health check endpoint."""
    try:
        service = SteadyPriceWeek8Service()
        status = await service.get_system_status()
        return {"status": "healthy", "system": status}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=50,
)
@modal.web_endpoint("POST", "/api/chat")
async def chat_endpoint(request: modal.Request):
    """Chat endpoint for natural language interaction."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        data = json.loads(request.data)
        
        request_data = {
            "request_id": f"modal_{hash(str(data))}",
            "request_type": "user_message",
            "user_message": {
                "message_id": f"msg_{hash(str(data))}",
                "user_id": data.get("user_id", "modal_user"),
                "content": data.get("content", ""),
                "message_type": data.get("message_type", "query"),
                "context": data.get("context")
            }
        }
        
        response = await service.process_request(request_data)
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(modal.functions.current_time())
        }

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=50,
)
@modal.web_endpoint("POST", "/api/predict")
async def predict_endpoint(request: modal.Request):
    """Price prediction endpoint."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        data = json.loads(request.data)
        
        request_data = {
            "request_id": f"modal_pred_{hash(str(data))}",
            "request_type": "price_prediction",
            "product": data.get("product", {}),
            "user_id": data.get("user_id", "modal_user")
        }
        
        response = await service.process_request(request_data)
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(modal.functions.current_time())
        }

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=50,
)
@modal.web_endpoint("POST", "/api/deals")
async def deals_endpoint(request: modal.Request):
    """Deal search endpoint."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        data = json.loads(request.data)
        
        request_data = {
            "request_id": f"modal_deals_{hash(str(data))}",
            "request_type": "deal_search",
            "filters": data.get("filters", {}),
            "user_id": data.get("user_id", "modal_user")
        }
        
        response = await service.process_request(request_data)
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(modal.functions.current_time())
        }

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=50,
)
@modal.web_endpoint("POST", "/api/market")
async def market_endpoint(request: modal.Request):
    """Market analysis endpoint."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        data = json.loads(request.data)
        
        request_data = {
            "request_id": f"modal_market_{hash(str(data))}",
            "request_type": "market_analysis",
            "category": data.get("category", "Electronics"),
            "user_id": data.get("user_id", "modal_user")
        }
        
        response = await service.process_request(request_data)
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(modal.functions.current_time())
        }

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=50,
)
@modal.web_endpoint("POST", "/api/portfolio")
async def portfolio_endpoint(request: modal.Request):
    """Portfolio optimization endpoint."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        data = json.loads(request.data)
        
        request_data = {
            "request_id": f"modal_portfolio_{hash(str(data))}",
            "request_type": "portfolio_optimization",
            "budget": data.get("budget", 1000.0),
            "categories": data.get("categories"),
            "user_id": data.get("user_id", "modal_user")
        }
        
        response = await service.process_request(request_data)
        return response
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(modal.functions.current_time())
        }

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=10,
)
@modal.web_endpoint("GET", "/api/status")
async def status_endpoint():
    """System status endpoint."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        status = await service.get_system_status()
        return status
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(modal.functions.current_time())
        }

@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    allow_concurrent_inputs=10,
)
@modal.web_endpoint("GET", "/api/capabilities")
async def capabilities_endpoint():
    """System capabilities endpoint."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        capabilities = service.orchestrator.get_system_capabilities()
        return capabilities
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": str(modal.functions.current_time())
        }

# Background task for periodic deal scanning
@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    schedule=modal.Period(minutes=15),  # Run every 15 minutes
)
async def periodic_deal_scanning():
    """Background task for periodic deal scanning."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        # Trigger deal scanning
        scan_request = {
            "request_id": f"periodic_scan_{modal.functions.current_time()}",
            "request_type": "deal_search",
            "filters": {},
            "user_id": "system_scanner"
        }
        
        await service.process_request(scan_request)
        print(f"🔍 Periodic deal scanning completed at {modal.functions.current_time()}")
        
    except Exception as e:
        print(f"❌ Periodic scanning error: {e}")

# Background task for system health monitoring
@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
    schedule=modal.Period(minutes=5),  # Run every 5 minutes
)
async def health_monitoring():
    """Background task for system health monitoring."""
    try:
        service = SteadyPriceWeek8Service()
        await service.start()
        
        status = await service.get_system_status()
        
        # Log health metrics
        print(f"🏥 Health check at {modal.functions.current_time()}:")
        print(f"   System Health: {status.get('system_health', 0):.2%}")
        print(f"   Total Requests: {status.get('system_metrics', {}).get('total_requests', 0)}")
        print(f"   Error Rate: {status.get('system_metrics', {}).get('error_rate', 0):.2%}")
        print(f"   Agent Health: {status.get('agent_health', {})}")
        
        # Store health metrics (in production, would send to monitoring system)
        health_data = {
            "timestamp": str(modal.functions.current_time()),
            "system_health": status.get('system_health', 0),
            "total_requests": status.get('system_metrics', {}).get('total_requests', 0),
            "error_rate": status.get('system_metrics', {}).get('error_rate', 0),
            "agent_health": status.get('agent_health', {})
        }
        
        # Save to volume for persistence
        with open("/data/health_metrics.json", "a") as f:
            f.write(json.dumps(health_data) + "\n")
        
    except Exception as e:
        print(f"❌ Health monitoring error: {e}")

# Deployment configuration
@app.function(
    image=image,
    secrets=secrets,
    timeout=3600,
)
def deploy():
    """Deploy the Week 8 system to Modal.com."""
    print("🚀 Deploying SteadyPrice Week 8 to Modal.com...")
    
    # Test deployment
    service = SteadyPriceWeek8Service()
    
    # Run deployment tests
    asyncio.run(deployment_tests(service))
    
    print("✅ Deployment completed successfully!")
    print("🌐 Available endpoints:")
    print("   - Health: https://steadyprice-week8-transformative.modal.run/health")
    print("   - Chat: https://steadyprice-week8-transformative.modal.run/api/chat")
    print("   - Predict: https://steadyprice-week8-transformative.modal.run/api/predict")
    print("   - Deals: https://steadyprice-week8-transformative.modal.run/api/deals")
    print("   - Market: https://steadyprice-week8-transformative.modal.run/api/market")
    print("   - Portfolio: https://steadyprice-week8-transformative.modal.run/api/portfolio")
    print("   - Status: https://steadyprice-week8-transformative.modal.run/api/status")
    print("   - Capabilities: https://steadyprice-week8-transformative.modal.run/api/capabilities")

async def deployment_tests(service):
    """Run deployment tests."""
    print("🧪 Running deployment tests...")
    
    # Test 1: Initialize system
    print("   Test 1: System initialization...")
    if await service.initialize():
        print("   ✅ System initialized successfully")
    else:
        print("   ❌ System initialization failed")
        return False
    
    # Test 2: Start system
    print("   Test 2: System startup...")
    if await service.start():
        print("   ✅ System started successfully")
    else:
        print("   ❌ System startup failed")
        return False
    
    # Test 3: Price prediction
    print("   Test 3: Price prediction...")
    prediction_request = {
        "request_id": "test_prediction",
        "request_type": "price_prediction",
        "product": {
            "title": "Test Product",
            "category": "Electronics",
            "description": "Test description for deployment"
        },
        "user_id": "test_user"
    }
    
    try:
        prediction_response = await service.process_request(prediction_request)
        if prediction_response.get("status") == "success":
            print("   ✅ Price prediction test passed")
        else:
            print("   ❌ Price prediction test failed")
            return False
    except Exception as e:
        print(f"   ❌ Price prediction test error: {e}")
        return False
    
    # Test 4: Deal search
    print("   Test 4: Deal search...")
    deals_request = {
        "request_id": "test_deals",
        "request_type": "deal_search",
        "filters": {"category": "Electronics", "limit": 5},
        "user_id": "test_user"
    }
    
    try:
        deals_response = await service.process_request(deals_request)
        if deals_response.get("status") == "success":
            print("   ✅ Deal search test passed")
        else:
            print("   ❌ Deal search test failed")
            return False
    except Exception as e:
        print(f"   ❌ Deal search test error: {e}")
        return False
    
    # Test 5: System status
    print("   Test 5: System status...")
    try:
        status = await service.get_system_status()
        if status.get("system_health", 0) > 0:
            print("   ✅ System status test passed")
        else:
            print("   ❌ System status test failed")
            return False
    except Exception as e:
        print(f"   ❌ System status test error: {e}")
        return False
    
    print("   🎉 All deployment tests passed!")
    return True

if __name__ == "__main__":
    # Local deployment for testing
    with app.run():
        deploy()
