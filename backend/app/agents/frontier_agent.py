"""
FrontierAgent - Premium Model Integration

This agent integrates with frontier models (Claude 4.5 Sonnet, GPT 4.1 Nano)
for high-accuracy price prediction and validation.
"""

import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import os
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentRequest, AgentResponse
from ..core.config import get_settings

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for frontier models"""
    name: str
    provider: str
    api_endpoint: str
    model_id: str
    max_tokens: int
    temperature: float
    known_mae: float
    cost_per_1k_tokens: float
    max_concurrent: int

class FrontierAgent(BaseAgent):
    """
    FrontierAgent that integrates with premium models for high-accuracy
    price prediction and validation.
    
    Capabilities:
    - Claude 4.5 Sonnet ($47.10 MAE)
    - GPT 4.1 Nano ($62.51 MAE)
    - Smart routing based on query complexity
    - Cost optimization through intelligent model selection
    """
    
    def __init__(self):
        # Define agent capabilities
        capability = AgentCapability(
            name="Frontier Model Agent",
            description="Premium model integration for high-accuracy price prediction",
            max_concurrent_tasks=20,
            average_response_time=0.8,  # 800ms average
            accuracy_metric=0.91,  # 91% accuracy
            cost_per_request=0.05,  # Higher cost for premium models
            supported_categories=["Electronics", "Appliances", "Automotive", "Furniture", "Clothing", "Books", "Sports", "Home", "Beauty", "Toys"]
        )
        
        super().__init__(AgentType.FRONTIER, capability)
        
        # Frontier model configurations
        self.models = {
            "claude_4_5_sonnet": ModelConfig(
                name="Claude 4.5 Sonnet",
                provider="anthropic",
                api_endpoint="https://api.anthropic.com/v1/messages",
                model_id="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0.1,
                known_mae=47.10,
                cost_per_1k_tokens=0.015,
                max_concurrent=10
            ),
            "gpt_4_1_nano": ModelConfig(
                name="GPT 4.1 Nano",
                provider="openai",
                api_endpoint="https://api.openai.com/v1/chat/completions",
                model_id="gpt-4.1-nano",
                max_tokens=100,
                temperature=0.1,
                known_mae=62.51,
                cost_per_1k_tokens=0.15,
                max_concurrent=15
            )
        }
        
        # API keys and session
        self.settings = get_settings()
        self.session: Optional[aiohttp.ClientSession] = None
        self.model_performance = {
            "claude_4_5_sonnet": {"requests": 0, "total_cost": 0.0, "avg_response_time": 0.0},
            "gpt_4_1_nano": {"requests": 0, "total_cost": 0.0, "avg_response_time": 0.0}
        }
        
        # Smart routing logic
        self.routing_rules = {
            "high_value": {"threshold": 500.0, "preferred_model": "claude_4_5_sonnet"},
            "complex_description": {"length_threshold": 200, "preferred_model": "claude_4_5_sonnet"},
            "cost_sensitive": {"budget_threshold": 0.02, "preferred_model": "gpt_4_1_nano"}
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the FrontierAgent with API connections.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing FrontierAgent with premium model APIs...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0),
                headers={"User-Agent": "SteadyPrice-FrontierAgent/1.0"}
            )
            
            # Validate API keys
            anthropic_key = self.settings.ANTHROPIC_API_KEY
            openai_key = self.settings.OPENAI_API_KEY
            
            if not anthropic_key and not openai_key:
                logger.error("No API keys found for frontier models")
                return False
            
            # Test API connections
            claude_available = await self._test_model_connection("claude_4_5_sonnet", anthropic_key)
            gpt_available = await self._test_model_connection("gpt_4_1_nano", openai_key)
            
            if not claude_available and not gpt_available:
                logger.error("No frontier models are available")
                return False
            
            logger.info(f"FrontierAgent initialized - Claude: {claude_available}, GPT: {gpt_available}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing FrontierAgent: {e}")
            return False
    
    async def _test_model_connection(self, model_key: str, api_key: str) -> bool:
        """Test connection to a specific model API."""
        try:
            model_config = self.models[model_key]
            
            # Create a simple test request
            test_payload = self._create_payload(model_config, "Test product", "Electronics", "Test description")
            
            headers = self._get_headers(model_config, api_key)
            
            async with self.session.post(
                model_config.api_endpoint,
                json=test_payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info(f"Successfully connected to {model_config.name}")
                    return True
                else:
                    logger.warning(f"Failed to connect to {model_config.name}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error testing {model_key} connection: {e}")
            return False
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a price prediction request using frontier models.
        
        Args:
            request: Agent request with product information
            
        Returns:
            AgentResponse with price prediction and confidence
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract product information
            product_data = request.payload.get('product', {})
            title = product_data.get('title', '')
            category = product_data.get('category', '')
            description = product_data.get('description', '')
            
            # Route to optimal model
            selected_model = self._route_request(title, category, description)
            model_config = self.models[selected_model]
            
            # Generate prediction
            prediction_data = await self._predict_with_model(
                selected_model, title, category, description
            )
            
            # Calculate confidence based on model performance
            confidence = self._calculate_confidence(selected_model, prediction_data['predicted_price'])
            
            # Update model performance metrics
            self._update_model_performance(selected_model, prediction_data)
            
            response = AgentResponse(
                request_id=request.request_id,
                agent_type=self.agent_type,
                status="success",
                data={
                    "predicted_price": prediction_data['predicted_price'],
                    "confidence": confidence,
                    "category": category,
                    "model_used": model_config.name,
                    "model_provider": model_config.provider,
                    "processing_details": {
                        "known_mae": model_config.known_mae,
                        "cost_estimate": prediction_data.get('cost', 0.0),
                        "response_time": prediction_data.get('response_time', 0.0)
                    }
                },
                confidence=confidence,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            logger.info(f"FrontierAgent prediction: ${prediction_data['predicted_price']:.2f} using {model_config.name}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            raise
    
    def _route_request(self, title: str, category: str, description: str) -> str:
        """
        Route request to the optimal model based on routing rules.
        
        Args:
            title: Product title
            category: Product category
            description: Product description
            
        Returns:
            Selected model key
        """
        # Check for high-value items (prefer Claude for accuracy)
        if self._estimate_price_range(title, description) > self.routing_rules["high_value"]["threshold"]:
            return self.routing_rules["high_value"]["preferred_model"]
        
        # Check for complex descriptions (prefer Claude for understanding)
        if len(description) > self.routing_rules["complex_description"]["length_threshold"]:
            return self.routing_rules["complex_description"]["preferred_model"]
        
        # Default to GPT for cost efficiency
        return self.routing_rules["cost_sensitive"]["preferred_model"]
    
    def _estimate_price_range(self, title: str, description: str) -> float:
        """Estimate price range for routing decisions."""
        # Simple heuristic based on keywords
        high_value_keywords = ["premium", "pro", "professional", "luxury", "advanced", "4k", "8k", "gaming"]
        low_value_keywords = ["basic", "entry", "budget", "compact", "mini"]
        
        text = (title + " " + description).lower()
        
        if any(keyword in text for keyword in high_value_keywords):
            return 1000.0  # High estimate
        elif any(keyword in text for keyword in low_value_keywords):
            return 50.0   # Low estimate
        else:
            return 200.0  # Medium estimate
    
    async def _predict_with_model(self, model_key: str, title: str, category: str, description: str) -> Dict[str, Any]:
        """
        Predict price using a specific frontier model.
        
        Args:
            model_key: Key of the model to use
            title: Product title
            category: Product category
            description: Product description
            
        Returns:
            Dictionary with prediction and metadata
        """
        model_config = self.models[model_key]
        start_time = datetime.utcnow()
        
        try:
            # Create payload
            payload = self._create_payload(model_config, title, category, description)
            
            # Get API key
            api_key = self._get_api_key(model_config.provider)
            headers = self._get_headers(model_config, api_key)
            
            # Make API request
            async with self.session.post(
                model_config.api_endpoint,
                json=payload,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")
                
                response_data = await response.json()
                predicted_price = self._extract_price_from_response(response_data, model_config.provider)
                
                # Calculate cost
                response_time = (datetime.utcnow() - start_time).total_seconds()
                estimated_tokens = len(payload.get("prompt", "")) / 4  # Rough estimate
                cost = (estimated_tokens / 1000) * model_config.cost_per_1k_tokens
                
                return {
                    "predicted_price": predicted_price,
                    "response_time": response_time,
                    "cost": cost,
                    "tokens_used": estimated_tokens
                }
                
        except Exception as e:
            logger.error(f"Error predicting with {model_config.name}: {e}")
            raise
    
    def _create_payload(self, model_config: ModelConfig, title: str, category: str, description: str) -> Dict[str, Any]:
        """Create API payload for the specific model."""
        prompt = f"""You are an expert pricing analyst. Please predict the price of the following product:

Product: {title}
Category: {category}
Description: {description}

Please respond with only a numerical price in USD (e.g., 299.99). Do not include any explanation or additional text."""

        if model_config.provider == "anthropic":
            return {
                "model": model_config.model_id,
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
        elif model_config.provider == "openai":
            return {
                "model": model_config.model_id,
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    def _get_headers(self, model_config: ModelConfig, api_key: str) -> Dict[str, str]:
        """Get API headers for the specific model."""
        if model_config.provider == "anthropic":
            return {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
        elif model_config.provider == "openai":
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    def _get_api_key(self, provider: str) -> str:
        """Get API key for the specific provider."""
        if provider == "anthropic":
            return self.settings.ANTHROPIC_API_KEY
        elif provider == "openai":
            return self.settings.OPENAI_API_KEY
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _extract_price_from_response(self, response_data: Dict[str, Any], provider: str) -> float:
        """Extract price from model response."""
        try:
            if provider == "anthropic":
                content = response_data.get("content", [{}])[0].get("text", "")
            elif provider == "openai":
                content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Extract numerical price
            import re
            price_match = re.search(r'\$?(\d+(?:\.\d{2})?)', content.strip())
            
            if price_match:
                return float(price_match.group(1))
            else:
                raise ValueError(f"Could not extract price from response: {content}")
                
        except Exception as e:
            logger.error(f"Error extracting price from {provider} response: {e}")
            raise
    
    def _calculate_confidence(self, model_key: str, predicted_price: float) -> float:
        """Calculate confidence based on model performance."""
        model_config = self.models[model_key]
        
        # Base confidence from known MAE
        base_confidence = max(0.7, min(0.95, 1.0 - (model_config.known_mae / 100.0)))
        
        # Adjust for price range
        if predicted_price < 50:
            price_factor = 0.85
        elif predicted_price > 1000:
            price_factor = 0.8
        else:
            price_factor = 1.0
        
        return min(0.95, base_confidence * price_factor)
    
    def _update_model_performance(self, model_key: str, prediction_data: Dict[str, Any]):
        """Update performance metrics for a model."""
        if model_key in self.model_performance:
            perf = self.model_performance[model_key]
            perf["requests"] += 1
            perf["total_cost"] += prediction_data.get("cost", 0.0)
            
            # Update average response time
            total_requests = perf["requests"]
            current_avg = perf["avg_response_time"]
            new_response_time = prediction_data.get("response_time", 0.0)
            perf["avg_response_time"] = ((current_avg * (total_requests - 1)) + new_response_time) / total_requests
    
    async def health_check(self) -> bool:
        """Check if the FrontierAgent is healthy."""
        try:
            if not self.session:
                return False
            
            # Test with the most reliable model
            for model_key in ["claude_4_5_sonnet", "gpt_4_1_nano"]:
                if self._get_api_key(self.models[model_key].provider):
                    try:
                        test_request = AgentRequest(
                            request_id="health_check",
                            agent_type=self.agent_type,
                            task_type="price_prediction",
                            payload={
                                "product": {
                                    "title": "Test Product",
                                    "category": "Electronics",
                                    "description": "Test description"
                                }
                            }
                        )
                        
                        response = await self.process_request(test_request)
                        return response.status == "success"
                    except:
                        continue
            
            return False
            
        except Exception as e:
            logger.error(f"FrontierAgent health check failed: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "available_models": list(self.models.keys()),
            "model_performance": self.model_performance,
            "routing_rules": self.routing_rules,
            "supported_categories": self.capability.supported_categories,
            "total_requests": sum(perf["requests"] for perf in self.model_performance.values()),
            "total_cost": sum(perf["total_cost"] for perf in self.model_performance.values()),
            "current_status": self.status.value
        }
    
    async def shutdown(self):
        """Gracefully shutdown the FrontierAgent."""
        await super().shutdown()
        
        if self.session:
            await self.session.close()
            logger.info("FrontierAgent shutdown complete")
