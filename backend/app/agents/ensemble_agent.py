"""
EnsembleAgent - Multi-Model Fusion

This agent combines predictions from multiple models (Specialist, Frontier)
using intelligent ensemble techniques to achieve superior accuracy.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import json

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentRequest, AgentResponse
from .specialist_agent import SpecialistAgent
from .frontier_agent import FrontierAgent

logger = logging.getLogger(__name__)

class EnsembleMethod(Enum):
    """Different ensemble methods for combining predictions"""
    WEIGHTED_AVERAGE = "weighted_average"
    DYNAMIC_WEIGHTING = "dynamic_weighting"
    CONFIDENCE_BASED = "confidence_based"
    STACKING = "stacking"
    BAYESIAN = "bayesian"

@dataclass
class ModelPrediction:
    """Individual model prediction with metadata"""
    model_name: str
    predicted_price: float
    confidence: float
    response_time: float
    cost: float
    historical_mae: float
    category_performance: Dict[str, float]

@dataclass
class EnsembleWeights:
    """Weights for different models in ensemble"""
    specialist_weight: float = 0.4
    claude_weight: float = 0.35
    gpt_weight: float = 0.25
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = self.specialist_weight + self.claude_weight + self.gpt_weight
        if total > 0:
            self.specialist_weight /= total
            self.claude_weight /= total
            self.gpt_weight /= total

class EnsembleAgent(BaseAgent):
    """
    EnsembleAgent that combines predictions from multiple models
    using intelligent fusion techniques.
    
    Capabilities:
    - Weighted ensemble of Specialist and Frontier models
    - Target: <$35 MAE through intelligent fusion
    - Dynamic weight adjustment based on performance
    - Confidence scoring and uncertainty quantification
    """
    
    def __init__(self):
        # Define agent capabilities
        capability = AgentCapability(
            name="Ensemble Agent",
            description="Multi-model fusion for superior price prediction accuracy",
            max_concurrent_tasks=15,
            average_response_time=0.6,  # 600ms average (parallel processing)
            accuracy_metric=0.96,  # Target 96% accuracy
            cost_per_request=0.03,  # Moderate cost through optimization
            supported_categories=["Electronics", "Appliances", "Automotive", "Furniture", "Clothing", "Books", "Sports", "Home", "Beauty", "Toys"]
        )
        
        super().__init__(AgentType.ENSEMBLE, capability)
        
        # Component agents
        self.specialist_agent: Optional[SpecialistAgent] = None
        self.frontier_agent: Optional[FrontierAgent] = None
        
        # Ensemble configuration
        self.ensemble_method = EnsembleMethod.DYNAMIC_WEIGHTING
        self.weights = EnsembleWeights()
        self.category_weights = {
            "Electronics": EnsembleWeights(specialist_weight=0.5, claude_weight=0.3, gpt_weight=0.2),
            "Appliances": EnsembleWeights(specialist_weight=0.4, claude_weight=0.35, gpt_weight=0.25),
            "Automotive": EnsembleWeights(specialist_weight=0.45, claude_weight=0.3, gpt_weight=0.25),
            "Furniture": EnsembleWeights(specialist_weight=0.3, claude_weight=0.4, gpt_weight=0.3),
            "Clothing": EnsembleWeights(specialist_weight=0.25, claude_weight=0.4, gpt_weight=0.35),
            "Books": EnsembleWeights(specialist_weight=0.2, claude_weight=0.4, gpt_weight=0.4),
            "Sports": EnsembleWeights(specialist_weight=0.35, claude_weight=0.35, gpt_weight=0.3),
            "Home": EnsembleWeights(specialist_weight=0.3, claude_weight=0.4, gpt_weight=0.3),
            "Beauty": EnsembleWeights(specialist_weight=0.25, claude_weight=0.45, gpt_weight=0.3),
            "Toys": EnsembleWeights(specialist_weight=0.3, claude_weight=0.35, gpt_weight=0.35)
        }
        
        # Performance tracking
        self.ensemble_performance = {
            "total_predictions": 0,
            "ensemble_mae": 0.0,
            "best_single_model_mae": 0.0,
            "improvement_over_best": 0.0,
            "category_performance": {}
        }
        
        # Historical data for dynamic weighting
        self.prediction_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    async def initialize(self, specialist_agent: SpecialistAgent, frontier_agent: FrontierAgent) -> bool:
        """
        Initialize the EnsembleAgent with component agents.
        
        Args:
            specialist_agent: The SpecialistAgent instance
            frontier_agent: The FrontierAgent instance
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing EnsembleAgent with component agents...")
            
            # Set component agents
            self.specialist_agent = specialist_agent
            self.frontier_agent = frontier_agent
            
            # Validate component agents
            if not specialist_agent or not frontier_agent:
                logger.error("Component agents not provided")
                return False
            
            # Initialize category performance tracking
            for category in self.capability.supported_categories:
                self.ensemble_performance["category_performance"][category] = {
                    "predictions": 0,
                    "ensemble_mae": 0.0,
                    "specialist_mae": 0.0,
                    "frontier_mae": 0.0,
                    "improvement": 0.0
                }
            
            logger.info("EnsembleAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing EnsembleAgent: {e}")
            return False
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a price prediction request using ensemble methods.
        
        Args:
            request: Agent request with product information
            
        Returns:
            AgentResponse with ensemble prediction and confidence
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract product information
            product_data = request.payload.get('product', {})
            title = product_data.get('title', '')
            category = product_data.get('category', '')
            description = product_data.get('description', '')
            
            # Get predictions from all component agents
            model_predictions = await self._get_model_predictions(title, category, description)
            
            # Apply ensemble method
            ensemble_result = await self._apply_ensemble_method(model_predictions, category)
            
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(model_predictions, ensemble_result)
            
            # Update performance metrics
            self._update_ensemble_performance(category, ensemble_result, model_predictions)
            
            # Store prediction in history
            self._store_prediction_history(title, category, model_predictions, ensemble_result)
            
            response = AgentResponse(
                request_id=request.request_id,
                agent_type=self.agent_type,
                status="success",
                data={
                    "predicted_price": ensemble_result["predicted_price"],
                    "confidence": ensemble_confidence,
                    "category": category,
                    "ensemble_method": self.ensemble_method.value,
                    "individual_predictions": [
                        {
                            "model": pred.model_name,
                            "price": pred.predicted_price,
                            "confidence": pred.confidence
                        } for pred in model_predictions
                    ],
                    "ensemble_weights": self._get_current_weights(category),
                    "processing_details": {
                        "target_mae": 35.0,
                        "models_used": len(model_predictions),
                        "uncertainty_score": ensemble_result.get("uncertainty", 0.0)
                    }
                },
                confidence=ensemble_confidence,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            logger.info(f"Ensemble prediction: ${ensemble_result['predicted_price']:.2f} for {category}")
            return response
            
        except Exception as e:
            logger.error(f"Error processing ensemble request {request.request_id}: {e}")
            raise
    
    async def _get_model_predictions(self, title: str, category: str, description: str) -> List[ModelPrediction]:
        """Get predictions from all component models."""
        predictions = []
        
        # Create requests for component agents
        specialist_request = AgentRequest(
            request_id=f"ensemble_specialist_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.SPECIALIST,
            task_type="price_prediction",
            payload={"product": {"title": title, "category": category, "description": description}}
        )
        
        frontier_request = AgentRequest(
            request_id=f"ensemble_frontier_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.FRONTIER,
            task_type="price_prediction",
            payload={"product": {"title": title, "category": category, "description": description}}
        )
        
        # Get predictions concurrently
        tasks = []
        
        if category in self.specialist_agent.capability.supported_categories:
            tasks.append(self._get_specialist_prediction(specialist_request))
        
        if category in self.frontier_agent.capability.supported_categories:
            tasks.append(self._get_frontier_prediction(frontier_request))
        
        # Wait for all predictions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, ModelPrediction):
                predictions.append(result)
            else:
                logger.error(f"Error getting prediction: {result}")
        
        return predictions
    
    async def _get_specialist_prediction(self, request: AgentRequest) -> ModelPrediction:
        """Get prediction from SpecialistAgent."""
        try:
            response = await self.specialist_agent.process_request(request)
            
            return ModelPrediction(
                model_name="Specialist QLoRA",
                predicted_price=response.data["predicted_price"],
                confidence=response.confidence,
                response_time=response.processing_time,
                cost=0.001,  # Low cost for quantized model
                historical_mae=39.85,
                category_performance=self._get_specialist_category_performance()
            )
        except Exception as e:
            logger.error(f"Error getting specialist prediction: {e}")
            raise
    
    async def _get_frontier_prediction(self, request: AgentRequest) -> ModelPrediction:
        """Get prediction from FrontierAgent."""
        try:
            response = await self.frontier_agent.process_request(request)
            
            return ModelPrediction(
                model_name=response.data["model_used"],
                predicted_price=response.data["predicted_price"],
                confidence=response.confidence,
                response_time=response.processing_time,
                cost=response.data["processing_details"]["cost_estimate"],
                historical_mae=response.data["processing_details"]["known_mae"],
                category_performance=self._get_frontier_category_performance()
            )
        except Exception as e:
            logger.error(f"Error getting frontier prediction: {e}")
            raise
    
    async def _apply_ensemble_method(self, predictions: List[ModelPrediction], category: str) -> Dict[str, Any]:
        """Apply the selected ensemble method to combine predictions."""
        if not predictions:
            raise ValueError("No predictions to ensemble")
        
        if self.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_ensemble(predictions, category)
        elif self.ensemble_method == EnsembleMethod.DYNAMIC_WEIGHTING:
            return self._dynamic_weighting_ensemble(predictions, category)
        elif self.ensemble_method == EnsembleMethod.CONFIDENCE_BASED:
            return self._confidence_based_ensemble(predictions)
        elif self.ensemble_method == EnsembleMethod.STACKING:
            return self._stacking_ensemble(predictions, category)
        elif self.ensemble_method == EnsembleMethod.BAYESIAN:
            return self._bayesian_ensemble(predictions, category)
        else:
            # Default to weighted average
            return self._weighted_average_ensemble(predictions, category)
    
    def _weighted_average_ensemble(self, predictions: List[ModelPrediction], category: str) -> Dict[str, Any]:
        """Simple weighted average ensemble."""
        weights = self.category_weights.get(category, self.weights)
        weights.normalize()
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = self._get_weight_for_model(pred.model_name, weights)
            weighted_sum += pred.predicted_price * weight
            total_weight += weight
        
        ensemble_price = weighted_sum / total_weight if total_weight > 0 else np.mean([p.predicted_price for p in predictions])
        
        # Calculate uncertainty (standard deviation of predictions)
        prices = [p.predicted_price for p in predictions]
        uncertainty = np.std(prices)
        
        return {
            "predicted_price": ensemble_price,
            "uncertainty": uncertainty,
            "method": "weighted_average"
        }
    
    def _dynamic_weighting_ensemble(self, predictions: List[ModelPrediction], category: str) -> Dict[str, Any]:
        """Dynamic weighting based on recent performance."""
        # Get recent performance for this category
        recent_performance = self._get_recent_category_performance(category)
        
        # Adjust weights based on recent performance
        adjusted_weights = self._adjust_weights_by_performance(recent_performance)
        
        # Apply weighted average with adjusted weights
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = self._get_weight_for_model(pred.model_name, adjusted_weights)
            # Further adjust by individual prediction confidence
            confidence_adjusted_weight = weight * pred.confidence
            weighted_sum += pred.predicted_price * confidence_adjusted_weight
            total_weight += confidence_adjusted_weight
        
        ensemble_price = weighted_sum / total_weight if total_weight > 0 else np.mean([p.predicted_price for p in predictions])
        
        # Calculate uncertainty
        prices = [p.predicted_price for p in predictions]
        uncertainties = [pred.historical_mae for p in predictions]
        combined_uncertainty = np.sqrt(np.var(prices) + np.mean(uncertainties))
        
        return {
            "predicted_price": ensemble_price,
            "uncertainty": combined_uncertainty,
            "method": "dynamic_weighting"
        }
    
    def _confidence_based_ensemble(self, predictions: List[ModelPrediction]) -> Dict[str, Any]:
        """Confidence-based weighted ensemble."""
        total_confidence = sum(p.confidence for p in predictions)
        
        if total_confidence == 0:
            # Fallback to simple average
            ensemble_price = np.mean([p.predicted_price for p in predictions])
        else:
            weighted_sum = sum(p.predicted_price * p.confidence for p in predictions)
            ensemble_price = weighted_sum / total_confidence
        
        # Calculate uncertainty based on confidence distribution
        confidences = [p.confidence for p in predictions]
        uncertainty = 1.0 - (np.mean(confidences) * np.max(confidences))
        
        return {
            "predicted_price": ensemble_price,
            "uncertainty": uncertainty,
            "method": "confidence_based"
        }
    
    def _stacking_ensemble(self, predictions: List[ModelPrediction], category: str) -> Dict[str, Any]:
        """Stacking ensemble using meta-features."""
        # Extract meta-features
        meta_features = []
        for pred in predictions:
            features = [
                pred.predicted_price,
                pred.confidence,
                pred.response_time,
                pred.cost,
                pred.historical_mae
            ]
            meta_features.append(features)
        
        # Simple stacking: weighted combination based on historical performance
        # In production, this would use a trained meta-model
        historical_weights = {
            "Specialist QLoRA": 0.4,
            "Claude 4.5 Sonnet": 0.35,
            "GPT 4.1 Nano": 0.25
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = historical_weights.get(pred.model_name, 0.33)
            # Adjust by confidence
            adjusted_weight = weight * pred.confidence
            weighted_sum += pred.predicted_price * adjusted_weight
            total_weight += adjusted_weight
        
        ensemble_price = weighted_sum / total_weight if total_weight > 0 else np.mean([p.predicted_price for p in predictions])
        
        return {
            "predicted_price": ensemble_price,
            "uncertainty": np.std([p.predicted_price for p in predictions]),
            "method": "stacking"
        }
    
    def _bayesian_ensemble(self, predictions: List[ModelPrediction], category: str) -> Dict[str, Any]:
        """Bayesian ensemble combining predictions with uncertainty."""
        # Treat each prediction as a Gaussian distribution
        # Mean = predicted price, StdDev = historical MAE
        
        combined_mean = 0.0
        combined_precision = 0.0  # 1/variance
        
        for pred in predictions:
            variance = pred.historical_mae ** 2
            precision = 1.0 / variance
            
            combined_mean += pred.predicted_price * precision
            combined_precision += precision
        
        # Bayesian combination
        if combined_precision > 0:
            ensemble_price = combined_mean / combined_precision
            ensemble_variance = 1.0 / combined_precision
            uncertainty = np.sqrt(ensemble_variance)
        else:
            # Fallback
            ensemble_price = np.mean([p.predicted_price for p in predictions])
            uncertainty = np.std([p.predicted_price for p in predictions])
        
        return {
            "predicted_price": ensemble_price,
            "uncertainty": uncertainty,
            "method": "bayesian"
        }
    
    def _get_weight_for_model(self, model_name: str, weights: EnsembleWeights) -> float:
        """Get weight for a specific model."""
        if "Specialist" in model_name:
            return weights.specialist_weight
        elif "Claude" in model_name:
            return weights.claude_weight
        elif "GPT" in model_name:
            return weights.gpt_weight
        else:
            return 0.33  # Default weight
    
    def _calculate_ensemble_confidence(self, predictions: List[ModelPrediction], ensemble_result: Dict[str, Any]) -> float:
        """Calculate confidence for the ensemble prediction."""
        if not predictions:
            return 0.0
        
        # Base confidence from individual model confidences
        individual_confidences = [p.confidence for p in predictions]
        avg_individual_confidence = np.mean(individual_confidences)
        
        # Adjust based on uncertainty
        uncertainty = ensemble_result.get("uncertainty", 0.0)
        uncertainty_penalty = min(0.3, uncertainty / 100.0)  # Max 30% penalty
        
        # Adjust based on agreement between models
        prices = [p.predicted_price for p in predictions]
        price_std = np.std(prices)
        agreement_bonus = max(0.0, 0.2 - price_std / 100.0)  # Up to 20% bonus
        
        # Calculate final confidence
        ensemble_confidence = avg_individual_confidence - uncertainty_penalty + agreement_bonus
        return max(0.0, min(1.0, ensemble_confidence))
    
    def _update_ensemble_performance(self, category: str, ensemble_result: Dict[str, Any], predictions: List[ModelPrediction]):
        """Update ensemble performance metrics."""
        self.ensemble_performance["total_predictions"] += 1
        
        # Update category performance
        if category in self.ensemble_performance["category_performance"]:
            cat_perf = self.ensemble_performance["category_performance"][category]
            cat_perf["predictions"] += 1
    
    def _store_prediction_history(self, title: str, category: str, predictions: List[ModelPrediction], ensemble_result: Dict[str, Any]):
        """Store prediction in history for dynamic weighting."""
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "title": title,
            "category": category,
            "predictions": [
                {
                    "model": pred.model_name,
                    "price": pred.predicted_price,
                    "confidence": pred.confidence,
                    "mae": pred.historical_mae
                } for pred in predictions
            ],
            "ensemble_price": ensemble_result["predicted_price"],
            "uncertainty": ensemble_result.get("uncertainty", 0.0),
            "method": ensemble_result["method"]
        }
        
        self.prediction_history.append(history_entry)
        
        # Limit history size
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history = self.prediction_history[-self.max_history_size:]
    
    def _get_recent_category_performance(self, category: str) -> Dict[str, float]:
        """Get recent performance for a specific category."""
        recent_predictions = [
            entry for entry in self.prediction_history[-100:]  # Last 100 predictions
            if entry["category"] == category
        ]
        
        if not recent_predictions:
            return {}
        
        # Calculate recent performance for each model
        model_performance = {}
        for entry in recent_predictions:
            for pred in entry["predictions"]:
                model_name = pred["model"]
                if model_name not in model_performance:
                    model_performance[model_name] = []
                model_performance[model_name].append(pred["mae"])
        
        # Calculate average performance
        avg_performance = {}
        for model_name, mae_values in model_performance.items():
            avg_performance[model_name] = np.mean(mae_values)
        
        return avg_performance
    
    def _adjust_weights_by_performance(self, recent_performance: Dict[str, float]) -> EnsembleWeights:
        """Adjust weights based on recent performance."""
        base_weights = self.weights
        
        if not recent_performance:
            return base_weights
        
        # Calculate performance ratios (lower MAE = higher weight)
        avg_mae = np.mean(list(recent_performance.values()))
        
        adjusted_weights = EnsembleWeights()
        
        for model_name, mae in recent_performance.items():
            performance_ratio = avg_mae / mae  # Higher ratio = better performance
            
            if "Specialist" in model_name:
                adjusted_weights.specialist_weight = performance_ratio
            elif "Claude" in model_name:
                adjusted_weights.claude_weight = performance_ratio
            elif "GPT" in model_name:
                adjusted_weights.gpt_weight = performance_ratio
        
        adjusted_weights.normalize()
        return adjusted_weights
    
    def _get_current_weights(self, category: str) -> Dict[str, float]:
        """Get current weights for a category."""
        weights = self.category_weights.get(category, self.weights)
        return {
            "specialist": weights.specialist_weight,
            "claude": weights.claude_weight,
            "gpt": weights.gpt_weight
        }
    
    def _get_specialist_category_performance(self) -> Dict[str, float]:
        """Get specialist agent category performance."""
        if self.specialist_agent:
            summary = self.specialist_agent.get_week7_performance_summary()
            return {cat: perf.get("avg_mae", 39.85) for cat, perf in summary["category_performance"].items()}
        return {}
    
    def _get_frontier_category_performance(self) -> Dict[str, float]:
        """Get frontier agent category performance."""
        # Simplified - would be more sophisticated in production
        return {
            "Electronics": 45.0,
            "Appliances": 50.0,
            "Automotive": 55.0,
            "Furniture": 60.0,
            "Clothing": 65.0,
            "Books": 70.0,
            "Sports": 58.0,
            "Home": 62.0,
            "Beauty": 68.0,
            "Toys": 64.0
        }
    
    async def health_check(self) -> bool:
        """Check if the EnsembleAgent is healthy."""
        try:
            if not self.specialist_agent or not self.frontier_agent:
                return False
            
            # Check component agents
            specialist_healthy = await self.specialist_agent.health_check()
            frontier_healthy = await self.frontier_agent.health_check()
            
            return specialist_healthy or frontier_healthy  # At least one component healthy
            
        except Exception as e:
            logger.error(f"EnsembleAgent health check failed: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance summary."""
        return {
            "ensemble_method": self.ensemble_method.value,
            "total_predictions": self.ensemble_performance["total_predictions"],
            "target_mae": 35.0,
            "category_performance": self.ensemble_performance["category_performance"],
            "current_weights": {
                category: self._get_current_weights(category)
                for category in self.capability.supported_categories
            },
            "prediction_history_size": len(self.prediction_history),
            "supported_categories": self.capability.supported_categories,
            "current_status": self.status.value
        }
