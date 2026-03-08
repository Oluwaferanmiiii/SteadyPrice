"""
AutonomousPlannerAgent - Strategic Intelligence

This agent provides strategic planning, market analysis, and intelligent
decision-making for optimal deal timing and portfolio management.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
from collections import defaultdict, deque

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentRequest, AgentResponse
from .scanner_agent import ScannerAgent, DiscoveredDeal
from .ensemble_agent import EnsembleAgent

logger = logging.getLogger(__name__)

class PlanningStrategy(Enum):
    """Different planning strategies"""
    IMMEDIATE_PURCHASE = "immediate_purchase"
    WAIT_FOR_BETTER_DEAL = "wait_for_better_deal"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    BUDGET_CONSTRAINED = "budget_constrained"
    SEASONAL_TIMING = "seasonal_timing"

class MarketTrend(Enum):
    """Market trend indicators"""
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"

@dataclass
class MarketAnalysis:
    """Market analysis results"""
    category: str
    trend: MarketTrend
    price_volatility: float
    average_discount: float
    deal_frequency: float
    seasonal_factor: float
    confidence: float
    timeframe: str  # "daily", "weekly", "monthly"

@dataclass
class PurchaseRecommendation:
    """Purchase recommendation for a deal"""
    deal_id: str
    action: str  # "buy_now", "wait", "skip"
    confidence: float
    reasoning: str
    optimal_timing: Optional[datetime]
    expected_savings: float
    risk_assessment: str
    budget_impact: float

@dataclass
class PortfolioOptimization:
    """Portfolio optimization results"""
    total_budget: float
    allocated_budget: float
    recommended_deals: List[str]
    expected_total_savings: float
    risk_score: float
    diversification_score: float
    time_horizon: str

class AutonomousPlannerAgent(BaseAgent):
    """
    AutonomousPlannerAgent that provides strategic intelligence
    for optimal deal timing and portfolio management.
    
    Capabilities:
    - Market trend analysis and prediction
    - Optimal deal timing recommendations
    - Portfolio optimization across multiple deals
    - Budget allocation and risk management
    - Seasonal and temporal analysis
    """
    
    def __init__(self):
        # Define agent capabilities
        capability = AgentCapability(
            name="Autonomous Planner Agent",
            description="Strategic intelligence for deal timing and portfolio optimization",
            max_concurrent_tasks=25,
            average_response_time=0.5,  # 500ms average
            accuracy_metric=0.89,  # 89% accuracy in recommendations
            cost_per_request=0.005,  # Low cost
            supported_categories=["Electronics", "Appliances", "Automotive", "Furniture", "Clothing", "Books", "Sports", "Home", "Beauty", "Toys"]
        )
        
        super().__init__(AgentType.PLANNER, capability)
        
        # Component agents
        self.scanner_agent: Optional[ScannerAgent] = None
        self.ensemble_agent: Optional[EnsembleAgent] = None
        
        # Market data storage
        self.market_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.deal_frequency: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # Planning parameters
        self.planning_horizons = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30,
            "quarterly": 90
        }
        
        # Seasonal patterns
        self.seasonal_patterns = {
            "Electronics": {
                "peak_months": [11, 12, 6],  # Black Friday, Christmas, Back to School
                "low_months": [1, 2, 4],      # Post-holiday, Spring
                "multiplier": 1.3
            },
            "Appliances": {
                "peak_months": [5, 9, 10],    # Memorial Day, Labor Day
                "low_months": [1, 2, 7],      # Post-holiday, Summer
                "multiplier": 1.2
            },
            "Automotive": {
                "peak_months": [8, 9, 12],    # End of year, Labor Day
                "low_months": [1, 2, 3],      # Post-holiday
                "multiplier": 1.15
            }
        }
        
        # Performance metrics
        self.planning_metrics = {
            "recommendations_made": 0,
            "accurate_recommendations": 0,
            "average_savings_achieved": 0.0,
            "portfolio_optimizations": 0,
            "market_analyses_performed": 0,
            "last_analysis_time": None
        }
    
    async def initialize(self, scanner_agent: ScannerAgent, ensemble_agent: EnsembleAgent) -> bool:
        """
        Initialize the AutonomousPlannerAgent with component agents.
        
        Args:
            scanner_agent: The ScannerAgent instance
            ensemble_agent: The EnsembleAgent instance
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing AutonomousPlannerAgent with component agents...")
            
            # Set component agents
            self.scanner_agent = scanner_agent
            self.ensemble_agent = ensemble_agent
            
            # Validate component agents
            if not scanner_agent or not ensemble_agent:
                logger.error("Component agents not provided")
                return False
            
            # Initialize market data structures
            for category in self.capability.supported_categories:
                self.market_data[category] = deque(maxlen=1000)
                self.deal_frequency[category] = deque(maxlen=500)
            
            logger.info("AutonomousPlannerAgent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing AutonomousPlannerAgent: {e}")
            return False
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a planning request and provide strategic recommendations.
        
        Args:
            request: Agent request with planning parameters
            
        Returns:
            AgentResponse with planning recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            task_type = request.payload.get('task_type', 'market_analysis')
            
            if task_type == 'market_analysis':
                result = await self._analyze_market(request.payload.get('category'))
            elif task_type == 'deal_recommendation':
                result = await self._recommend_deal_action(request.payload.get('deal_id'))
            elif task_type == 'portfolio_optimization':
                result = await self._optimize_portfolio(request.payload.get('budget', 1000.0))
            elif task_type == 'timing_analysis':
                result = await self._analyze_optimal_timing(request.payload.get('category'))
            elif task_type == 'strategic_planning':
                result = await self._strategic_planning(request.payload)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            response = AgentResponse(
                request_id=request.request_id,
                agent_type=self.agent_type,
                status="success",
                data=result,
                confidence=0.85,  # High confidence in strategic analysis
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            logger.info(f"AutonomousPlannerAgent processed {task_type} request")
            return response
            
        except Exception as e:
            logger.error(f"Error processing planning request {request.request_id}: {e}")
            raise
    
    async def _analyze_market(self, category: str) -> Dict[str, Any]:
        """Analyze market conditions for a specific category."""
        try:
            # Get recent deals for this category
            scanner_request = AgentRequest(
                request_id=f"market_analysis_{category}_{datetime.utcnow().timestamp()}",
                agent_type=AgentType.SCANNER,
                task_type="get_deals",
                payload={"filters": {"category": category, "limit": 100}}
            )
            
            scanner_response = await self.scanner_agent.process_request(scanner_request)
            recent_deals = scanner_response.data.get("deals", [])
            
            # Analyze market trends
            market_analysis = self._calculate_market_metrics(category, recent_deals)
            
            # Update metrics
            self.planning_metrics["market_analyses_performed"] += 1
            self.planning_metrics["last_analysis_time"] = datetime.utcnow()
            
            return {
                "category": category,
                "analysis": asdict(market_analysis),
                "insights": self._generate_market_insights(market_analysis),
                "recommendations": self._generate_market_recommendations(market_analysis),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market for {category}: {e}")
            raise
    
    def _calculate_market_metrics(self, category: str, deals: List[Dict[str, Any]]) -> MarketAnalysis:
        """Calculate market analysis metrics."""
        if not deals:
            return MarketAnalysis(
                category=category,
                trend=MarketTrend.STABLE,
                price_volatility=0.0,
                average_discount=0.0,
                deal_frequency=0.0,
                seasonal_factor=1.0,
                confidence=0.0,
                timeframe="weekly"
            )
        
        # Extract price data
        current_prices = [float(deal.get("current_price", 0)) for deal in deals]
        original_prices = [float(deal.get("original_price", 0)) for deal in deals]
        discounts = [float(deal.get("discount_percentage", 0)) for deal in deals]
        
        # Calculate metrics
        avg_current_price = np.mean(current_prices)
        avg_original_price = np.mean(original_prices)
        avg_discount = np.mean(discounts)
        
        # Price volatility (coefficient of variation)
        price_volatility = np.std(current_prices) / avg_current_price if avg_current_price > 0 else 0.0
        
        # Deal frequency (deals per day)
        if deals:
            earliest_date = datetime.fromisoformat(deals[-1].get("discovered_at", datetime.utcnow().isoformat()))
            latest_date = datetime.fromisoformat(deals[0].get("discovered_at", datetime.utcnow().isoformat()))
            days_diff = (latest_date - earliest_date).days + 1
            deal_frequency = len(deals) / max(1, days_diff)
        else:
            deal_frequency = 0.0
        
        # Determine trend
        if len(current_prices) >= 10:
            # Simple trend analysis based on recent price movement
            recent_prices = current_prices[:10]  # Most recent
            older_prices = current_prices[10:20] if len(current_prices) > 20 else current_prices[10:]
            
            if older_prices:
                recent_avg = np.mean(recent_prices)
                older_avg = np.mean(older_prices)
                
                price_change = (recent_avg - older_avg) / older_avg
                
                if price_change > 0.05:
                    trend = MarketTrend.RISING
                elif price_change < -0.05:
                    trend = MarketTrend.FALLING
                else:
                    trend = MarketTrend.STABLE
            else:
                trend = MarketTrend.STABLE
        else:
            trend = MarketTrend.STABLE
        
        # Seasonal factor
        current_month = datetime.utcnow().month
        seasonal_factor = self._calculate_seasonal_factor(category, current_month)
        
        # Confidence based on data quality
        confidence = min(1.0, len(deals) / 50.0)  # More data = higher confidence
        
        return MarketAnalysis(
            category=category,
            trend=trend,
            price_volatility=price_volatility,
            average_discount=avg_discount,
            deal_frequency=deal_frequency,
            seasonal_factor=seasonal_factor,
            confidence=confidence,
            timeframe="weekly"
        )
    
    def _calculate_seasonal_factor(self, category: str, month: int) -> float:
        """Calculate seasonal factor for a category and month."""
        if category in self.seasonal_patterns:
            pattern = self.seasonal_patterns[category]
            
            if month in pattern["peak_months"]:
                return pattern["multiplier"]
            elif month in pattern["low_months"]:
                return 1.0 / pattern["multiplier"]
            else:
                return 1.0
        
        return 1.0
    
    def _generate_market_insights(self, analysis: MarketAnalysis) -> List[str]:
        """Generate insights from market analysis."""
        insights = []
        
        # Trend insights
        if analysis.trend == MarketTrend.RISING:
            insights.append(f"Prices in {analysis.category} are trending upward - consider buying sooner rather than later")
        elif analysis.trend == MarketTrend.FALLING:
            insights.append(f"Prices in {analysis.category} are falling - good time to wait for better deals")
        else:
            insights.append(f"Prices in {analysis.category} are stable - normal deal-hunting strategy applies")
        
        # Volatility insights
        if analysis.price_volatility > 0.3:
            insights.append(f"High price volatility detected ({analysis.price_volatility:.2f}) - prices may change significantly")
        elif analysis.price_volatility < 0.1:
            insights.append(f"Low price volatility ({analysis.price_volatility:.2f}) - prices are relatively stable")
        
        # Discount insights
        if analysis.average_discount > 30:
            insights.append(f"Excellent discount opportunities available ({analysis.average_discount:.1f}% average)")
        elif analysis.average_discount < 15:
            insights.append(f"Limited discount availability ({analysis.average_discount:.1f}% average) - may need to be patient")
        
        # Deal frequency insights
        if analysis.deal_frequency > 5:
            insights.append(f"High deal frequency ({analysis.deal_frequency:.1f} deals/day) - many opportunities")
        elif analysis.deal_frequency < 1:
            insights.append(f"Low deal frequency ({analysis.deal_frequency:.1f} deals/day) - act quickly when deals appear")
        
        # Seasonal insights
        if analysis.seasonal_factor > 1.1:
            insights.append(f"Currently in peak season for {analysis.category} - more deals available")
        elif analysis.seasonal_factor < 0.9:
            insights.append(f"Currently in off-season for {analysis.category} - fewer deals expected")
        
        return insights
    
    def _generate_market_recommendations(self, analysis: MarketAnalysis) -> List[str]:
        """Generate recommendations based on market analysis."""
        recommendations = []
        
        # Trend-based recommendations
        if analysis.trend == MarketTrend.RISING:
            recommendations.append("Consider purchasing soon to avoid price increases")
        elif analysis.trend == MarketTrend.FALLING:
            recommendations.append("Wait for potentially better deals as prices decrease")
        
        # Volatility-based recommendations
        if analysis.price_volatility > 0.3:
            recommendations.append("Set price alerts for significant drops due to high volatility")
        
        # Discount-based recommendations
        if analysis.average_discount > 25:
            recommendations.append("Current market offers excellent discount opportunities")
        elif analysis.average_discount < 15:
            recommendations.append("Be patient and wait for better discount opportunities")
        
        # Seasonal recommendations
        current_month = datetime.utcnow().month
        if analysis.category in self.seasonal_patterns:
            pattern = self.seasonal_patterns[analysis.category]
            
            # Check if approaching peak season
            upcoming_months = [(current_month + i) % 12 or 12 for i in range(1, 4)]
            if any(month in pattern["peak_months"] for month in upcoming_months):
                recommendations.append(f"Peak season approaching - expect better deals in the coming months")
        
        return recommendations
    
    async def _recommend_deal_action(self, deal_id: str) -> Dict[str, Any]:
        """Provide recommendation for a specific deal."""
        try:
            # Get deal details
            scanner_request = AgentRequest(
                request_id=f"deal_rec_{deal_id}_{datetime.utcnow().timestamp()}",
                agent_type=AgentType.SCANNER,
                task_type="get_deals",
                payload={"filters": {"limit": 1000}}  # Get all deals to find the specific one
            )
            
            scanner_response = await self.scanner_agent.process_request(scanner_request)
            all_deals = scanner_response.data.get("deals", [])
            
            # Find the specific deal
            target_deal = None
            for deal in all_deals:
                if deal.get("deal_id") == deal_id:
                    target_deal = deal
                    break
            
            if not target_deal:
                raise ValueError(f"Deal {deal_id} not found")
            
            # Get market analysis for this category
            category = target_deal.get("category", "Electronics")
            market_analysis = await self._analyze_market(category)
            
            # Generate recommendation
            recommendation = self._generate_deal_recommendation(target_deal, market_analysis["analysis"])
            
            # Update metrics
            self.planning_metrics["recommendations_made"] += 1
            
            return {
                "deal_id": deal_id,
                "deal_details": target_deal,
                "recommendation": asdict(recommendation),
                "market_context": market_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating deal recommendation for {deal_id}: {e}")
            raise
    
    def _generate_deal_recommendation(self, deal: Dict[str, Any], market_analysis: MarketAnalysis) -> PurchaseRecommendation:
        """Generate a specific recommendation for a deal."""
        discount = float(deal.get("discount_percentage", 0))
        current_price = float(deal.get("current_price", 0))
        original_price = float(deal.get("original_price", 0))
        category = deal.get("category", "Electronics")
        
        # Calculate action based on multiple factors
        action_score = 0.0
        reasoning_parts = []
        
        # Discount factor
        if discount > 40:
            action_score += 3
            reasoning_parts.append(f"Excellent discount ({discount:.1f}%)")
        elif discount > 25:
            action_score += 2
            reasoning_parts.append(f"Good discount ({discount:.1f}%)")
        elif discount > 15:
            action_score += 1
            reasoning_parts.append(f"Moderate discount ({discount:.1f}%)")
        else:
            action_score -= 1
            reasoning_parts.append(f"Low discount ({discount:.1f}%)")
        
        # Market trend factor
        if market_analysis.trend == MarketTrend.RISING:
            action_score += 2
            reasoning_parts.append("Prices are rising - buy now")
        elif market_analysis.trend == MarketTrend.FALLING:
            action_score -= 1
            reasoning_parts.append("Prices are falling - consider waiting")
        
        # Seasonal factor
        if market_analysis.seasonal_factor > 1.1:
            action_score += 1
            reasoning_parts.append("Peak season - good time to buy")
        elif market_analysis.seasonal_factor < 0.9:
            action_score -= 1
            reasoning_parts.append("Off-season - better deals may come")
        
        # Deal frequency factor
        if market_analysis.deal_frequency < 1:
            action_score += 1
            reasoning_parts.append("Low deal frequency - act on good deals")
        elif market_analysis.deal_frequency > 5:
            action_score -= 1
            reasoning_parts.append("High deal frequency - can wait for better")
        
        # Determine action
        if action_score >= 4:
            action = "buy_now"
            confidence = min(0.95, 0.6 + (action_score - 4) * 0.1)
        elif action_score >= 1:
            action = "wait"
            confidence = min(0.85, 0.5 + action_score * 0.1)
        else:
            action = "skip"
            confidence = min(0.75, 0.4 + max(0, action_score + 2) * 0.1)
        
        # Calculate expected savings and risk
        expected_savings = original_price - current_price
        risk_assessment = self._assess_deal_risk(deal, market_analysis)
        
        # Determine optimal timing
        optimal_timing = None
        if action == "wait":
            optimal_timing = datetime.utcnow() + timedelta(days=14)  # Estimate
        
        reasoning = "; ".join(reasoning_parts)
        
        return PurchaseRecommendation(
            deal_id=deal.get("deal_id", ""),
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            optimal_timing=optimal_timing,
            expected_savings=expected_savings,
            risk_assessment=risk_assessment,
            budget_impact=current_price
        )
    
    def _assess_deal_risk(self, deal: Dict[str, Any], market_analysis: MarketAnalysis) -> str:
        """Assess the risk level of a deal."""
        risk_factors = []
        
        # Price volatility risk
        if market_analysis.price_volatility > 0.3:
            risk_factors.append("high_volatility")
        
        # Low confidence risk
        if market_analysis.confidence < 0.5:
            risk_factors.append("low_confidence")
        
        # Low discount risk
        if float(deal.get("discount_percentage", 0)) < 15:
            risk_factors.append("low_discount")
        
        # Seasonal risk
        if market_analysis.seasonal_factor < 0.9:
            risk_factors.append("off_season")
        
        # Determine overall risk
        if len(risk_factors) >= 3:
            return "high"
        elif len(risk_factors) >= 1:
            return "medium"
        else:
            return "low"
    
    async def _optimize_portfolio(self, budget: float) -> Dict[str, Any]:
        """Optimize a portfolio of deals within budget constraints."""
        try:
            # Get all available deals
            scanner_request = AgentRequest(
                request_id=f"portfolio_opt_{datetime.utcnow().timestamp()}",
                agent_type=AgentType.SCANNER,
                task_type="get_deals",
                payload={"filters": {"min_discount": 10, "limit": 100}}
            )
            
            scanner_response = await self.scanner_agent.process_request(scanner_request)
            all_deals = scanner_response.data.get("deals", [])
            
            # Generate recommendations for each deal
            deal_recommendations = []
            for deal in all_deals:
                market_analysis = await self._analyze_market(deal.get("category", "Electronics"))
                recommendation = self._generate_deal_recommendation(deal, market_analysis["analysis"])
                
                # Only include "buy_now" recommendations
                if recommendation.action == "buy_now":
                    deal_recommendations.append((deal, recommendation))
            
            # Sort by expected savings per dollar (ROI)
            deal_recommendations.sort(
                key=lambda x: x[1].expected_savings / max(1, x[1].budget_impact),
                reverse=True
            )
            
            # Select optimal deals within budget
            selected_deals = []
            allocated_budget = 0.0
            total_savings = 0.0
            
            for deal, recommendation in deal_recommendations:
                deal_cost = recommendation.budget_impact
                
                if allocated_budget + deal_cost <= budget:
                    selected_deals.append(recommendation.deal_id)
                    allocated_budget += deal_cost
                    total_savings += recommendation.expected_savings
                else:
                    break
            
            # Calculate portfolio metrics
            risk_score = self._calculate_portfolio_risk(selected_deals, all_deals)
            diversification_score = self._calculate_diversification(selected_deals, all_deals)
            
            optimization = PortfolioOptimization(
                total_budget=budget,
                allocated_budget=allocated_budget,
                recommended_deals=selected_deals,
                expected_total_savings=total_savings,
                risk_score=risk_score,
                diversification_score=diversification_score,
                time_horizon="30_days"
            )
            
            # Update metrics
            self.planning_metrics["portfolio_optimizations"] += 1
            
            return {
                "optimization": asdict(optimization),
                "selected_deals": [
                    deal for deal in all_deals 
                    if deal.get("deal_id") in selected_deals
                ],
                "analysis": {
                    "total_considered": len(deal_recommendations),
                    "budget_utilization": allocated_budget / budget,
                    "expected_roi": (total_savings / allocated_budget * 100) if allocated_budget > 0 else 0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise
    
    def _calculate_portfolio_risk(self, selected_deals: List[str], all_deals: List[Dict[str, Any]]) -> float:
        """Calculate portfolio risk score."""
        if not selected_deals:
            return 0.0
        
        # Get risk levels for selected deals
        deal_details = {deal.get("deal_id"): deal for deal in all_deals}
        risk_scores = []
        
        for deal_id in selected_deals:
            deal = deal_details.get(deal_id, {})
            # Simple risk calculation based on discount and category
            discount = float(deal.get("discount_percentage", 0))
            
            if discount > 40:
                risk_score = 0.2  # Low risk
            elif discount > 25:
                risk_score = 0.4  # Medium risk
            else:
                risk_score = 0.7  # High risk
            
            risk_scores.append(risk_score)
        
        return np.mean(risk_scores)
    
    def _calculate_diversification(self, selected_deals: List[str], all_deals: List[Dict[str, Any]]) -> float:
        """Calculate portfolio diversification score."""
        if not selected_deals:
            return 0.0
        
        # Count categories in selected deals
        deal_details = {deal.get("deal_id"): deal for deal in all_deals}
        categories = set()
        
        for deal_id in selected_deals:
            deal = deal_details.get(deal_id, {})
            categories.add(deal.get("category", "Electronics"))
        
        # Diversification score based on category variety
        return min(1.0, len(categories) / 5.0)  # Normalize to max 5 categories
    
    async def _analyze_optimal_timing(self, category: str) -> Dict[str, Any]:
        """Analyze optimal timing for purchases in a category."""
        try:
            # Get market analysis
            market_analysis = await self._analyze_market(category)
            
            # Analyze seasonal patterns
            seasonal_analysis = self._analyze_seasonal_patterns(category)
            
            # Predict optimal timing
            optimal_timing = self._predict_optimal_timing(category, market_analysis["analysis"], seasonal_analysis)
            
            return {
                "category": category,
                "optimal_timing": optimal_timing,
                "seasonal_analysis": seasonal_analysis,
                "market_analysis": market_analysis["analysis"],
                "recommendations": self._generate_timing_recommendations(optimal_timing, market_analysis["analysis"]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing optimal timing for {category}: {e}")
            raise
    
    def _analyze_seasonal_patterns(self, category: str) -> Dict[str, Any]:
        """Analyze seasonal patterns for a category."""
        current_month = datetime.utcnow().month
        current_season = self._get_season(current_month)
        
        if category in self.seasonal_patterns:
            pattern = self.seasonal_patterns[category]
            
            # Determine current season status
            if current_month in pattern["peak_months"]:
                season_status = "peak"
                multiplier = pattern["multiplier"]
            elif current_month in pattern["low_months"]:
                season_status = "low"
                multiplier = 1.0 / pattern["multiplier"]
            else:
                season_status = "normal"
                multiplier = 1.0
            
            # Find next peak season
            upcoming_peak = self._find_next_peak_season(current_month, pattern["peak_months"])
            
            return {
                "current_season": season_status,
                "current_multiplier": multiplier,
                "peak_months": pattern["peak_months"],
                "low_months": pattern["low_months"],
                "next_peak_season": upcoming_peak,
                "seasonal_strength": pattern["multiplier"]
            }
        else:
            return {
                "current_season": "normal",
                "current_multiplier": 1.0,
                "peak_months": [],
                "low_months": [],
                "next_peak_season": None,
                "seasonal_strength": 1.0
            }
    
    def _get_season(self, month: int) -> str:
        """Get season for a month."""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    def _find_next_peak_season(self, current_month: int, peak_months: List[int]) -> Optional[str]:
        """Find the next peak season."""
        if not peak_months:
            return None
        
        # Sort peak months
        sorted_peaks = sorted(peak_months)
        
        # Find next peak month
        for peak_month in sorted_peaks:
            if peak_month > current_month:
                months_until = peak_month - current_month
                season_name = self._get_season(peak_month)
                return f"{season_name.capitalize()} (in {months_until} months)"
        
        # If no peak month found this year, look at next year
        next_year_peak = sorted_peaks[0]
        months_until = (12 - current_month) + next_year_peak
        season_name = self._get_season(next_year_peak)
        return f"{season_name.capitalize()} (in {months_until} months)"
    
    def _predict_optimal_timing(self, category: str, market_analysis: MarketAnalysis, seasonal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal timing for purchases."""
        current_date = datetime.utcnow()
        
        # Base timing recommendation
        if market_analysis.trend == MarketTrend.RISING:
            base_recommendation = "buy_soon"
            urgency_days = 7
        elif market_analysis.trend == MarketTrend.FALLING:
            base_recommendation = "wait_for_better_deal"
            urgency_days = 21
        else:
            base_recommendation = "normal_timing"
            urgency_days = 14
        
        # Adjust for seasonal factors
        seasonal_multiplier = seasonal_analysis.get("current_multiplier", 1.0)
        
        if seasonal_multiplier > 1.1:
            # Peak season - good time to buy
            base_recommendation = "buy_now"
            urgency_days = max(3, urgency_days - 7)
        elif seasonal_multiplier < 0.9:
            # Off season - consider waiting
            if base_recommendation == "buy_now":
                base_recommendation = "wait_for_peak_season"
                urgency_days = 30
        
        # Calculate optimal date
        optimal_date = current_date + timedelta(days=urgency_days)
        
        return {
            "recommendation": base_recommendation,
            "optimal_date": optimal_date.isoformat(),
            "urgency_days": urgency_days,
            "confidence": market_analysis.confidence,
            "factors": {
                "market_trend": market_analysis.trend.value,
                "seasonal_multiplier": seasonal_multiplier,
                "price_volatility": market_analysis.price_volatility,
                "deal_frequency": market_analysis.deal_frequency
            }
        }
    
    def _generate_timing_recommendations(self, optimal_timing: Dict[str, Any], market_analysis: MarketAnalysis) -> List[str]:
        """Generate timing recommendations."""
        recommendations = []
        recommendation = optimal_timing["recommendation"]
        
        if recommendation == "buy_now":
            recommendations.append("Current conditions are optimal - purchase immediately")
            recommendations.append(f"High deal frequency ({market_analysis.deal_frequency:.1f}/day) supports immediate action")
        elif recommendation == "buy_soon":
            recommendations.append(f"Purchase within {optimal_timing['urgency_days']} days for best results")
            recommendations.append("Prices may rise soon - don't wait too long")
        elif recommendation == "wait_for_better_deal":
            recommendations.append(f"Wait {optimal_timing['urgency_days']} days for potentially better deals")
            recommendations.append("Market conditions suggest better opportunities ahead")
        elif recommendation == "wait_for_peak_season":
            recommendations.append("Wait for peak season for better deals and selection")
            recommendations.append(f"Next peak season: {optimal_timing['factors'].get('next_peak_season', 'Unknown')}")
        else:
            recommendations.append("Normal market conditions - standard deal-hunting approach applies")
        
        return recommendations
    
    async def _strategic_planning(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive strategic planning."""
        try:
            categories = request_payload.get("categories", self.capability.supported_categories)
            budget = request_payload.get("budget", 1000.0)
            time_horizon = request_payload.get("time_horizon", "monthly")
            
            # Analyze each category
            category_analyses = {}
            for category in categories:
                analysis = await self._analyze_market(category)
                category_analyses[category] = analysis
            
            # Portfolio optimization
            portfolio_opt = await self._optimize_portfolio(budget)
            
            # Overall strategy
            strategy = self._generate_overall_strategy(category_analyses, portfolio_opt, budget, time_horizon)
            
            return {
                "strategy": strategy,
                "category_analyses": category_analyses,
                "portfolio_optimization": portfolio_opt,
                "recommendations": self._generate_strategic_recommendations(strategy, category_analyses),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in strategic planning: {e}")
            raise
    
    def _generate_overall_strategy(self, category_analyses: Dict[str, Any], portfolio_opt: Dict[str, Any], budget: float, time_horizon: str) -> Dict[str, Any]:
        """Generate overall strategic recommendations."""
        # Analyze overall market conditions
        rising_categories = []
        falling_categories = []
        stable_categories = []
        
        for category, analysis in category_analyses.items():
            trend = analysis["analysis"]["trend"]
            if trend == "rising":
                rising_categories.append(category)
            elif trend == "falling":
                falling_categories.append(category)
            else:
                stable_categories.append(category)
        
        # Determine overall strategy
        if len(rising_categories) > len(falling_categories):
            overall_trend = "bullish"
            strategy_type = "aggressive"
        elif len(falling_categories) > len(rising_categories):
            overall_trend = "bearish"
            strategy_type = "conservative"
        else:
            overall_trend = "neutral"
            strategy_type = "balanced"
        
        return {
            "overall_trend": overall_trend,
            "strategy_type": strategy_type,
            "rising_categories": rising_categories,
            "falling_categories": falling_categories,
            "stable_categories": stable_categories,
            "budget_allocation": portfolio_opt["optimization"]["allocated_budget"],
            "expected_savings": portfolio_opt["optimization"]["expected_total_savings"],
            "risk_level": portfolio_opt["optimization"]["risk_score"],
            "diversification": portfolio_opt["optimization"]["diversification_score"]
        }
    
    def _generate_strategic_recommendations(self, strategy: Dict[str, Any], category_analyses: Dict[str, Any]) -> List[str]:
        """Generate high-level strategic recommendations."""
        recommendations = []
        
        strategy_type = strategy["strategy_type"]
        
        if strategy_type == "aggressive":
            recommendations.append("Market is bullish - consider accelerating purchases in rising categories")
            recommendations.append("Focus on categories with upward price trends")
        elif strategy_type == "conservative":
            recommendations.append("Market is bearish - wait for better deals in falling categories")
            recommendations.append("Prioritize essential purchases only")
        else:
            recommendations.append("Market is neutral - follow standard deal-hunting strategy")
        
        # Budget recommendations
        budget_utilization = strategy["budget_allocation"] / 100.0  # Assuming $100 budget reference
        if budget_utilization > 0.8:
            recommendations.append("High budget utilization - consider increasing budget for better opportunities")
        elif budget_utilization < 0.5:
            recommendations.append("Low budget utilization - good opportunities available within budget")
        
        # Risk recommendations
        if strategy["risk_level"] > 0.6:
            recommendations.append("Portfolio has higher risk - consider diversifying across categories")
        elif strategy["risk_level"] < 0.3:
            recommendations.append("Portfolio is low risk - consider more aggressive opportunities")
        
        return recommendations
    
    async def health_check(self) -> bool:
        """Check if the AutonomousPlannerAgent is healthy."""
        try:
            if not self.scanner_agent or not self.ensemble_agent:
                return False
            
            # Test with a simple market analysis
            try:
                result = await self._analyze_market("Electronics")
                return True
            except:
                return False
            
        except Exception as e:
            logger.error(f"AutonomousPlannerAgent health check failed: {e}")
            return False
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive planning metrics."""
        return {
            **self.planning_metrics,
            "market_data_categories": len(self.market_data),
            "supported_categories": self.capability.supported_categories,
            "seasonal_patterns_available": list(self.seasonal_patterns.keys()),
            "current_status": self.status.value
        }
