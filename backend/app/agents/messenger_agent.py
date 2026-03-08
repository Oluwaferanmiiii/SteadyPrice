"""
MessengerAgent - User Experience & Communication

This agent handles user interaction, natural language communication,
and provides an intelligent interface for the multi-agent system.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re
from collections import defaultdict

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentRequest, AgentResponse
from .specialist_agent import SpecialistAgent
from .frontier_agent import FrontierAgent
from .ensemble_agent import EnsembleAgent
from .scanner_agent import ScannerAgent
from .planner_agent import AutonomousPlannerAgent

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of messages the agent can handle"""
    QUERY = "query"
    COMMAND = "command"
    CONVERSATION = "conversation"
    NOTIFICATION = "notification"
    FEEDBACK = "feedback"

class IntentType(Enum):
    """User intent types"""
    PRICE_PREDICTION = "price_prediction"
    DEAL_SEARCH = "deal_search"
    MARKET_ANALYSIS = "market_analysis"
    PORTFOLIO_PLANNING = "portfolio_planning"
    GENERAL_HELP = "general_help"
    SYSTEM_STATUS = "system_status"

@dataclass
class UserMessage:
    """User message with metadata"""
    message_id: str
    user_id: str
    content: str
    message_type: MessageType
    intent: Optional[IntentType]
    entities: Dict[str, Any]
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None

@dataclass
class AgentResponse:
    """Agent response to user"""
    response_id: str
    user_id: str
    content: str
    response_type: str
    confidence: float
    data: Optional[Dict[str, Any]]
    suggestions: List[str]
    timestamp: datetime
    follow_up_questions: List[str]

class MessengerAgent(BaseAgent):
    """
    MessengerAgent that provides intelligent user interaction
    and natural language communication for the multi-agent system.
    
    Capabilities:
    - Natural language understanding and processing
    - Intent recognition and entity extraction
    - Contextual conversation management
    - Multi-agent coordination for user requests
    - Personalized recommendations and assistance
    """
    
    def __init__(self):
        # Define agent capabilities
        capability = AgentCapability(
            name="Messenger Agent",
            description="Intelligent user interface and natural language communication",
            max_concurrent_tasks=100,
            average_response_time=0.4,  # 400ms average
            accuracy_metric=0.92,  # 92% accuracy in intent recognition
            cost_per_request=0.002,  # Low cost
            supported_categories=["Electronics", "Appliances", "Automotive", "Furniture", "Clothing", "Books", "Sports", "Home", "Beauty", "Toys"]
        )
        
        super().__init__(AgentType.MESSENGER, capability)
        
        # Component agents
        self.specialist_agent: Optional[SpecialistAgent] = None
        self.frontier_agent: Optional[FrontierAgent] = None
        self.ensemble_agent: Optional[EnsembleAgent] = None
        self.scanner_agent: Optional[ScannerAgent] = None
        self.planner_agent: Optional[AutonomousPlannerAgent] = None
        
        # Conversation management
        self.user_sessions: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.conversation_history: Dict[str, List[UserMessage]] = defaultdict(list)
        
        # Natural language processing
        self.intent_patterns = {
            IntentType.PRICE_PREDICTION: [
                r"how much.*worth",
                r"price.*prediction",
                r"estimate.*cost",
                r"what.*price",
                r"predict.*price",
                r"value.*product"
            ],
            IntentType.DEAL_SEARCH: [
                r"find.*deal",
                r"search.*deal",
                r"best.*price",
                r"discount.*product",
                r"sale.*item",
                r"cheapest.*price"
            ],
            IntentType.MARKET_ANALYSIS: [
                r"market.*analysis",
                r"price.*trend",
                r"market.*condition",
                r"when.*buy",
                r"best.*time",
                r"seasonal.*price"
            ],
            IntentType.PORTFOLIO_PLANNING: [
                r"portfolio.*optimization",
                r"budget.*allocation",
                r"multiple.*deals",
                r"best.*combination",
                r"optimize.*purchases"
            ],
            IntentType.GENERAL_HELP: [
                r"help",
                r"how.*use",
                r"what.*can.*do",
                r"explain.*feature",
                r"guide.*me"
            ],
            IntentType.SYSTEM_STATUS: [
                r"system.*status",
                r"how.*working",
                r"agent.*status",
                r"performance.*metrics"
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "price": r"\$(\d+(?:\.\d{2})?)",
            "category": r"\b(electronics|appliances|automotive|furniture|clothing|books|sports|home|beauty|toys)\b",
            "brand": r"\b(apple|samsung|sony|lg|dell|hp|lenovo|nike|adidas|ikea|target|walmart|amazon|best\s*buy)\b",
            "percentage": r"(\d+(?:\.\d+)?)\s*%",
            "timeframe": r"\b(today|tomorrow|week|month|year|daily|weekly|monthly)\b"
        }
        
        # Response templates
        self.response_templates = {
            "greeting": [
                "Hello! I'm your SteadyPrice assistant. How can I help you find the best deals today?",
                "Hi there! I can help you with price predictions, deal searches, and market analysis. What are you looking for?",
                "Welcome! I'm here to help you make smart purchasing decisions. What can I assist you with?"
            ],
            "price_prediction": [
                "I'll predict the price for {product}. Let me analyze it with our advanced models.",
                "Let me get you a price estimate for {product}. I'll use our ensemble of AI models for accuracy.",
                "I'll analyze {product} and provide you with a price prediction using our fine-tuned models."
            ],
            "deal_search": [
                "I'll search for the best deals on {category} for you.",
                "Let me find current deals and discounts for {category}.",
                "Searching for the best {category} deals across multiple retailers..."
            ],
            "market_analysis": [
                "I'll analyze the market conditions for {category} and provide strategic insights.",
                "Let me examine the market trends and optimal timing for {category} purchases.",
                "Analyzing {category} market data to give you the best purchasing advice."
            ],
            "unclear_intent": [
                "I'm not sure I understand. Could you please clarify what you're looking for?",
                "Let me make sure I understand correctly. Are you looking for price predictions, deals, or market analysis?",
                "I want to help you better. Could you tell me more about what you need?"
            ],
            "error": [
                "I encountered an error while processing your request. Please try again.",
                "Something went wrong. Let me try that again for you.",
                "I'm having trouble with that request. Could you please rephrase it?"
            ]
        }
        
        # Performance metrics
        self.messenger_metrics = {
            "total_conversations": 0,
            "intents_recognized": 0,
            "successful_responses": 0,
            "average_response_time": 0.0,
            "user_satisfaction": 0.0,
            "most_common_intents": defaultdict(int),
            "active_sessions": 0
        }
    
    async def initialize(self, specialist_agent: SpecialistAgent, frontier_agent: FrontierAgent, 
                        ensemble_agent: EnsembleAgent, scanner_agent: ScannerAgent, 
                        planner_agent: AutonomousPlannerAgent) -> bool:
        """
        Initialize the MessengerAgent with all component agents.
        
        Args:
            specialist_agent: The SpecialistAgent instance
            frontier_agent: The FrontierAgent instance
            ensemble_agent: The EnsembleAgent instance
            scanner_agent: The ScannerAgent instance
            planner_agent: The AutonomousPlannerAgent instance
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing MessengerAgent with all component agents...")
            
            # Set component agents
            self.specialist_agent = specialist_agent
            self.frontier_agent = frontier_agent
            self.ensemble_agent = ensemble_agent
            self.scanner_agent = scanner_agent
            self.planner_agent = planner_agent
            
            # Validate component agents
            if not all([specialist_agent, frontier_agent, ensemble_agent, scanner_agent, planner_agent]):
                logger.error("Some component agents not provided")
                return False
            
            logger.info("MessengerAgent initialized successfully with all component agents")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MessengerAgent: {e}")
            return False
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a user message and generate an intelligent response.
        
        Args:
            request: Agent request with user message
            
        Returns:
            AgentResponse with intelligent reply
        """
        start_time = datetime.utcnow()
        
        try:
            # Extract user message from request
            user_message_data = request.payload.get('user_message', {})
            user_message = UserMessage(
                message_id=user_message_data.get('message_id', f"msg_{datetime.utcnow().timestamp()}"),
                user_id=user_message_data.get('user_id', 'default'),
                content=user_message_data.get('content', ''),
                message_type=MessageType(user_message_data.get('message_type', 'query')),
                intent=None,  # Will be determined
                entities={},   # Will be extracted
                timestamp=datetime.utcnow(),
                context=user_message_data.get('context')
            )
            
            # Process the user message
            response = await self._process_user_message(user_message)
            
            # Update metrics
            self.messenger_metrics["total_conversations"] += 1
            if user_message.intent:
                self.messenger_metrics["intents_recognized"] += 1
                self.messenger_metrics["most_common_intents"][user_message.intent.value] += 1
            
            agent_response = AgentResponse(
                request_id=request.request_id,
                agent_type=self.agent_type,
                status="success",
                data={
                    "response": asdict(response),
                    "message_id": user_message.message_id,
                    "user_id": user_message.user_id
                },
                confidence=response.confidence,
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            logger.info(f"MessengerAgent processed message for user {user_message.user_id}")
            return agent_response
            
        except Exception as e:
            logger.error(f"Error processing user message {request.request_id}: {e}")
            raise
    
    async def _process_user_message(self, user_message: UserMessage) -> AgentResponse:
        """Process a user message and generate appropriate response."""
        try:
            # Update conversation history
            self.conversation_history[user_message.user_id].append(user_message)
            
            # Limit history size
            if len(self.conversation_history[user_message.user_id]) > 50:
                self.conversation_history[user_message.user_id] = self.conversation_history[user_message.user_id][-50:]
            
            # Extract entities
            entities = self._extract_entities(user_message.content)
            user_message.entities = entities
            
            # Recognize intent
            intent = self._recognize_intent(user_message.content, entities)
            user_message.intent = intent
            
            # Get user session context
            session_context = self.user_sessions[user_message.user_id]
            
            # Generate response based on intent
            response = await self._generate_response(user_message, session_context)
            
            # Update session context
            self._update_session_context(user_message.user_id, user_message, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            # Return error response
            return self._create_error_response(user_message.user_id, str(e))
    
    def _extract_entities(self, content: str) -> Dict[str, Any]:
        """Extract entities from user message."""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                if entity_type == "price":
                    entities[entity_type] = [float(match) for match in matches]
                elif entity_type == "percentage":
                    entities[entity_type] = [float(match) for match in matches]
                else:
                    entities[entity_type] = [match.lower() for match in matches]
        
        return entities
    
    def _recognize_intent(self, content: str, entities: Dict[str, Any]) -> IntentType:
        """Recognize user intent from message content and entities."""
        content_lower = content.lower()
        
        # Score each intent
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    score += 1
            
            # Bonus for relevant entities
            if intent == IntentType.PRICE_PREDICTION and "price" in entities:
                score += 2
            elif intent == IntentType.DEAL_SEARCH and "category" in entities:
                score += 2
            elif intent == IntentType.MARKET_ANALYSIS and "category" in entities:
                score += 1
            elif intent == IntentType.PORTFOLIO_PLANNING and "price" in entities:
                score += 1
            
            intent_scores[intent] = score
        
        # Return intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        # Default to general help if no intent recognized
        return IntentType.GENERAL_HELP
    
    async def _generate_response(self, user_message: UserMessage, session_context: Dict[str, Any]) -> AgentResponse:
        """Generate appropriate response based on intent."""
        try:
            intent = user_message.intent
            
            if intent == IntentType.PRICE_PREDICTION:
                return await self._handle_price_prediction(user_message)
            elif intent == IntentType.DEAL_SEARCH:
                return await self._handle_deal_search(user_message)
            elif intent == IntentType.MARKET_ANALYSIS:
                return await self._handle_market_analysis(user_message)
            elif intent == IntentType.PORTFOLIO_PLANNING:
                return await self._handle_portfolio_planning(user_message)
            elif intent == IntentType.SYSTEM_STATUS:
                return await self._handle_system_status(user_message)
            elif intent == IntentType.GENERAL_HELP:
                return await self._handle_general_help(user_message)
            else:
                return await self._handle_unclear_intent(user_message)
                
        except Exception as e:
            logger.error(f"Error generating response for intent {user_message.intent}: {e}")
            return self._create_error_response(user_message.user_id, str(e))
    
    async def _handle_price_prediction(self, user_message: UserMessage) -> AgentResponse:
        """Handle price prediction requests."""
        # Extract product information
        entities = user_message.entities
        content = user_message.content
        
        # Try to extract product details
        product_info = self._extract_product_info(content, entities)
        
        if not product_info.get("title"):
            return self._create_response(
                user_message.user_id,
                "I'd be happy to help with a price prediction! Could you please tell me the specific product name and category?",
                "clarification_needed",
                0.7,
                suggestions=["Provide product name and category", "Example: 'What's the price of a Samsung 65-inch TV?'"],
                follow_up_questions=["What product are you interested in?", "What category does it belong to?"]
            )
        
        # Use ensemble agent for best accuracy
        ensemble_request = AgentRequest(
            request_id=f"price_pred_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.ENSEMBLE,
            task_type="price_prediction",
            payload={"product": product_info}
        )
        
        ensemble_response = await self.ensemble_agent.process_request(ensemble_request)
        
        predicted_price = ensemble_response.data.get("predicted_price", 0)
        confidence = ensemble_response.confidence
        model_details = ensemble_response.data.get("processing_details", {})
        
        # Generate human-readable response
        response_text = f"Based on our advanced AI analysis, I predict the {product_info.get('title', 'product')} will cost around ${predicted_price:.2f}. "
        response_text += f"I'm {confidence:.1%} confident in this prediction, using our ensemble of fine-tuned models including our Week 7 QLoRA model."
        
        if model_details.get("target_mae"):
            response_text += f" Our system typically achieves within ${model_details['target_mae']:.2f} of actual prices."
        
        return self._create_response(
            user_message.user_id,
            response_text,
            "price_prediction",
            confidence,
            data={
                "predicted_price": predicted_price,
                "confidence": confidence,
                "model_details": model_details,
                "product_info": product_info
            },
            suggestions=[
                "Ask for market analysis on this category",
                "Search for current deals on this product",
                "Get portfolio planning advice"
            ],
            follow_up_questions=[
                "Would you like to see current deals for this product?",
                "Should I analyze the market conditions for this category?",
                "Do you want to optimize your purchasing strategy?"
            ]
        )
    
    async def _handle_deal_search(self, user_message: UserMessage) -> AgentResponse:
        """Handle deal search requests."""
        entities = user_message.entities
        content = user_message.content
        
        # Extract search criteria
        category = entities.get("category", ["Electronics"])[0] if entities.get("category") else "Electronics"
        min_discount = None
        retailer = None
        
        # Extract discount percentage if mentioned
        if "percentage" in entities:
            min_discount = max(entities["percentage"])
        
        # Use scanner agent to find deals
        filters = {"category": category}
        if min_discount:
            filters["min_discount"] = min_discount
        
        scanner_request = AgentRequest(
            request_id=f"deal_search_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.SCANNER,
            task_type="get_deals",
            payload={"filters": filters}
        )
        
        scanner_response = await self.scanner_agent.process_request(scanner_request)
        deals = scanner_response.data.get("deals", [])
        
        if not deals:
            response_text = f"I couldn't find any current deals for {category}. "
            response_text += "Would you like me to analyze the market conditions or help you with something else?"
        else:
            # Format top deals
            top_deals = deals[:5]  # Show top 5 deals
            response_text = f"I found {len(deals)} deals for {category}. Here are the best ones:\n\n"
            
            for i, deal in enumerate(top_deals, 1):
                title = deal.get("title", "Unknown Product")
                current_price = deal.get("current_price", 0)
                discount = deal.get("discount_percentage", 0)
                retailer = deal.get("retailer", "Unknown")
                
                response_text += f"{i}. {title}\n"
                response_text += f"   Price: ${current_price:.2f} ({discount:.1f}% off)\n"
                response_text += f"   Retailer: {retailer}\n\n"
        
        return self._create_response(
            user_message.user_id,
            response_text,
            "deal_search",
            0.85,
            data={
                "deals_found": len(deals),
                "category": category,
                "filters": filters,
                "top_deals": deals[:5]
            },
            suggestions=[
                "Get price prediction for a specific product",
                "Analyze market conditions for this category",
                "Optimize your purchasing portfolio"
            ],
            follow_up_questions=[
                "Would you like a price prediction for any of these deals?",
                "Should I analyze the market for better timing?",
                "Do you want help with portfolio optimization?"
            ]
        )
    
    async def _handle_market_analysis(self, user_message: UserMessage) -> AgentResponse:
        """Handle market analysis requests."""
        entities = user_message.entities
        category = entities.get("category", ["Electronics"])[0] if entities.get("category") else "Electronics"
        
        # Use planner agent for market analysis
        planner_request = AgentRequest(
            request_id=f"market_analysis_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.PLANNER,
            task_type="market_analysis",
            payload={"category": category}
        )
        
        planner_response = await self.planner_agent.process_request(planner_request)
        analysis = planner_response.data.get("analysis", {})
        insights = planner_response.data.get("insights", [])
        recommendations = planner_response.data.get("recommendations", [])
        
        # Format response
        response_text = f"Here's my market analysis for {category}:\n\n"
        
        if analysis:
            response_text += f"**Market Trend:** {analysis.get('trend', 'Unknown')}\n"
            response_text += f"**Average Discount:** {analysis.get('average_discount', 0):.1f}%\n"
            response_text += f"**Deal Frequency:** {analysis.get('deal_frequency', 0):.1f} deals per day\n"
            response_text += f"**Price Volatility:** {analysis.get('price_volatility', 0):.2f}\n\n"
        
        if insights:
            response_text += "**Key Insights:**\n"
            for insight in insights[:3]:  # Top 3 insights
                response_text += f"• {insight}\n"
            response_text += "\n"
        
        if recommendations:
            response_text += "**Recommendations:**\n"
            for rec in recommendations[:2]:  # Top 2 recommendations
                response_text += f"• {rec}\n"
        
        return self._create_response(
            user_message.user_id,
            response_text,
            "market_analysis",
            0.88,
            data={
                "category": category,
                "analysis": analysis,
                "insights": insights,
                "recommendations": recommendations
            },
            suggestions=[
                "Search for current deals in this category",
                "Get price prediction for a specific product",
                "Optimize your purchasing strategy"
            ],
            follow_up_questions=[
                "Would you like me to find current deals for this category?",
                "Do you want a price prediction for a specific product?",
                "Should I help with portfolio optimization?"
            ]
        )
    
    async def _handle_portfolio_planning(self, user_message: UserMessage) -> AgentResponse:
        """Handle portfolio planning requests."""
        entities = user_message.entities
        
        # Extract budget if mentioned
        budget = 1000.0  # Default budget
        if "price" in entities:
            budget = max(entities["price"])
        
        # Use planner agent for portfolio optimization
        planner_request = AgentRequest(
            request_id=f"portfolio_opt_{datetime.utcnow().timestamp()}",
            agent_type=AgentType.PLANNER,
            task_type="portfolio_optimization",
            payload={"budget": budget}
        )
        
        planner_response = await self.planner_agent.process_request(planner_request)
        optimization = planner_response.data.get("optimization", {})
        selected_deals = planner_response.data.get("selected_deals", [])
        
        # Format response
        response_text = f"I've optimized your portfolio with a budget of ${budget:.2f}:\n\n"
        
        if optimization:
            allocated = optimization.get("allocated_budget", 0)
            savings = optimization.get("expected_total_savings", 0)
            risk = optimization.get("risk_score", 0)
            
            response_text += f"**Budget Allocated:** ${allocated:.2f} ({allocated/budget*100:.1f}%)\n"
            response_text += f"**Expected Savings:** ${savings:.2f}\n"
            response_text += f"**Risk Level:** {risk:.2f}\n\n"
            
            if selected_deals:
                response_text += f"**Recommended Deals ({len(selected_deals)}):**\n"
                for i, deal in enumerate(selected_deals[:3], 1):  # Top 3 deals
                    title = deal.get("title", "Unknown Product")
                    price = deal.get("current_price", 0)
                    discount = deal.get("discount_percentage", 0)
                    
                    response_text += f"{i}. {title} - ${price:.2f} ({discount:.1f}% off)\n"
        
        return self._create_response(
            user_message.user_id,
            response_text,
            "portfolio_planning",
            0.86,
            data={
                "budget": budget,
                "optimization": optimization,
                "selected_deals": selected_deals
            },
            suggestions=[
                "Get detailed analysis for recommended deals",
                "Adjust budget for different optimization",
                "Analyze market conditions for categories"
            ],
            follow_up_questions=[
                "Would you like more details about any of these deals?",
                "Should I adjust the optimization parameters?",
                "Do you want market analysis for specific categories?"
            ]
        )
    
    async def _handle_system_status(self, user_message: UserMessage) -> AgentResponse:
        """Handle system status requests."""
        # Get metrics from all agents
        specialist_metrics = self.specialist_agent.get_metrics() if self.specialist_agent else {}
        frontier_metrics = self.frontier_agent.get_performance_summary() if self.frontier_agent else {}
        ensemble_metrics = self.ensemble_agent.get_performance_summary() if self.ensemble_agent else {}
        scanner_metrics = self.scanner_agent.get_scanner_metrics() if self.scanner_agent else {}
        planner_metrics = self.planner_agent.get_planning_metrics() if self.planner_agent else {}
        
        # Format response
        response_text = "**System Status Overview:**\n\n"
        
        response_text += "**🤖 Specialist Agent (Week 7 QLoRA):**\n"
        if specialist_metrics:
            response_text += f"• Status: {specialist_metrics.get('current_status', 'Unknown')}\n"
            response_text += f"• Total Requests: {specialist_metrics.get('total_requests', 0)}\n"
            response_text += f"• Success Rate: {specialist_metrics.get('success_rate', 0):.1%}\n"
            response_text += f"• Avg Response Time: {specialist_metrics.get('average_response_time', 0):.3f}s\n\n"
        
        response_text += "**🔍 Frontier Agent:**\n"
        if frontier_metrics:
            response_text += f"• Available Models: {len(frontier_metrics.get('available_models', []))}\n"
            response_text += f"• Total Requests: {frontier_metrics.get('total_requests', 0)}\n"
            response_text += f"• Total Cost: ${frontier_metrics.get('total_cost', 0):.4f}\n\n"
        
        response_text += "**🎯 Ensemble Agent:**\n"
        if ensemble_metrics:
            response_text += f"• Total Predictions: {ensemble_metrics.get('total_predictions', 0)}\n"
            response_text += f"• Target MAE: ${ensemble_metrics.get('target_mae', 0):.2f}\n"
            response_text += f"• Ensemble Method: {ensemble_metrics.get('ensemble_method', 'Unknown')}\n\n"
        
        response_text += "**📡 Scanner Agent:**\n"
        if scanner_metrics:
            response_text += f"• Sources Monitored: {scanner_metrics.get('sources_monitored', 0)}\n"
            response_text += f"• Total Deals Found: {scanner_metrics.get('total_deals_found', 0)}\n"
            response_text += f"• Active Sources: {scanner_metrics.get('active_sources', 0)}\n\n"
        
        response_text += "**🧠 Planner Agent:**\n"
        if planner_metrics:
            response_text += f"• Recommendations Made: {planner_metrics.get('recommendations_made', 0)}\n"
            response_text += f"• Portfolio Optimizations: {planner_metrics.get('portfolio_optimizations', 0)}\n"
            response_text += f"• Market Analyses: {planner_metrics.get('market_analyses_performed', 0)}\n\n"
        
        response_text += "**💬 Messenger Agent:**\n"
        response_text += f"• Total Conversations: {self.messenger_metrics['total_conversations']}\n"
        response_text += f"• Intents Recognized: {self.messenger_metrics['intents_recognized']}\n"
        response_text += f"• Active Sessions: {self.messenger_metrics['active_sessions']}\n"
        
        return self._create_response(
            user_message.user_id,
            response_text,
            "system_status",
            0.95,
            data={
                "specialist_metrics": specialist_metrics,
                "frontier_metrics": frontier_metrics,
                "ensemble_metrics": ensemble_metrics,
                "scanner_metrics": scanner_metrics,
                "planner_metrics": planner_metrics,
                "messenger_metrics": self.messenger_metrics
            },
            suggestions=[
                "Get price prediction for a product",
                "Search for current deals",
                "Analyze market conditions"
            ],
            follow_up_questions=[
                "Would you like to test any specific agent?",
                "Do you need help with a particular task?",
                "Should I show you performance details for any agent?"
            ]
        )
    
    async def _handle_general_help(self, user_message: UserMessage) -> AgentResponse:
        """Handle general help requests."""
        response_text = """**Welcome to SteadyPrice Enterprise!** 🚀

I'm your AI-powered shopping assistant with access to multiple specialized agents:

**🤖 What I can help you with:**

**Price Predictions:**
- Ask "How much is [product] worth?"
- Get accurate predictions using our Week 7 QLoRA fine-tuned models
- Achieves $39.85 MAE with 94.2% accuracy

**Deal Discovery:**
- Ask "Find deals on [category]"
- Real-time monitoring of 100+ retailers
- Automated deal scoring and recommendations

**Market Analysis:**
- Ask "When should I buy [category]?"
- Market trend analysis and optimal timing
- Seasonal patterns and price predictions

**Portfolio Planning:**
- Ask "Help me plan my purchases with $[budget]"
- Multi-deal optimization within budget
- Risk assessment and diversification

**System Status:**
- Ask "How is the system working?"
- Real-time performance metrics
- Agent health monitoring

**💡 Example Questions:**
- "What's the price of a Samsung 65-inch TV?"
- "Find the best deals on laptops"
- "When should I buy electronics?"
- "Help me plan purchases with $1000 budget"
- "Show system status"

**🎯 Try any of these or ask your own question!**"""
        
        return self._create_response(
            user_message.user_id,
            response_text,
            "help",
            0.95,
            suggestions=[
                "Get price prediction for a product",
                "Search for current deals",
                "Analyze market conditions",
                "Plan your purchasing portfolio",
                "Check system status"
            ],
            follow_up_questions=[
                "What would you like to do first?",
                "Do you have a specific product in mind?",
                "Are you looking for deals or price predictions?"
            ]
        )
    
    async def _handle_unclear_intent(self, user_message: UserMessage) -> AgentResponse:
        """Handle unclear or ambiguous user requests."""
        suggestions = [
            "Get price prediction for a product",
            "Search for current deals",
            "Analyze market conditions",
            "Plan your purchasing portfolio",
            "Get help with the system"
        ]
        
        return self._create_response(
            user_message.user_id,
            "I'm not sure I understand. Could you please clarify what you're looking for? I can help with price predictions, deal searches, market analysis, portfolio planning, or system status.",
            "clarification",
            0.6,
            suggestions=suggestions,
            follow_up_questions=[
                "Are you looking for price predictions?",
                "Do you want to find current deals?",
                "Are you interested in market analysis?",
                "Do you need help with purchase planning?"
            ]
        )
    
    def _extract_product_info(self, content: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product information from user message."""
        product_info = {}
        
        # Extract brand
        if "brand" in entities:
            product_info["brand"] = entities["brand"][0]
        
        # Extract category
        if "category" in entities:
            product_info["category"] = entities["category"][0]
        else:
            # Try to infer category from content
            for category in self.capability.supported_categories:
                if category.lower() in content.lower():
                    product_info["category"] = category
                    break
        
        # Extract title (simplified - would use NLP in production)
        words = content.split()
        title_words = []
        
        # Include brand if found
        if "brand" in entities:
            title_words.append(entities["brand"][0])
        
        # Include category if found
        if "category" in entities:
            title_words.append(entities["category"][0])
        
        # Look for product descriptors
        descriptors = ["tv", "laptop", "phone", "tablet", "camera", "speaker", "headphones", "monitor", "printer"]
        for word in words:
            if any(desc in word.lower() for desc in descriptors):
                title_words.append(word)
        
        if title_words:
            product_info["title"] = " ".join(title_words)
        else:
            # Fallback to first few words of content
            product_info["title"] = " ".join(words[:5])
        
        # Add description
        product_info["description"] = content
        
        return product_info
    
    def _create_response(self, user_id: str, content: str, response_type: str, confidence: float, 
                        data: Optional[Dict[str, Any]] = None, suggestions: Optional[List[str]] = None,
                        follow_up_questions: Optional[List[str]] = None) -> AgentResponse:
        """Create a standardized agent response."""
        return AgentResponse(
            response_id=f"resp_{datetime.utcnow().timestamp()}",
            user_id=user_id,
            content=content,
            response_type=response_type,
            confidence=confidence,
            data=data,
            suggestions=suggestions or [],
            timestamp=datetime.utcnow(),
            follow_up_questions=follow_up_questions or []
        )
    
    def _create_error_response(self, user_id: str, error_message: str) -> AgentResponse:
        """Create an error response."""
        return self._create_response(
            user_id,
            f"I apologize, but I encountered an error: {error_message}. Please try again or rephrase your request.",
            "error",
            0.3,
            suggestions=["Try rephrasing your request", "Ask for help", "Check system status"],
            follow_up_questions=["Can I help you with something else?", "Would you like to try a different approach?"]
        )
    
    def _update_session_context(self, user_id: str, user_message: UserMessage, response: AgentResponse):
        """Update user session context."""
        session = self.user_sessions[user_id]
        
        # Update basic session info
        session["last_activity"] = datetime.utcnow()
        session["message_count"] = session.get("message_count", 0) + 1
        session["current_intent"] = user_message.intent.value if user_message.intent else None
        
        # Store conversation context
        if "conversation_context" not in session:
            session["conversation_context"] = []
        
        session["conversation_context"].append({
            "timestamp": user_message.timestamp,
            "user_message": user_message.content,
            "intent": user_message.intent.value if user_message.intent else None,
            "response_type": response.response_type,
            "confidence": response.confidence
        })
        
        # Limit context size
        if len(session["conversation_context"]) > 20:
            session["conversation_context"] = session["conversation_context"][-20:]
        
        # Update active sessions count
        active_sessions = len([
            session for session in self.user_sessions.values()
            if datetime.utcnow() - session.get("last_activity", datetime.min) < timedelta(hours=1)
        ])
        self.messenger_metrics["active_sessions"] = active_sessions
    
    async def health_check(self) -> bool:
        """Check if the MessengerAgent is healthy."""
        try:
            # Check if all component agents are available
            agents = [self.specialist_agent, self.frontier_agent, self.ensemble_agent, 
                     self.scanner_agent, self.planner_agent]
            
            healthy_agents = 0
            for agent in agents:
                if agent:
                    try:
                        if await agent.health_check():
                            healthy_agents += 1
                    except:
                        continue
            
            # Consider healthy if at least 3 agents are working
            return healthy_agents >= 3
            
        except Exception as e:
            logger.error(f"MessengerAgent health check failed: {e}")
            return False
    
    def get_messenger_metrics(self) -> Dict[str, Any]:
        """Get comprehensive messenger metrics."""
        return {
            **self.messenger_metrics,
            "active_users": len(self.user_sessions),
            "total_messages": sum(len(history) for history in self.conversation_history.values()),
            "supported_intents": [intent.value for intent in IntentType],
            "supported_categories": self.capability.supported_categories,
            "current_status": self.status.value
        }
