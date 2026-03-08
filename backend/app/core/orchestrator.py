"""
Multi-Agent Orchestrator - Week 8 Transformative System

This module orchestrates all agents in the SteadyPrice multi-agent system,
providing unified access and coordination between components.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..agents.base_agent import AgentOrchestrator, AgentType
from ..agents.specialist_agent import SpecialistAgent
from ..agents.frontier_agent import FrontierAgent
from ..agents.ensemble_agent import EnsembleAgent
from ..agents.scanner_agent import ScannerAgent
from ..agents.planner_agent import AutonomousPlannerAgent
from ..agents.messenger_agent import MessengerAgent

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    total_requests: int
    successful_requests: int
    average_response_time: float
    system_uptime: datetime
    agent_health: Dict[str, bool]
    error_rate: float
    throughput: float

class SteadyPriceOrchestrator:
    """
    Main orchestrator for the SteadyPrice Week 8 multi-agent system.
    
    This class coordinates all agents, manages system lifecycle,
    and provides unified access to the transformative capabilities.
    """
    
    def __init__(self):
        # Agent instances
        self.specialist_agent: Optional[SpecialistAgent] = None
        self.frontier_agent: Optional[FrontierAgent] = None
        self.ensemble_agent: Optional[EnsembleAgent] = None
        self.scanner_agent: Optional[ScannerAgent] = None
        self.planner_agent: Optional[AutonomousPlannerAgent] = None
        self.messenger_agent: Optional[MessengerAgent] = None
        
        # Base orchestrator for agent management
        self.base_orchestrator = AgentOrchestrator()
        
        # System metrics
        self.system_metrics = SystemMetrics(
            total_requests=0,
            successful_requests=0,
            average_response_time=0.0,
            system_uptime=datetime.utcnow(),
            agent_health={},
            error_rate=0.0,
            throughput=0.0
        )
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> bool:
        """
        Initialize all agents in the system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing SteadyPrice Multi-Agent System...")
            
            # Create agent instances
            self.specialist_agent = SpecialistAgent()
            self.frontier_agent = FrontierAgent()
            self.ensemble_agent = EnsembleAgent()
            self.scanner_agent = ScannerAgent()
            self.planner_agent = AutonomousPlannerAgent()
            self.messenger_agent = MessengerAgent()
            
            # Initialize individual agents
            specialist_ready = await self.specialist_agent.initialize()
            frontier_ready = await self.frontier_agent.initialize()
            scanner_ready = await self.scanner_agent.initialize()
            
            if not all([specialist_ready, frontier_ready, scanner_ready]):
                logger.error("Failed to initialize core agents")
                return False
            
            # Initialize ensemble agent with component agents
            ensemble_ready = await self.ensemble_agent.initialize(
                self.specialist_agent, self.frontier_agent
            )
            
            if not ensemble_ready:
                logger.error("Failed to initialize ensemble agent")
                return False
            
            # Initialize planner agent with component agents
            planner_ready = await self.planner_agent.initialize(
                self.scanner_agent, self.ensemble_agent
            )
            
            if not planner_ready:
                logger.error("Failed to initialize planner agent")
                return False
            
            # Initialize messenger agent with all component agents
            messenger_ready = await self.messenger_agent.initialize(
                self.specialist_agent, self.frontier_agent, self.ensemble_agent,
                self.scanner_agent, self.planner_agent
            )
            
            if not messenger_ready:
                logger.error("Failed to initialize messenger agent")
                return False
            
            # Register agents with base orchestrator
            await self.base_orchestrator.register_agent(self.specialist_agent)
            await self.base_orchestrator.register_agent(self.frontier_agent)
            await self.base_orchestrator.register_agent(self.ensemble_agent)
            await self.base_orchestrator.register_agent(self.scanner_agent)
            await self.base_orchestrator.register_agent(self.planner_agent)
            await self.base_orchestrator.register_agent(self.messenger_agent)
            
            # Initialize all agents through base orchestrator
            all_ready = await self.base_orchestrator.initialize_all_agents()
            
            if not all_ready:
                logger.error("Failed to initialize all agents through orchestrator")
                return False
            
            self.is_initialized = True
            logger.info("SteadyPrice Multi-Agent System initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    async def start(self) -> bool:
        """
        Start the multi-agent system.
        
        Returns:
            True if startup successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.error("System not initialized")
                return False
            
            logger.info("Starting SteadyPrice Multi-Agent System...")
            self.startup_time = datetime.utcnow()
            
            # Start all agent processing loops
            await self.base_orchestrator.start_all_agents()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            logger.info("SteadyPrice Multi-Agent System started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting system: {e}")
            return False
    
    async def _start_background_tasks(self):
        """Start background system tasks."""
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.append(metrics_task)
        
        # Deal scanning task
        scanning_task = asyncio.create_task(self._periodic_deal_scanning())
        self.background_tasks.append(scanning_task)
        
        logger.info("Background tasks started")
    
    async def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        while self.is_running:
            try:
                await self._check_agent_health()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(300)  # Collect every 5 minutes
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(300)
    
    async def _periodic_deal_scanning(self):
        """Background periodic deal scanning."""
        while self.is_running:
            try:
                if self.scanner_agent:
                    # Trigger deal scanning
                    scan_request = {
                        "task_type": "scan_all",
                        "priority": 1
                    }
                    await self.scanner_agent.submit_request(
                        AgentRequest(
                            request_id=f"periodic_scan_{datetime.utcnow().timestamp()}",
                            agent_type=AgentType.SCANNER,
                            task_type="scan_all",
                            payload=scan_request
                        )
                    )
                
                await asyncio.sleep(900)  # Scan every 15 minutes
            except Exception as e:
                logger.error(f"Error in periodic scanning: {e}")
                await asyncio.sleep(900)
    
    async def _check_agent_health(self):
        """Check health of all agents."""
        health_checks = {
            "specialist": False,
            "frontier": False,
            "ensemble": False,
            "scanner": False,
            "planner": False,
            "messenger": False
        }
        
        agents = {
            "specialist": self.specialist_agent,
            "frontier": self.frontier_agent,
            "ensemble": self.ensemble_agent,
            "scanner": self.scanner_agent,
            "planner": self.planner_agent,
            "messenger": self.messenger_agent
        }
        
        for name, agent in agents.items():
            if agent:
                try:
                    health_checks[name] = await agent.health_check()
                except Exception as e:
                    logger.error(f"Health check failed for {name} agent: {e}")
                    health_checks[name] = False
        
        self.system_metrics.agent_health = health_checks
        
        # Log any unhealthy agents
        unhealthy = [name for name, healthy in health_checks.items() if not healthy]
        if unhealthy:
            logger.warning(f"Unhealthy agents: {unhealthy}")
    
    async def _collect_system_metrics(self):
        """Collect comprehensive system metrics."""
        try:
            # Calculate uptime
            if self.startup_time:
                uptime = datetime.utcnow() - self.startup_time
                self.system_metrics.system_uptime = uptime
            
            # Collect agent-specific metrics
            agent_metrics = {}
            
            if self.specialist_agent:
                agent_metrics["specialist"] = self.specialist_agent.get_metrics()
            
            if self.frontier_agent:
                agent_metrics["frontier"] = self.frontier_agent.get_performance_summary()
            
            if self.ensemble_agent:
                agent_metrics["ensemble"] = self.ensemble_agent.get_performance_summary()
            
            if self.scanner_agent:
                agent_metrics["scanner"] = self.scanner_agent.get_scanner_metrics()
            
            if self.planner_agent:
                agent_metrics["planner"] = self.planner_agent.get_planning_metrics()
            
            if self.messenger_agent:
                agent_metrics["messenger"] = self.messenger_agent.get_messenger_metrics()
            
            # Store metrics (in production, would send to monitoring system)
            logger.info(f"System metrics collected: {len(agent_metrics)} agents")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def process_user_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user request through the appropriate agent.
        
        Args:
            request_data: User request data
            
        Returns:
            Response data
        """
        start_time = datetime.utcnow()
        
        try:
            self.system_metrics.total_requests += 1
            
            # Determine request type and route to appropriate agent
            request_type = request_data.get("request_type", "user_message")
            
            if request_type == "user_message":
                # Route to messenger agent for natural language processing
                response = await self._process_messenger_request(request_data)
            elif request_type == "price_prediction":
                # Route to ensemble agent for best accuracy
                response = await self._process_price_prediction(request_data)
            elif request_type == "deal_search":
                # Route to scanner agent
                response = await self._process_deal_search(request_data)
            elif request_type == "market_analysis":
                # Route to planner agent
                response = await self._process_market_analysis(request_data)
            elif request_type == "portfolio_optimization":
                # Route to planner agent
                response = await self._process_portfolio_optimization(request_data)
            else:
                # Default to messenger agent
                response = await self._process_messenger_request(request_data)
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.system_metrics.successful_requests += 1
            
            # Update average response time
            total = self.system_metrics.total_requests
            current_avg = self.system_metrics.average_response_time
            new_avg = ((current_avg * (total - 1)) + processing_time) / total
            self.system_metrics.average_response_time = new_avg
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user request: {e}")
            self.system_metrics.total_requests += 1  # Count failed request
            
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _process_messenger_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request through messenger agent."""
        if not self.messenger_agent:
            raise RuntimeError("Messenger agent not available")
        
        agent_request = AgentRequest(
            request_id=request_data.get("request_id", f"req_{datetime.utcnow().timestamp()}"),
            agent_type=AgentType.MESSENGER,
            task_type="process_message",
            payload=request_data
        )
        
        response = await self.messenger_agent.process_request(agent_request)
        
        return {
            "status": "success",
            "response_type": "messenger",
            "data": response.data,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_price_prediction(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process price prediction request."""
        if not self.ensemble_agent:
            raise RuntimeError("Ensemble agent not available")
        
        agent_request = AgentRequest(
            request_id=request_data.get("request_id", f"req_{datetime.utcnow().timestamp()}"),
            agent_type=AgentType.ENSEMBLE,
            task_type="price_prediction",
            payload=request_data
        )
        
        response = await self.ensemble_agent.process_request(agent_request)
        
        return {
            "status": "success",
            "response_type": "price_prediction",
            "data": response.data,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_deal_search(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process deal search request."""
        if not self.scanner_agent:
            raise RuntimeError("Scanner agent not available")
        
        agent_request = AgentRequest(
            request_id=request_data.get("request_id", f"req_{datetime.utcnow().timestamp()}"),
            agent_type=AgentType.SCANNER,
            task_type="get_deals",
            payload=request_data
        )
        
        response = await self.scanner_agent.process_request(agent_request)
        
        return {
            "status": "success",
            "response_type": "deal_search",
            "data": response.data,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_market_analysis(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market analysis request."""
        if not self.planner_agent:
            raise RuntimeError("Planner agent not available")
        
        agent_request = AgentRequest(
            request_id=request_data.get("request_id", f"req_{datetime.utcnow().timestamp()}"),
            agent_type=AgentType.PLANNER,
            task_type="market_analysis",
            payload=request_data
        )
        
        response = await self.planner_agent.process_request(agent_request)
        
        return {
            "status": "success",
            "response_type": "market_analysis",
            "data": response.data,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _process_portfolio_optimization(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process portfolio optimization request."""
        if not self.planner_agent:
            raise RuntimeError("Planner agent not available")
        
        agent_request = AgentRequest(
            request_id=request_data.get("request_id", f"req_{datetime.utcnow().timestamp()}"),
            agent_type=AgentType.PLANNER,
            task_type="portfolio_optimization",
            payload=request_data
        )
        
        response = await self.planner_agent.process_request(agent_request)
        
        return {
            "status": "success",
            "response_type": "portfolio_optimization",
            "data": response.data,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Collect agent statuses
            agent_statuses = {}
            
            agents = {
                "specialist": self.specialist_agent,
                "frontier": self.frontier_agent,
                "ensemble": self.ensemble_agent,
                "scanner": self.scanner_agent,
                "planner": self.planner_agent,
                "messenger": self.messenger_agent
            }
            
            for name, agent in agents.items():
                if agent:
                    try:
                        healthy = await agent.health_check()
                        agent_statuses[name] = {
                            "healthy": healthy,
                            "status": agent.status.value,
                            "queue_size": getattr(agent, 'request_queue', None)
                        }
                    except Exception as e:
                        agent_statuses[name] = {
                            "healthy": False,
                            "status": "error",
                            "error": str(e)
                        }
                else:
                    agent_statuses[name] = {
                        "healthy": False,
                        "status": "not_initialized"
                    }
            
            # Calculate system health
            healthy_agents = sum(1 for status in agent_statuses.values() if status.get("healthy", False))
            total_agents = len(agent_statuses)
            system_health = healthy_agents / total_agents if total_agents > 0 else 0.0
            
            return {
                "system_health": system_health,
                "is_initialized": self.is_initialized,
                "is_running": self.is_running,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "uptime": str(datetime.utcnow() - self.startup_time) if self.startup_time else None,
                "agent_statuses": agent_statuses,
                "system_metrics": {
                    "total_requests": self.system_metrics.total_requests,
                    "successful_requests": self.system_metrics.successful_requests,
                    "error_rate": self.calculate_error_rate(),
                    "average_response_time": self.system_metrics.average_response_time,
                    "throughput": self.calculate_throughput()
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def calculate_error_rate(self) -> float:
        """Calculate system error rate."""
        if self.system_metrics.total_requests == 0:
            return 0.0
        
        failed_requests = self.system_metrics.total_requests - self.system_metrics.successful_requests
        return failed_requests / self.system_metrics.total_requests
    
    def calculate_throughput(self) -> float:
        """Calculate system throughput (requests per second)."""
        if not self.startup_time:
            return 0.0
        
        uptime_seconds = (datetime.utcnow() - self.startup_time).total_seconds()
        if uptime_seconds == 0:
            return 0.0
        
        return self.system_metrics.total_requests / uptime_seconds
    
    async def shutdown(self):
        """Gracefully shutdown the multi-agent system."""
        try:
            logger.info("Shutting down SteadyPrice Multi-Agent System...")
            
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown all agents
            await self.base_orchestrator.shutdown_all_agents()
            
            logger.info("SteadyPrice Multi-Agent System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_system_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities and features."""
        return {
            "agents": {
                "specialist": {
                    "description": "Week 7 QLoRA fine-tuned model",
                    "mae": 39.85,
                    "improvement": 44.9,
                    "categories": ["Electronics", "Appliances", "Automotive"]
                },
                "frontier": {
                    "description": "Premium model integration",
                    "models": ["Claude 4.5 Sonnet", "GPT 4.1 Nano"],
                    "mae_values": [47.10, 62.51]
                },
                "ensemble": {
                    "description": "Multi-model fusion",
                    "target_mae": 35.0,
                    "methods": ["weighted_average", "dynamic_weighting", "confidence_based"]
                },
                "scanner": {
                    "description": "Real-time deal discovery",
                    "sources_monitored": 100,
                    "deal_types": ["price_drop", "flash_sale", "coupon", "clearance"]
                },
                "planner": {
                    "description": "Strategic intelligence",
                    "capabilities": ["market_analysis", "timing_optimization", "portfolio_planning"]
                },
                "messenger": {
                    "description": "Natural language interface",
                    "intents": ["price_prediction", "deal_search", "market_analysis", "portfolio_planning"]
                }
            },
            "system_features": {
                "automated_deal_discovery": True,
                "real_time_price_prediction": True,
                "market_trend_analysis": True,
                "portfolio_optimization": True,
                "natural_language_interface": True,
                "multi_model_ensemble": True,
                "strategic_planning": True,
                "enterprise_monitoring": True
            },
            "performance_targets": {
                "ensemble_mae": 35.0,
                "response_time": 0.1,  # 100ms
                "system_uptime": 99.99,
                "concurrent_users": 100000
            },
            "business_impact": {
                "roi_target": 500,  # 500%
                "cost_savings": 100000,  # $100K monthly
                "user_engagement": 10,  # 10x increase
                "market_coverage": 50  # 50+ retailers
            }
        }
