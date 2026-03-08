"""
Base Agent Framework for SteadyPrice Multi-Agent System

This module provides the foundation for all specialized agents in the
Week 8 transformative multi-agent architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Enumeration of different agent types"""
    SPECIALIST = "specialist"
    FRONTIER = "frontier"
    ENSEMBLE = "ensemble"
    SCANNER = "scanner"
    PLANNER = "planner"
    MESSENGER = "messenger"

class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class AgentCapability:
    """Defines agent capabilities and specifications"""
    name: str
    description: str
    max_concurrent_tasks: int
    average_response_time: float
    accuracy_metric: float
    cost_per_request: float
    supported_categories: List[str]

@dataclass
class AgentRequest:
    """Standardized request format for all agents"""
    request_id: str
    agent_type: AgentType
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class AgentResponse:
    """Standardized response format from all agents"""
    request_id: str
    agent_type: AgentType
    status: str
    data: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the SteadyPrice system.
    
    Provides common functionality and enforces consistent interface
    across all specialized agents.
    """
    
    def __init__(self, agent_type: AgentType, capability: AgentCapability):
        self.agent_type = agent_type
        self.capability = capability
        self.status = AgentStatus.IDLE
        self.request_queue = asyncio.Queue()
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'last_updated': datetime.utcnow()
        }
        
    @abstractmethod
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """
        Process a request and return a response.
        
        Args:
            request: The agent request to process
            
        Returns:
            AgentResponse with results and metadata
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent with necessary resources.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the agent is healthy and operational.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    async def start_processing(self):
        """Start the main processing loop for the agent."""
        logger.info(f"Starting {self.agent_type.value} agent processing loop")
        
        while True:
            try:
                # Get next request from queue
                request = await asyncio.wait_for(
                    self.request_queue.get(), 
                    timeout=1.0
                )
                
                # Process the request
                self.status = AgentStatus.PROCESSING
                start_time = datetime.utcnow()
                
                try:
                    response = await self.process_request(request)
                    
                    # Update performance metrics
                    processing_time = (datetime.utcnow() - start_time).total_seconds()
                    self._update_metrics(processing_time, success=True)
                    
                    # Send response (in production, this would go to a response handler)
                    await self._handle_response(response)
                    
                except Exception as e:
                    logger.error(f"Error processing request {request.request_id}: {e}")
                    self._update_metrics(0.0, success=False)
                    
                    # Create error response
                    error_response = AgentResponse(
                        request_id=request.request_id,
                        agent_type=self.agent_type,
                        status="error",
                        data={"error": str(e)},
                        confidence=0.0,
                        processing_time=0.0
                    )
                    await self._handle_response(error_response)
                
                finally:
                    self.status = AgentStatus.IDLE
                    
            except asyncio.TimeoutError:
                # No requests in queue, continue
                continue
            except Exception as e:
                logger.error(f"Unexpected error in processing loop: {e}")
                self.status = AgentStatus.ERROR
                await asyncio.sleep(5)  # Wait before retrying
    
    async def submit_request(self, request: AgentRequest) -> bool:
        """
        Submit a request to the agent's processing queue.
        
        Args:
            request: The request to submit
            
        Returns:
            True if successfully queued, False if queue is full
        """
        try:
            self.request_queue.put_nowait(request)
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full for {self.agent_type.value} agent")
            return False
    
    async def _handle_response(self, response: AgentResponse):
        """
        Handle agent response (placeholder for actual implementation).
        
        In production, this would send the response to the appropriate
        response handler or callback mechanism.
        """
        logger.debug(f"Handling response for {response.request_id}")
        # Implementation would depend on the specific architecture
        pass
    
    def _update_metrics(self, processing_time: float, success: bool):
        """Update internal performance metrics."""
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
            
            # Update average response time
            total = self.performance_metrics['total_requests']
            current_avg = self.performance_metrics['average_response_time']
            new_avg = ((current_avg * (total - 1)) + processing_time) / total
            self.performance_metrics['average_response_time'] = new_avg
        
        self.performance_metrics['last_updated'] = datetime.utcnow()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.performance_metrics,
            'success_rate': (
                self.performance_metrics['successful_requests'] / 
                max(1, self.performance_metrics['total_requests'])
            ),
            'current_status': self.status.value,
            'queue_size': self.request_queue.qsize(),
            'capability': self.capability
        }
    
    async def shutdown(self):
        """Gracefully shutdown the agent."""
        logger.info(f"Shutting down {self.agent_type.value} agent")
        self.status = AgentStatus.MAINTENANCE
        
        # Process remaining requests in queue
        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                logger.warning(f"Discarding request {request.request_id} during shutdown")
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Shutdown complete for {self.agent_type.value} agent")

class AgentOrchestrator:
    """
    Orchestrates multiple agents and manages their lifecycle.
    
    This class coordinates the interaction between different agents,
    handles request routing, and manages overall system performance.
    """
    
    def __init__(self):
        self.agents: Dict[AgentType, BaseAgent] = {}
        self.request_router = None
        self.performance_monitor = None
        
    async def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator."""
        self.agents[agent.agent_type] = agent
        logger.info(f"Registered {agent.agent_type.value} agent")
        
    async def initialize_all_agents(self) -> bool:
        """Initialize all registered agents."""
        success = True
        
        for agent_type, agent in self.agents.items():
            try:
                if await agent.initialize():
                    logger.info(f"Successfully initialized {agent_type.value} agent")
                else:
                    logger.error(f"Failed to initialize {agent_type.value} agent")
                    success = False
            except Exception as e:
                logger.error(f"Error initializing {agent_type.value} agent: {e}")
                success = False
        
        return success
    
    async def start_all_agents(self):
        """Start all agents' processing loops."""
        tasks = []
        
        for agent_type, agent in self.agents.items():
            task = asyncio.create_task(agent.start_processing())
            tasks.append(task)
            logger.info(f"Started processing for {agent_type.value} agent")
        
        # Wait for all tasks (in production, this would be managed differently)
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def shutdown_all_agents(self):
        """Gracefully shutdown all agents."""
        for agent_type, agent in self.agents.items():
            await agent.shutdown()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        metrics = {
            'total_agents': len(self.agents),
            'agents': {}
        }
        
        for agent_type, agent in self.agents.items():
            metrics['agents'][agent_type.value] = agent.get_metrics()
        
        return metrics
