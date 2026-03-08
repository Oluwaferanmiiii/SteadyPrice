"""
Monitoring and Analytics System for SteadyPrice Week 8

This module provides comprehensive monitoring, analytics, and alerting
capabilities for the multi-agent system.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics

# External monitoring libraries
import prometheus_client as prom
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, generate_latest

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str]
    unit: str
    threshold: Optional[float] = None

class MonitoringSystem:
    """
    Comprehensive monitoring and analytics system for the
    SteadyPrice Week 8 multi-agent system.
    """
    
    def __init__(self):
        """Initialize the monitoring system."""
        # Prometheus metrics registry
        self.registry = CollectorRegistry()
        
        # System metrics
        self.metrics: Dict[str, Any] = {}
        self._setup_prometheus_metrics()
        
        # Alert management
        self.alerts: deque = deque(maxlen=1000)
        self.alert_handlers: List[Callable] = []
        
        # Performance tracking
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Health checks
        self.health_checks: Dict[str, Callable] = {}
        
        # Analytics data
        self.analytics_data: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "agent_performance": {},
            "system_load": {},
            "user_activity": {}
        }
        
        # Monitoring configuration
        self.config = {
            "alert_thresholds": {
                "error_rate": 0.05,  # 5%
                "response_time": 1.0,  # 1 second
                "cpu_usage": 0.8,  # 80%
                "memory_usage": 0.85,  # 85%
                "disk_usage": 0.9,  # 90%
            },
            "retention_period": timedelta(days=30),
            "alert_cooldown": timedelta(minutes=5),
            "health_check_interval": 30,  # seconds
        }
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics."""
        # Request metrics
        self.metrics["total_requests"] = Counter(
            'steadyprice_total_requests',
            'Total number of requests',
            ['agent', 'method'],
            registry=self.registry
        )
        
        self.metrics["request_duration"] = Histogram(
            'steadyprice_request_duration_seconds',
            'Request duration in seconds',
            ['agent', 'method'],
            registry=self.registry
        )
        
        self.metrics["active_requests"] = Gauge(
            'steadyprice_active_requests',
            'Number of active requests',
            ['agent'],
            registry=self.registry
        )
        
        # Agent metrics
        self.metrics["agent_health"] = Gauge(
            'steadyprice_agent_health',
            'Agent health status (1=healthy, 0=unhealthy)',
            ['agent'],
            registry=self.registry
        )
        
        self.metrics["agent_queue_size"] = Gauge(
            'steadyprice_agent_queue_size',
            'Agent queue size',
            ['agent'],
            registry=self.registry
        )
        
        # System metrics
        self.metrics["system_cpu_usage"] = Gauge(
            'steadyprice_system_cpu_usage',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics["system_memory_usage"] = Gauge(
            'steadyprice_system_memory_usage',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.metrics["system_disk_usage"] = Gauge(
            'steadyprice_system_disk_usage',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Business metrics
        self.metrics["predictions_made"] = Counter(
            'steadyprice_predictions_made',
            'Total number of price predictions made',
            ['model'],
            registry=self.registry
        )
        
        self.metrics["deals_found"] = Counter(
            'steadyprice_deals_found',
            'Total number of deals found',
            ['retailer', 'category'],
            registry=self.registry
        )
        
        self.metrics["user_satisfaction'] = Gauge(
            'steadyprice_user_satisfaction',
            'User satisfaction score',
            registry=self.registry
        )
    
    async def start(self):
        """Start the monitoring system."""
        try:
            logger.info("Starting monitoring system...")
            
            self.is_running = True
            
            # Start background monitoring tasks
            await self._start_background_tasks()
            
            # Register health checks
            self._register_health_checks()
            
            logger.info("Monitoring system started successfully")
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # System metrics collection
        metrics_task = asyncio.create_task(self._collect_system_metrics_loop())
        self.background_tasks.append(metrics_task)
        
        # Health monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Alert processing
        alert_task = asyncio.create_task(self._alert_processing_loop())
        self.background_tasks.append(alert_task)
        
        # Analytics aggregation
        analytics_task = asyncio.create_task(self._analytics_aggregation_loop())
        self.background_tasks.append(analytics_task)
    
    async def _collect_system_metrics_loop(self):
        """Background loop for collecting system metrics."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                logger.error(f"Error in system metrics collection: {e}")
                await asyncio.sleep(10)
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.metrics["system_cpu_usage"].set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics["system_memory_usage"].set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics["system_disk_usage"].set(disk.percent)
            
            # Store in performance data
            timestamp = datetime.utcnow()
            self.performance_data["cpu_usage"].append((timestamp, cpu_percent))
            self.performance_data["memory_usage"].append((timestamp, memory.percent))
            self.performance_data["disk_usage"].append((timestamp, disk.percent))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _health_monitoring_loop(self):
        """Background loop for health monitoring."""
        while self.is_running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.config["health_check_interval"])
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.config["health_check_interval"])
    
    async def _run_health_checks(self):
        """Run all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                is_healthy = await check_func()
                
                # Update agent health metric
                if "agent" in name.lower():
                    agent_name = name.split("_")[0]
                    self.metrics["agent_health"].labels(agent=agent_name).set(1 if is_healthy else 0)
                
                # Create alert if unhealthy
                if not is_healthy:
                    await self._create_alert(
                        AlertSeverity.ERROR,
                        f"Health Check Failed: {name}",
                        f"Health check {name} failed",
                        name
                    )
                
            except Exception as e:
                logger.error(f"Error in health check {name}: {e}")
                await self._create_alert(
                    AlertSeverity.ERROR,
                    f"Health Check Error: {name}",
                    f"Health check {name} encountered an error: {str(e)}",
                    name
                )
    
    async def _alert_processing_loop(self):
        """Background loop for processing alerts."""
        while self.is_running:
            try:
                await self._process_alerts()
                await asyncio.sleep(60)  # Process alerts every minute
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                await asyncio.sleep(60)
    
    async def _process_alerts(self):
        """Process pending alerts."""
        current_time = datetime.utcnow()
        
        # Check for alert conditions
        await self._check_error_rate()
        await self._check_response_time()
        await self._check_system_resources()
        
        # Clean old alerts
        cutoff_time = current_time - self.config["retention_period"]
        while self.alerts and self.alerts[0].timestamp < cutoff_time:
            self.alerts.popleft()
    
    async def _check_error_rate(self):
        """Check error rate and create alerts if needed."""
        if self.analytics_data["total_requests"] > 0:
            error_rate = self.analytics_data["failed_requests"] / self.analytics_data["total_requests"]
            
            if error_rate > self.config["alert_thresholds"]["error_rate"]:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    "High Error Rate",
                    f"Error rate is {error_rate:.2%}, threshold is {self.config['alert_thresholds']['error_rate']:.2%}",
                    "system"
                )
    
    async def _check_response_time(self):
        """Check response time and create alerts if needed."""
        avg_response_time = self.analytics_data["average_response_time"]
        
        if avg_response_time > self.config["alert_thresholds"]["response_time"]:
            await self._create_alert(
                AlertSeverity.WARNING,
                "High Response Time",
                f"Average response time is {avg_response_time:.2f}s, threshold is {self.config['alert_thresholds']['response_time']}s",
                "system"
            )
    
    async def _check_system_resources(self):
        """Check system resource usage."""
        try:
            import psutil
            
            # Check CPU
            cpu_percent = psutil.cpu_percent()
            if cpu_percent > self.config["alert_thresholds"]["cpu_usage"] * 100:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    "High CPU Usage",
                    f"CPU usage is {cpu_percent:.1f}%, threshold is {self.config['alert_thresholds']['cpu_usage']*100:.1f}%",
                    "system"
                )
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.percent > self.config["alert_thresholds"]["memory_usage"] * 100:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    "High Memory Usage",
                    f"Memory usage is {memory.percent:.1f}%, threshold is {self.config['alert_thresholds']['memory_usage']*100:.1f}%",
                    "system"
                )
            
            # Check disk
            disk = psutil.disk_usage('/')
            if disk.percent > self.config["alert_thresholds"]["disk_usage"] * 100:
                await self._create_alert(
                    AlertSeverity.ERROR,
                    "High Disk Usage",
                    f"Disk usage is {disk.percent:.1f}%, threshold is {self.config['alert_thresholds']['disk_usage']*100:.1f}%",
                    "system"
                )
                
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
    
    async def _analytics_aggregation_loop(self):
        """Background loop for analytics aggregation."""
        while self.is_running:
            try:
                await self._aggregate_analytics()
                await asyncio.sleep(300)  # Aggregate every 5 minutes
            except Exception as e:
                logger.error(f"Error in analytics aggregation: {e}")
                await asyncio.sleep(300)
    
    async def _aggregate_analytics(self):
        """Aggregate analytics data."""
        try:
            # Calculate averages from performance data
            if self.performance_data["cpu_usage"]:
                cpu_values = [value for _, value in self.performance_data["cpu_usage"]]
                self.analytics_data["system_load"]["avg_cpu"] = statistics.mean(cpu_values)
                self.analytics_data["system_load"]["max_cpu"] = max(cpu_values)
            
            if self.performance_data["memory_usage"]:
                memory_values = [value for _, value in self.performance_data["memory_usage"]]
                self.analytics_data["system_load"]["avg_memory"] = statistics.mean(memory_values)
                self.analytics_data["system_load"]["max_memory"] = max(memory_values)
            
            # Update timestamp
            self.analytics_data["last_updated"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            logger.error(f"Error aggregating analytics: {e}")
    
    def _register_health_checks(self):
        """Register health checks for system components."""
        # Example health checks
        self.health_checks["system_memory"] = self._check_memory_health
        self.health_checks["system_disk"] = self._check_disk_health
        self.health_checks["prometheus_metrics"] = self._check_prometheus_health
    
    async def _check_memory_health(self) -> bool:
        """Check system memory health."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Less than 90% usage
        except:
            return False
    
    async def _check_disk_health(self) -> bool:
        """Check system disk health."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return disk.percent < 95  # Less than 95% usage
        except:
            return False
    
    async def _check_prometheus_health(self) -> bool:
        """Check Prometheus metrics health."""
        try:
            # Try to generate metrics
            generate_latest(self.registry)
            return True
        except:
            return False
    
    def record_request(self, agent: str, method: str, duration: float, success: bool):
        """Record a request in the monitoring system."""
        # Update Prometheus metrics
        self.metrics["total_requests"].labels(agent=agent, method=method).inc()
        self.metrics["request_duration"].labels(agent=agent, method=method).observe(duration)
        
        # Update analytics
        self.analytics_data["total_requests"] += 1
        
        if success:
            self.analytics_data["successful_requests"] += 1
        else:
            self.analytics_data["failed_requests"] += 1
        
        # Update average response time
        total = self.analytics_data["total_requests"]
        current_avg = self.analytics_data["average_response_time"]
        new_avg = ((current_avg * (total - 1)) + duration) / total
        self.analytics_data["average_response_time"] = new_avg
        
        # Update error rate
        self.analytics_data["error_rate"] = self.analytics_data["failed_requests"] / total
        
        # Store in performance data
        timestamp = datetime.utcnow()
        self.performance_data[f"{agent}_response_time"].append((timestamp, duration))
    
    def record_prediction(self, model: str, confidence: float, mae: float):
        """Record a prediction in the monitoring system."""
        self.metrics["predictions_made"].labels(model=model).inc()
        
        # Store in analytics
        if "predictions" not in self.analytics_data:
            self.analytics_data["predictions"] = {}
        
        self.analytics_data["predictions"][f"{model}_count"] = (
            self.analytics_data["predictions"].get(f"{model}_count", 0) + 1
        )
        
        self.analytics_data["predictions"][f"{model}_avg_confidence"] = confidence
        self.analytics_data["predictions"][f"{model}_avg_mae"] = mae
    
    def record_deal(self, retailer: str, category: str, discount: float):
        """Record a deal discovery in the monitoring system."""
        self.metrics["deals_found"].labels(retailer=retailer, category=category).inc()
        
        # Store in analytics
        if "deals" not in self.analytics_data:
            self.analytics_data["deals"] = {}
        
        self.analytics_data["deals"][f"{retailer}_count"] = (
            self.analytics_data["deals"].get(f"{retailer}_count", 0) + 1
        )
        
        self.analytics_data["deals"][f"{category}_count"] = (
            self.analytics_data["deals"].get(f"{category}_count", 0) + 1
        )
        
        self.analytics_data["deals"][f"{retailer}_avg_discount"] = discount
    
    def update_agent_metrics(self, agent: str, queue_size: int, health: bool):
        """Update agent-specific metrics."""
        self.metrics["agent_queue_size"].labels(agent=agent).set(queue_size)
        self.metrics["agent_health"].labels(agent=agent).set(1 if health else 0)
        
        # Store in analytics
        if "agent_performance" not in self.analytics_data:
            self.analytics_data["agent_performance"] = {}
        
        self.analytics_data["agent_performance"][f"{agent}_queue_size"] = queue_size
        self.analytics_data["agent_performance"][f"{agent}_health"] = health
    
    async def _create_alert(self, severity: AlertSeverity, title: str, description: str, source: str, metadata: Optional[Dict[str, Any]] = None):
        """Create an alert."""
        alert = Alert(
            alert_id=f"alert_{int(time.time())}_{len(self.alerts)}",
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            source=source,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        logger.warning(f"Alert created: {title} ({severity.value})")
    
    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            "analytics": self.analytics_data,
            "alerts": [
                {
                    "id": alert.alert_id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "timestamp": alert.timestamp.isoformat(),
                    "source": alert.source,
                    "resolved": alert.resolved
                }
                for alert in list(self.alerts)[-20:]  # Last 20 alerts
            ],
            "performance": {
                name: [
                    {"timestamp": timestamp.isoformat(), "value": value}
                    for timestamp, value in data[-10:]  # Last 10 data points
                ]
                for name, data in self.performance_data.items()
            },
            "health_checks": {},
            "prometheus_metrics": generate_latest(self.registry).decode('utf-8')
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def health_check(self) -> bool:
        """Check if the monitoring system is healthy."""
        try:
            # Check if background tasks are running
            if not self.is_running:
                return False
            
            # Check if metrics are being collected
            if self.analytics_data["total_requests"] == 0 and len(self.performance_data) == 0:
                return False
            
            # Check Prometheus registry
            try:
                generate_latest(self.registry)
            except:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring system health check failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the monitoring system."""
        try:
            logger.info("Shutting down monitoring system...")
            
            self.is_running = False
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Final metrics collection
            self.analytics_data["shutdown_time"] = datetime.utcnow().isoformat()
            
            logger.info("Monitoring system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during monitoring system shutdown: {e}")

# Alert handlers
async def console_alert_handler(alert: Alert):
    """Console alert handler."""
    print(f"🚨 ALERT [{alert.severity.value.upper()}] {alert.title}")
    print(f"   Description: {alert.description}")
    print(f"   Source: {alert.source}")
    print(f"   Time: {alert.timestamp}")

async def log_alert_handler(alert: Alert):
    """Logging alert handler."""
    if alert.severity == AlertSeverity.CRITICAL:
        logger.critical(f"Alert: {alert.title} - {alert.description}")
    elif alert.severity == AlertSeverity.ERROR:
        logger.error(f"Alert: {alert.title} - {alert.description}")
    elif alert.severity == AlertSeverity.WARNING:
        logger.warning(f"Alert: {alert.title} - {alert.description}")
    else:
        logger.info(f"Alert: {alert.title} - {alert.description}")

# Global monitoring instance
monitoring_system = MonitoringSystem()

# Initialize alert handlers
async def initialize_monitoring():
    monitoring_system.add_alert_handler(console_alert_handler)
    monitoring_system.add_alert_handler(log_alert_handler)
