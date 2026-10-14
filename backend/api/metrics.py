"""
Metrics API Endpoints
Provides propagation metrics and analytics.
"""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from metrics import MetricsEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["metrics"])

# Store metrics instance (injected from main app)
metrics_engine: Optional[MetricsEngine] = None


class MetricsResponse(BaseModel):
    """Metrics response."""
    attack_success_rate: float
    average_propagation_depth: float
    agent_infection_rate: Dict[str, float]
    mutation_rate: float
    tool_trigger_rate: float
    average_latency_ms: float
    total_runs: int
    successful_detections: int


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics() -> Dict[str, Any]:
    """
    Get current propagation metrics.
    
    Returns:
        PropagationMetrics object as dictionary
    """
    if not metrics_engine:
        raise HTTPException(status_code=500, detail="Metrics engine not initialized")
    
    try:
        metrics_dict = metrics_engine.get_metrics_dict()
        return {
            "status": "success",
            "metrics": metrics_dict,
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get summary statistics for display.
    
    Returns:
        Summary statistics
    """
    if not metrics_engine:
        raise HTTPException(status_code=500, detail="Metrics engine not initialized")
    
    try:
        summary = metrics_engine.get_summary_stats()
        return {
            "status": "success",
            "summary": summary,
        }
    except Exception as e:
        logger.error(f"Error computing summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/attack-success-rate")
async def get_attack_success_rate() -> Dict[str, Any]:
    """
    Get attack success rate.
    
    Returns:
        Attack success rate
    """
    if not metrics_engine:
        return {"rate": 0.0}
    
    return {
        "rate": metrics_engine.compute_attack_success_rate(),
        "total_runs": metrics_engine.total_runs,
    }


@router.get("/metrics/propagation-depth")
async def get_average_propagation_depth() -> Dict[str, Any]:
    """
    Get average propagation depth.
    
    Returns:
        Average propagation depth
    """
    if not metrics_engine:
        return {"depth": 0.0}
    
    return {
        "average_depth": metrics_engine.compute_average_propagation_depth(),
        "total_runs": metrics_engine.total_runs,
    }


@router.get("/metrics/agent-infection-rates")
async def get_agent_infection_rates() -> Dict[str, Any]:
    """
    Get infection rate per agent.
    
    Returns:
        Agent infection rates
    """
    if not metrics_engine:
        return {"rates": {}}
    
    return {
        "infection_rates": metrics_engine.compute_agent_infection_rates(),
        "total_runs": metrics_engine.total_runs,
    }


@router.get("/metrics/mutation-rate")
async def get_mutation_rate() -> Dict[str, Any]:
    """
    Get mutation rate.
    
    Returns:
        Mutation rate
    """
    if not metrics_engine:
        return {"rate": 0.0}
    
    return {
        "mutation_rate": metrics_engine.compute_mutation_rate(),
        "total_runs": metrics_engine.total_runs,
    }


@router.get("/metrics/tool-trigger-rate")
async def get_tool_trigger_rate() -> Dict[str, Any]:
    """
    Get tool trigger rate.
    
    Returns:
        Tool trigger rate
    """
    if not metrics_engine:
        return {"rate": 0.0}
    
    return {
        "tool_trigger_rate": metrics_engine.compute_tool_trigger_rate(),
        "total_runs": metrics_engine.total_runs,
    }


@router.post("/metrics/reset")
async def reset_metrics() -> Dict[str, str]:
    """
    Reset metrics engine.
    
    Returns:
        Reset status
    """
    if not metrics_engine:
        raise HTTPException(status_code=500, detail="Metrics engine not initialized")
    
    metrics_engine.reset()
    logger.info("Metrics engine reset")
    return {"status": "reset"}


def set_metrics_engine(metrics):
    """Set the metrics engine instance."""
    global metrics_engine
    metrics_engine = metrics
