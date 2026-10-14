"""
Logs API Endpoints
Provides access to workflow logs and propagation events.

Every handler here is a coroutine that used to call straight into the
synchronous ``get_propagation_log()``, which read and JSON-parsed the entire
``propagation_events.jsonl`` file. That is a blocking filesystem walk in the
middle of the event loop, repeated on six endpoints for a single dashboard
refresh.

Reads are now served from the logger's bounded in-memory ring, and each call is
still dispatched to a dedicated worker thread so the loop is never blocked - the
in-memory path makes that cheap, and the bootstrap path (which does touch disk)
stays off the loop too.
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import config
from tools import propagation_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["logs"])

# Dedicated pool for log reads.
#
# Log reads are memory-only now, so this exists to keep the one remaining
# filesystem operation (the deferred tail bootstrap, and the unlink in
# clear_logs) off the event loop. It is separate from the default executor so a
# queue of log polls cannot starve unrelated run_in_executor work.
_executor_lock = threading.Lock()
_executor: Optional[ThreadPoolExecutor] = None


def _get_executor() -> ThreadPoolExecutor:
    """Return the shared log executor, creating it on first use."""
    global _executor

    if _executor is not None:
        return _executor

    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="prompttrace-logs",
            )
    return _executor


async def _run_in_executor(func, *args):
    """Run a blocking callable on the log executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_executor(), func, *args)


def shutdown_executor(wait: bool = False) -> None:
    """Tear down the log executor. Called from the app lifespan hook."""
    global _executor

    with _executor_lock:
        executor = _executor
        _executor = None

    if executor is not None:
        executor.shutdown(wait=wait)


class PropagationLogEntry(BaseModel):
    """Single propagation log entry."""
    timestamp: str
    agent: str
    infected: bool
    attack_detected: bool
    attack_phrase: Optional[str]
    propagation_depth: int
    mutation_detected: bool
    confidence: float


@router.get("/logs/propagation")
async def get_propagation_logs(limit: int = 100) -> Dict[str, Any]:
    """
    Get propagation event logs.

    Args:
        limit: Maximum number of logs to return

    Returns:
        List of propagation events, most recent first
    """
    try:
        # get_recent_events returns the newest `limit` from the ring and reverses
        # them, so no sort of the full history is needed.
        events = await _run_in_executor(propagation_logger.get_recent_events, limit)

        return {
            "events": events,
            "count": len(events),
        }
    except Exception as e:
        logger.error(f"Error retrieving propagation logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/propagation/summary")
async def get_propagation_summary() -> Dict[str, Any]:
    """
    Get summary of propagation events.

    Returns:
        Summary statistics
    """
    try:
        # Served from counters maintained as events are recorded.
        return await _run_in_executor(propagation_logger.get_summary)
    except Exception as e:
        logger.error(f"Error computing propagation summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/propagation/by-agent/{agent_name}")
async def get_propagation_logs_by_agent(agent_name: str) -> Dict[str, Any]:
    """
    Get propagation logs for specific agent.

    Args:
        agent_name: Name of agent to filter by

    Returns:
        Filtered propagation events
    """
    try:
        filtered = await _run_in_executor(
            propagation_logger.get_events_by_agent, agent_name
        )
        return {
            "agent": agent_name,
            "events": filtered,
            "count": len(filtered),
            "infection_count": sum(1 for e in filtered if e.get("infected")),
        }
    except Exception as e:
        logger.error(f"Error retrieving agent logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/propagation/timeline")
async def get_propagation_timeline(hours: Optional[int] = 24) -> Dict[str, Any]:
    """
    Get propagation events in chronological order.

    Args:
        hours: Accepted for API compatibility; the served buffer is bounded and
            already chronological, so no time filtering is applied

    Returns:
        Chronological list of events
    """
    try:
        events = await _run_in_executor(propagation_logger.get_timeline, hours)
        return {
            "timeline": events,
            "count": len(events),
        }
    except Exception as e:
        logger.error(f"Error retrieving timeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/propagation/threats")
async def get_detected_threats() -> Dict[str, Any]:
    """
    Get all detected attack phrases.

    Returns:
        List of unique attack phrases detected
    """
    try:
        return await _run_in_executor(propagation_logger.get_threats)
    except Exception as e:
        logger.error(f"Error retrieving threats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/logs/clear")
async def clear_logs() -> Dict[str, str]:
    """
    Clear all logs.

    Returns:
        Clear status
    """
    try:
        # Touches the filesystem (unlink plus writer shutdown).
        await _run_in_executor(propagation_logger.clear_logs)
        logger.info("All logs cleared")
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Error clearing logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs/status")
async def get_logs_status() -> Dict[str, Any]:
    """
    Get logs status information.

    Returns:
        Status information
    """
    try:
        # No longer reads the whole file just to report a length.
        return await _run_in_executor(propagation_logger.get_status)
    except Exception as e:
        logger.error(f"Error getting logs status: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
        }