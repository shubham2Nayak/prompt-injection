"""
PromptTrace Backend - Main FastAPI Application
Orchestrates the complete propagation analysis system.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import config
from graph import PromptTraceWorkflow
from jsonl_writer import close_jsonl_writers
from metrics import MetricsEngine
from tools import propagation_logger
from models import get_gpu_plan, get_model_info

# Import API routers
from api import upload, simulate, metrics as metrics_api, logs, graph as graph_api

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
)

# Global instances
workflow: PromptTraceWorkflow = None
metrics_engine: MetricsEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    logger.info("=" * 80)
    logger.info("PromptTrace Backend Starting")
    logger.info("=" * 80)
    logger.info(f"Environment: {config.ENV}")
    logger.info(f"Log Level: {config.LOG_LEVEL}")
    
    try:
        global workflow, metrics_engine

        # Report the VRAM plan before loading anything, so an over-subscribed
        # card is diagnosed here rather than as a CUDA OOM further down.
        plan = get_gpu_plan()
        logger.info(f"GPU load plan: {plan.describe()}")

        # Initialize workflow
        logger.info("Initializing PromptTrace workflow...")
        workflow = PromptTraceWorkflow()
        logger.info("Workflow initialized successfully")

        # Note: CUDA warmup already ran inside ModelPool.__init__, which is
        # reached while the workflow builds its context engine. Calling
        # warmup_cuda_kernels() again here issued a second full set of
        # generations and competed with model loading for the same VRAM, so the
        # explicit second pass was removed.

        # Initialize metrics engine
        logger.info("Initializing metrics engine...")
        metrics_engine = MetricsEngine()
        logger.info("Metrics engine initialized successfully")
        
        # Inject instances into API modules
        upload.set_workflow(workflow)
        simulate.set_workflow(workflow)
        simulate.set_metrics_engine(metrics_engine)
        metrics_api.set_metrics_engine(metrics_engine)
        graph_api.set_metrics_engine(metrics_engine)
        
        logger.info("All components initialized")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Failed to initialize PromptTrace: {str(e)}")
        raise
    
    yield

    # Shutdown
    logger.info("=" * 80)
    logger.info("PromptTrace Backend Shutting Down")
    logger.info("=" * 80)

    # Drain the background JSONL writers and stop the worker pools so no
    # telemetry or in-flight work is lost on the way out.
    close_jsonl_writers()
    simulate.shutdown_executor(wait=False)
    logs.shutdown_executor(wait=False)


# Create FastAPI app
app = FastAPI(
    title="PromptTrace Backend",
    description="Cross-Context Prompt Injection Propagation Analyzer",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(upload.router)
app.include_router(simulate.router)
app.include_router(metrics_api.router)
app.include_router(logs.router)
app.include_router(graph_api.router)


# Health check endpoints
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": config.ENV,
    }


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "PromptTrace Backend",
        "version": "1.0.0",
        "description": "Cross-Context Prompt Injection Propagation Analyzer",
        "endpoints": {
            "health": "/health",
            "upload": {
                "POST": "/api/upload",
                "GET": "/api/upload/list",
                "DELETE": "/api/upload/{filename}",
            },
            "simulate": {
                "POST": "/api/simulate",
                "GET": "/api/simulate/history",
                "GET": "/api/simulate/{simulation_id}",
            },
            "metrics": {
                "GET": "/api/metrics",
                "GET": "/api/metrics/summary",
                "GET": "/api/metrics/attack-success-rate",
                "GET": "/api/metrics/propagation-depth",
                "GET": "/api/metrics/agent-infection-rates",
                "GET": "/api/metrics/mutation-rate",
                "GET": "/api/metrics/tool-trigger-rate",
                "POST": "/api/metrics/reset",
            },
            "logs": {
                "GET": "/api/logs/propagation",
                "GET": "/api/logs/propagation/summary",
                "GET": "/api/logs/propagation/by-agent/{agent_name}",
                "GET": "/api/logs/propagation/timeline",
                "GET": "/api/logs/propagation/threats",
                "DELETE": "/api/logs/clear",
                "GET": "/api/logs/status",
            },
        },
    }


@app.get("/api/config")
async def get_config() -> dict:
    """Get current configuration (non-sensitive)."""
    return {
        "environment": config.ENV,
        "log_level": config.LOG_LEVEL,
        "models": {
            "research_agent": config.RESEARCH_AGENT_MODEL,
            "analyst_agent": config.ANALYST_AGENT_MODEL,
            "action_agent": config.ACTION_AGENT_MODEL,
            "propagation_monitor": config.PROPAGATION_MONITOR_MODEL,
            "embedding": config.EMBEDDING_MODEL,
            "annotation": config.ANNOTATION_MODEL,
        },
        "parameters": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "temperatures": {
                "research": config.TEMPERATURE_RESEARCH,
                "analyst": config.TEMPERATURE_ANALYST,
                "action": config.TEMPERATURE_ACTION,
                "annotation": config.TEMPERATURE_ANNOTATION,
            },
            "top_p": config.TOP_P,
            "top_k": config.TOP_K,
            "max_new_tokens": config.MAX_NEW_TOKENS,
        },
        "limits": {
            "max_pdf_size_mb": config.MAX_PDF_SIZE_MB,
            "max_chunks_per_query": config.MAX_CHUNKS_PER_QUERY,
        },
    }


@app.get("/api/gpu")
async def get_gpu_status() -> dict:
    """
    Report the VRAM budget plan alongside live model residency.

    Useful when a model had to be degraded at load time: the planned mode and
    the mode actually in use are both reported so a silent downgrade is visible.
    """
    return {
        "plan": get_gpu_plan().to_dict(),
        "models": get_model_info(),
    }


@app.get("/api/workflow/status")
async def get_workflow_status() -> dict:
    """Get workflow status."""
    if not workflow:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "components": {
            "pdf_parser": workflow.pdf_parser is not None,
            "text_chunker": workflow.text_chunker is not None,
            "embedding_generator": workflow.embedding_generator is not None,
            "vector_store": workflow.vector_store is not None,
            "research_agent": workflow.research_agent is not None,
            "analyst_agent": workflow.analyst_agent is not None,
            "action_agent": workflow.action_agent is not None,
            "propagation_monitor": workflow.propagation_monitor is not None,
            "langgraph": workflow.graph is not None,
        },
        "vector_store_stats": (
            workflow.vector_store.get_stats() if workflow.vector_store else None
        ),
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting Uvicorn server on {config.API_HOST}:{config.API_PORT}")
    uvicorn.run(
        "app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower(),
    )
