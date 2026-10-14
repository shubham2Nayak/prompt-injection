"""
Modal Phase 3 Deployment - FastAPI Backend on T4 GPU
Deploys the existing PromptTrace FastAPI backend to Modal with T4 GPU support.
"""

import modal
import os

# ──────────────────────────────────────────────────────────────────────────────
# Modal Image - Production dependencies (validated in Phase 2)
# ──────────────────────────────────────────────────────────────────────────────

# Base: Python 3.12 with CUDA 12.1 runtime via PyTorch index
phase3_image = (
    modal.Image.debian_slim(python_version="3.12")
    # Install PyTorch with CUDA 12.1 first
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install all other dependencies from standard PyPI
    .pip_install(
        # Core Framework
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.13.4",
        "pydantic-settings==2.15.0",
        # LLM & NLP
        "transformers==4.50.0",
        "huggingface-hub==0.26.2",
        "accelerate==1.0.1",
        "sentence-transformers==3.2.1",
        "bitsandbytes==0.50.2",
        "langchain==0.3.27",
        "langchain-community==0.3.27",
        "langgraph==0.5.4",
        "langsmith==0.3.45",
        # Vector Stores & Search
        "faiss-gpu-cu12==1.12.0",
        # PDF Processing
        "PyMuPDF==1.23.8",
        # Utilities
        "python-multipart==0.0.6",
        "python-dotenv==1.0.0",
        "networkx==3.2",
        # Async & Concurrency
        "httpx==0.28.1",
        "aiofiles==23.2.1",
        # Data Processing
        "numpy==1.26.4",
        "scipy==1.11.4",
        # Logging & Monitoring
        "python-json-logger==2.0.7",
        # Production Server
        "gunicorn==21.2.0",
        # Tokenizers compatibility
        "tokenizers==0.21.4",
    )
    # Environment variables for optimal GPU memory management
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Validated Phase 2 configuration
        "MODEL_DEVICE": "cuda",
        "EMBEDDING_DEVICE": "cpu",
        "ANNOTATION_DEVICE": "cuda",
        "MODEL_DTYPE": "bfloat16",
        "GPU_RESERVE_GB": "2.0",
        "GPU_LOAD_MODE_OVERRIDES": "reasoning=nf4,annotation=bf16,embedding=fp16",
        "FAISS_REQUIRE_GPU": "true",
        "FAISS_GPU_DEVICE": "0",
        "FAISS_GPU_DEVICE": "0",
        "ANNOTATION_BATCH_SIZE": "8",
        "REASONING_BATCH_SIZE": "1",
        "SIMULATION_CONCURRENCY": "1",
        # Hard caps from config.py
        "MAX_NEW_TOKENS": "512",
        "MAX_NEW_TOKENS_RESEARCH": "768",
        "MAX_NEW_TOKENS_ANALYST": "768",
        "MAX_NEW_TOKENS_ACTION": "768",
        "MAX_NEW_TOKENS_ANNOTATION": "256",
    })
    # Add the backend source code to the image
    .add_local_dir("backend", remote_path="/app/backend", copy=True)
    # Set working directory
    .workdir("/app")
)

# ──────────────────────────────────────────────────────────────────────────────
# Modal Volume for Hugging Face Cache
# ──────────────────────────────────────────────────────────────────────────────

hf_cache_volume = modal.Volume.from_name("prompttrace-hf-cache", create_if_missing=True)

# ──────────────────────────────────────────────────────────────────────────────
# Modal App
# ──────────────────────────────────────────────────────────────────────────────

app = modal.App("prompttrace-backend")

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI ASGI Application
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=phase3_image,
    gpu="T4",
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
        "/app/data": modal.Volume.from_name("prompttrace-data", create_if_missing=True),
    },
    timeout=1800,
    scaledown_window=300,
    max_containers=1,
)
@modal.asgi_app()
def fastapi_app():
    """
    ASGI application entry point for Modal.
    This wraps the existing FastAPI application from backend/app.py
    """
    import sys
    import os
    
    # Change to backend directory for proper imports
    os.chdir("/app/backend")
    sys.path.insert(0, "/app/backend")
    
    # Import the FastAPI app from the existing backend
    from app import app as fastapi_app
    
    return fastapi_app


# ──────────────────────────────────────────────────────────────────────────────
# Local Entrypoint for Testing
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def test_deployment():
    """Test the deployed API endpoint."""
    import subprocess
    import json
    
    # Get the deployed URL
    result = subprocess.run(
        ["modal", "app", "list", "--json"],
        capture_output=True,
        text=True,
    )
    print("Modal apps:")
    print(result.stdout)
    
    print("\nTo test the deployment:")
    print("1. Run: modal deploy modal_phase3.py")
    print("2. Get the URL from the deployment output")
    print("3. Test: curl https://<your-url>/health")
    print("4. Test: curl https://<your-url>/api/gpu")
    print("5. Test: curl -X POST https://<your-url>/api/upload -F 'file=@test.pdf'")

if __name__ == "__main__":
    test_deployment()