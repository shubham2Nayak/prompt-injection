"""
Modal Infrastructure for PromptTrace - Phase 1
Minimal T4 GPU validation infrastructure.
Does NOT load project models or application code.
"""

import datetime
import modal

# ──────────────────────────────────────────────────────────────────────────────
# Modal App
# ──────────────────────────────────────────────────────────────────────────────

app = modal.App("prompttrace-phase1-gpu-validation")

# ──────────────────────────────────────────────────────────────────────────────
# Modal Image - Phase 1: GPU validation only
# ──────────────────────────────────────────────────────────────────────────────

# Base: use Modal's pre-built CUDA image with Python 3.12
# This provides NVIDIA drivers, CUDA runtime, cuDNN, cuBLAS
# without needing to install the full CUDA toolkit (nvcc).
# The project's torch==2.5.1 expects CUDA 12.x.
# modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04") would include nvcc.
# We only need CUDA runtime, so use the slimmer runtime base or Modal's built-in.
# Modal's `debian_slim` with `apt_install` of cuda-toolkit is heavier than needed.
# Best: use Modal's pre-built CUDA image which has drivers + runtime.

# For Phase 1, we only need:
# - Python 3.12
# - CUDA runtime (drivers, libcudart, cuDNN, cuBLAS)
# - PyTorch 2.5.1 with CUDA 12 support
# - Core dependencies for GPU validation (torch, numpy)
# NOT needed in Phase 1: transformers, bitsandbytes, faiss, sentence-transformers, etc.

# Modal's `cuda` image tag provides the GPU-enabled base.
# We'll build on `modal.Image.debian_slim(python_version="3.12")` and add CUDA via
# Modal's `.apt_install` of nvidia-cuda-toolkit is NOT needed - Modal GPU functions
# already have the driver + runtime in the container environment.
# Instead, install PyTorch with CUDA index.

# CRITICAL: The project requires torch==2.5.1 with CUDA 12.x.
# torch 2.5.1's default PyPI wheel is CUDA 12.1 (cu121).
# We must install from the correct index to get CUDA-enabled wheels.

# Phase 1 Image: minimal GPU validation environment
phase1_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "modal==1.6.1",
    )
    .env({
        # Help PyTorch memory allocator on T4
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# ──────────────────────────────────────────────────────────────────────────────
# Future Image (for Phase 2+ reference - NOT used in Phase 1)
# ──────────────────────────────────────────────────────────────────────────────

# This is documented here for Phase 2 planning. It includes all project dependencies
# but is NOT built or used in Phase 1.
future_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Core Framework
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.13.4",
        "pydantic-settings==2.15.0",
        # LLM & NLP
        "transformers==4.50.0",
        "huggingface-hub==0.26.2",
        "torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121",
        "accelerate==1.0.1",
        "sentence-transformers==3.2.1",
        "bitsandbytes>=0.45.0",
        "langchain==0.3.27",
        "langchain-community==0.3.27",
        "langgraph==0.5.4",
        "langsmith==0.3.45",
        # Vector Stores & Search
        # faiss-gpu-cu12 requires CUDA 12.x runtime (provided by Modal T4 containers)
        "faiss-gpu-cu12>=1.8.0",
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
    )
    .env({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
)

# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 Diagnostic Function
# ──────────────────────────────────────────────────────────────────────────────

@app.function(
    image=phase1_image,
    gpu="T4",
    timeout=300,  # 5 minutes max for cold start + validation
)
def validate_t4_gpu() -> dict:
    """
    Validate that Modal can provide a working T4 GPU with CUDA and PyTorch.
    This is a pure infrastructure test - NO project models are loaded.
    """
    import subprocess
    import torch
    import numpy as np

    results = {
        "phase": "Phase 1 - Modal T4 GPU Validation",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # 1. nvidia-smi
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        results["nvidia_smi"] = {
            "exit_code": nvidia_smi.returncode,
            "stdout": nvidia_smi.stdout[:2000] if nvidia_smi.stdout else "",
            "stderr": nvidia_smi.stderr[:500] if nvidia_smi.stderr else "",
        }
    except Exception as e:
        results["nvidia_smi"] = {"error": str(e)}

    # 2. CUDA visibility from Python (basic)
    results["torch"] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }

    # 3. torch.cuda.is_available()
    cuda_available = torch.cuda.is_available()
    results["cuda_available"] = cuda_available

    if not cuda_available:
        results["error"] = "torch.cuda.is_available() returned False"
        return results

    # 4. GPU name and properties
    try:
        device_count = torch.cuda.device_count()
        results["device_count"] = device_count
        if device_count > 0:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            results["gpu"] = {
                "name": gpu_name,
                "major": props.major,
                "minor": props.minor,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "multi_processor_count": props.multi_processor_count,
            }
    except Exception as e:
        results["gpu_properties_error"] = str(e)

    # 5. CUDA tensor operation (proves GPU is actually executing)
    try:
        # Create tensors on CPU
        a_cpu = torch.randn(1024, 1024, dtype=torch.float32)
        b_cpu = torch.randn(1024, 1024, dtype=torch.float32)

        # Move to GPU
        a_gpu = a_cpu.to("cuda", non_blocking=True)
        b_gpu = b_cpu.to("cuda", non_blocking=True)

        # Perform matrix multiplication on GPU
        # This is a significant enough computation to verify real GPU execution
        c_gpu = torch.matmul(a_gpu, b_gpu)

        # Synchronize to ensure completion
        torch.cuda.synchronize()

        # Move result back to CPU and verify
        c_cpu = c_gpu.to("cpu")
        expected_shape = (1024, 1024)

        results["cuda_tensor_test"] = {
            "status": "PASS",
            "input_shape": list(a_cpu.shape),
            "output_shape": list(c_cpu.shape),
            "expected_shape": list(expected_shape),
            "shape_match": list(c_cpu.shape) == list(expected_shape),
            "sample_values": c_cpu[0, :5].tolist(),
        }

        # Also check memory allocation
        results["memory"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        }

    except Exception as e:
        results["cuda_tensor_test"] = {
            "status": "FAIL",
            "error": str(e),
        }

    # 6. Verify T4 specifically
    if "gpu" in results:
        gpu_name = results["gpu"].get("name", "").upper()
        results["is_t4"] = "T4" in gpu_name or "TESLA T4" in gpu_name
    else:
        results["is_t4"] = False

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Local Entrypoint for `modal run`
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """Run the GPU validation and print results."""
    import json
    result = validate_t4_gpu.remote()
    print(json.dumps(result, indent=2))