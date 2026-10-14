"""Shared Model Loader Package for PromptTrace.

Provides singleton model loading, pooling, and utilities for the
two-model architecture:
- Reasoning Model: Qwen/Qwen2.5-3B-Instruct
- Annotation Model: nvidia/Nemotron-3.5-Content-Safety
"""

from .loader import (
    get_reasoning_model,
    get_annotation_model,
    load_reasoning_model,
    load_annotation_model,
    release_models,
    get_model_info,
    warmup_cuda_kernels,
)
from .pool import ModelPool, get_model_pool, ModelInvocationLog
from .gpu_budget import (
    CPU as GPU_CPU,
    FP16 as GPU_FP16,
    INT8 as GPU_INT8,
    NF4 as GPU_NF4,
    LoadPlan,
    GpuBudgetError,
    build_plan,
    cuda_free_memory,
    get_gpu_plan,
)
from .utils import (
    extract_json,
    validate_json_schema,
    validate_confidence,
    validate_category,
    validate_annotation_output,
    validate_reasoning_output,
    format_prompt,
    count_tokens,
    truncate_to_tokens,
    ensure_cuda,
    get_gpu_memory_info,
    ANNOTATION_CATEGORIES,
)

__all__ = [
    # Loader
    "get_reasoning_model",
    "get_annotation_model",
    "load_reasoning_model",
    "load_annotation_model",
    "release_models",
    "get_model_info",
    "warmup_cuda_kernels",
    # Pool
    "ModelPool",
    "get_model_pool",
    "ModelInvocationLog",
    # GPU Budget
    "GPU_FP16",
    "GPU_INT8",
    "GPU_NF4",
    "GPU_CPU",
    "LoadPlan",
    "GpuBudgetError",
    "build_plan",
    "cuda_free_memory",
    "get_gpu_plan",
    # Utils
    "extract_json",
    "validate_json_schema",
    "validate_confidence",
    "validate_category",
    "validate_annotation_output",
    "validate_reasoning_output",
    "format_prompt",
    "count_tokens",
    "truncate_to_tokens",
    "ensure_cuda",
    "get_gpu_memory_info",
    "ANNOTATION_CATEGORIES",
]