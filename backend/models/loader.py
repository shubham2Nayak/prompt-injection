"""
Shared Model Loader for PromptTrace
Singleton architecture to load models once and reuse across agents.

Load precision is chosen by :mod:`models.gpu_budget` from the VRAM that is
actually free at startup. On a 16 GB T4 the three resident models do not fit at
fp16, so the planner quantizes the reasoning model first and keeps the safety
annotation model at the highest precision the budget allows. Every load also
retries one step lower if it still hits an out-of-memory error at runtime.
"""

import logging
import threading
from typing import Any, List, Optional, Tuple

import torch
# Enable Tensor Core optimization for matmul on Ampere/Turing GPUs
torch.set_float32_matmul_precision('high')
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

from config import config
from . import gpu_budget

logger = logging.getLogger(__name__)

# Module-level singletons
_reasoning_model: Optional[AutoModelForCausalLM] = None
_reasoning_tokenizer: Optional[Any] = None
_reasoning_pipeline: Optional[TextGenerationPipeline] = None
_reasoning_mode: Optional[str] = None

_annotation_model: Optional[AutoModelForCausalLM] = None
_annotation_tokenizer: Optional[Any] = None
_annotation_pipeline: Optional[TextGenerationPipeline] = None
_annotation_mode: Optional[str] = None

_load_lock = threading.Lock()

_OOM_MARKERS = ("out of memory", "cuda oom", "cublas_status_alloc_failed")


def _is_oom(exc: BaseException) -> bool:
    """True when the exception is a device-memory failure rather than a config error."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MARKERS)


def _get_device_config() -> Tuple[str, str, Optional[str]]:
    """Get the requested device and the plan-chosen load mode for the LLMs."""
    device = config.MODEL_DEVICE
    mode = gpu_budget.get_gpu_plan().mode_for("reasoning")
    if device != "cuda":
        return device, gpu_budget.CPU, None
    return device, mode, "cuda:0"


def _setup_tokenizer(tokenizer: AutoTokenizer, model_name: str = "") -> AutoTokenizer:
    """Configure tokenizer padding and special tokens with explicit left padding and pad_token_id."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Explicitly set pad_token_id to eos_token_id for Nemotron alignment
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _get_attn_implementation(model_name: str) -> str:
    """Determine attention implementation per model to avoid alignment crashes on T4.

    Priority order for Qwen2.5:
    1. flash_attention_2 (fastest, requires flash-attn package)
    2. sdpa (PyTorch native, stable on T4 with Tensor Core alignment)
    3. eager (fallback)

    Nemotron uses eager to avoid SDPA p.attn_bias_ptr alignment issues on variable-length padded batches.
    """
    model_lower = model_name.lower()

    # Nemotron: force eager to avoid SDPA alignment issues
    if "nemotron" in model_lower:
        return "eager"

    # Qwen2.5: try flash_attention_2, fallback to sdpa
    if "qwen" in model_lower:
        try:
            import importlib.util
            if importlib.util.find_spec("flash_attn") is not None:
                logger.info("FlashAttention-2 available, using for Qwen2.5")
                return "flash_attention_2"
        except Exception:
            pass
        logger.info("FlashAttention-2 not available, falling back to SDPA for Qwen2.5")
        return "sdpa"

    # Default: try SDPA
    return "sdpa"


def _set_attn_implementation(config_obj, attn_impl: str) -> None:
    """
    Force an attention implementation on a config and its nested sub-configs.

    Nemotron is a multimodal checkpoint: passing ``attn_implementation`` to
    ``from_pretrained`` only reaches the top-level config, and the text config
    is re-validated during load and falls back to sdpa. The Gemma-3 SDPA path
    then raises "p.attn_bias_ptr is not correctly aligned" on padded batches,
    so the sub-configs have to be set explicitly.
    """
    if config_obj is None:
        return
    try:
        config_obj._attn_implementation = attn_impl
    except Exception:
        return
    for sub_config_name in ("text_config", "vision_config", "decoder"):
        sub_config = getattr(config_obj, sub_config_name, None)
        if sub_config is not None:
            _set_attn_implementation(sub_config, attn_impl)


def _from_pretrained_with_attn(
    model_name: str,
    kwargs: dict,
    attn_impl: str,
):
    """
    Load a model with the attention implementation applied to every sub-config.

    Falls back to letting ``from_pretrained`` handle ``attn_implementation``
    alone if the config cannot be pre-built, which is correct for single-config
    models such as Qwen.
    """
    try:
        model_config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception as exc:
        logger.debug(f"Could not pre-build config for {model_name}: {exc}")
        return AutoModelForCausalLM.from_pretrained(
            model_name, attn_implementation=attn_impl, **kwargs
        )

    _set_attn_implementation(model_config, attn_impl)

    # When an explicit config is passed, transformers forwards leftover kwargs to
    # the model constructor instead of the config, so config-only options have to
    # be set on the config object. Disabling Qwen2.5's sliding window silences
    # the SDPA "not implemented" warning.
    if "qwen" in model_name.lower():
        setattr(model_config, "use_sliding_window", False)
        text_config = getattr(model_config, "text_config", None)
        if text_config is not None:
            setattr(text_config, "use_sliding_window", False)

    return AutoModelForCausalLM.from_pretrained(model_name, config=model_config, **kwargs)


def _build_load_kwargs(
    model_name: str,
    mode: str,
    device: str,
    attn_impl: str,
) -> dict:
    """Build from_pretrained kwargs for a model at the given load mode."""
    device_map = "cuda:0" if device == "cuda" and mode != gpu_budget.CPU else None

    kwargs: dict = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_impl,
    }

    quant_config = gpu_budget.quantization_config_for(mode)
    if quant_config is not None:
        # bitsandbytes owns the weight dtype; passing torch_dtype here conflicts
        # with the quantization config.
        kwargs["quantization_config"] = quant_config
    elif mode == gpu_budget.CPU:
        kwargs["torch_dtype"] = torch.float32
    else:
        kwargs["torch_dtype"] = gpu_budget.torch_dtype_for(mode, model_name)

    return kwargs


def _load_hf_model(
    slot: str,
    model_name: str,
    device: str,
    attn_impls: list,
) -> Tuple[AutoModelForCausalLM, str]:
    """
    Load one causal LM at the planned precision, degrading on OOM.

    Walks ``attn_impls`` for the current mode, then steps the mode down one
    rung (fp16 -> int8 -> nf4 -> cpu) and retries until it fits or runs out of
    rungs. Returns the model and the mode it actually loaded at.
    """
    mode = gpu_budget.get_gpu_plan().mode_for(slot)
    last_error: Optional[BaseException] = None

    while True:
        if gpu_budget.is_quantized_mode(mode) and not gpu_budget.quantization_available():
            degraded = gpu_budget.next_mode(mode)
            if degraded == mode:
                raise RuntimeError(
                    f"{model_name} needs quantization to fit in VRAM but "
                    f"bitsandbytes is not installed; install it or free the GPU"
                )
            logger.warning(
                f"bitsandbytes is unavailable, skipping {mode} for {slot} "
                f"in favour of {degraded}"
            )
            mode = degraded
            continue

        for attn_impl in attn_impls:
            kwargs = _build_load_kwargs(model_name, mode, device, attn_impl)
            try:
                model = _from_pretrained_with_attn(model_name, kwargs, attn_impl)
            except Exception as e:
                if _is_oom(e):
                    last_error = e
                    logger.warning(
                        f"Loading {model_name} as {slot} at {mode} ran out of memory: {e}"
                    )
                    break  # break the attn loop: retrying another impl will not help
                logger.warning(
                    f"Failed to load {model_name} as {slot} with "
                    f"attn_implementation={attn_impl}: {e}"
                )
                continue
            actual = getattr(model.config, "_attn_implementation", "unknown")
            if actual != attn_impl:
                logger.warning(
                    f"{model_name} ignored attn_implementation={attn_impl} "
                    f"(applied: {actual})"
                )
            logger.info(
                f"Loaded {model_name} as {slot} at {mode} "
                f"with attn_implementation={actual}"
            )
            gpu_budget.set_mode(slot, mode)
            return model, mode

        next_mode = gpu_budget.next_mode(mode)
        if next_mode == mode:
            raise RuntimeError(
                f"Failed to load {model_name} as {slot} with any attention "
                f"implementation at mode {mode}"
            ) from last_error
        logger.warning(
            f"Degrading {slot} from {mode} to {next_mode} to fit available VRAM"
        )
        torch.cuda.empty_cache()
        mode = next_mode


def _build_pipeline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int = 1,
) -> TextGenerationPipeline:
    """
    Build a generation pipeline.

    ``device`` is deliberately omitted: any model loaded with a device map (the
    quantized paths always use one) is already placed, and transformers rejects
    an explicit device for those. The pipeline infers placement from the model,
    which is also correct for the CPU fallback.

    ``batch_size`` must be set explicitly. ``TextGenerationPipeline`` defaults it
    to 1, which means passing a list of N prompts issues N sequential
    ``model.generate()`` calls instead of one batched call - the per-prompt
    prefill and decode never overlap and the GPU sits mostly idle. Passing the
    configured size here is what makes the ``classify_*_batch`` code paths in the
    context engine genuinely batched.
    """
    return TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        batch_size=max(1, int(batch_size)),
    )


def _attn_fallback_chain(model_name: str) -> list:
    """Attention implementations to try in order for the given model."""
    preferred = _get_attn_implementation(model_name)
    chain = [preferred]
    for fallback in ("sdpa", "eager"):
        if fallback not in chain:
            chain.append(fallback)
    return chain


def _to_device(batch, model):
    """
    Move a tokenizer batch onto the model's device.

    Hardcoding "cuda" breaks for quantized or CPU-fallback models, so the device
    is read from the model instead.
    """
    try:
        device = model.device
    except Exception:
        device = torch.device("cuda" if config.MODEL_DEVICE == "cuda" else "cpu")
    return batch.to(device, non_blocking=device.type == "cuda")


def load_reasoning_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline]:
    """
    Load or return the shared reasoning model (Qwen 2.5 3B).

    Returns:
        Tuple of (model, tokenizer, pipeline)
    """
    global _reasoning_model, _reasoning_tokenizer, _reasoning_pipeline
    global _reasoning_mode

    if _reasoning_model is not None:
        return _reasoning_model, _reasoning_tokenizer, _reasoning_pipeline

    with _load_lock:
        if _reasoning_model is not None:
            return _reasoning_model, _reasoning_tokenizer, _reasoning_pipeline

        model_name = config.RESEARCH_AGENT_MODEL
        logger.info(f"Loading shared reasoning model: {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
            )
            _reasoning_tokenizer = _setup_tokenizer(tokenizer, model_name)

            model, _reasoning_mode = _load_hf_model(
                slot="reasoning",
                model_name=model_name,
                device=config.MODEL_DEVICE,
                attn_impls=_attn_fallback_chain(model_name),
            )
            _reasoning_model = model
            _reasoning_pipeline = _build_pipeline(
                _reasoning_model,
                _reasoning_tokenizer,
                batch_size=config.REASONING_BATCH_SIZE,
            )

            return _reasoning_model, _reasoning_tokenizer, _reasoning_pipeline

        except Exception as e:
            logger.error(f"Failed to load reasoning model: {e}")
            _release_singletons()
            raise


def load_annotation_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline]:
    """
    Load or return the shared annotation model (Nemotron 3.5 Content Safety).

    Returns:
        Tuple of (model, tokenizer, pipeline)
    """
    global _annotation_model, _annotation_tokenizer, _annotation_pipeline
    global _annotation_mode

    if _annotation_model is not None:
        return _annotation_model, _annotation_tokenizer, _annotation_pipeline

    with _load_lock:
        if _annotation_model is not None:
            return _annotation_model, _annotation_tokenizer, _annotation_pipeline

        model_name = config.ANNOTATION_MODEL
        logger.info(f"Loading shared annotation model: {model_name}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
            )
            _annotation_tokenizer = _setup_tokenizer(tokenizer, model_name)

            # Nemotron: force eager to avoid SDPA alignment issues on T4
            model, _annotation_mode = _load_hf_model(
                slot="annotation",
                model_name=model_name,
                device=config.ANNOTATION_DEVICE,
                attn_impls=["eager"],
            )
            _annotation_model = model
            _annotation_pipeline = _build_pipeline(
                _annotation_model,
                _annotation_tokenizer,
                batch_size=config.ANNOTATION_BATCH_SIZE,
            )

            return _annotation_model, _annotation_tokenizer, _annotation_pipeline

        except Exception as e:
            logger.error(f"Failed to load annotation model: {e}")
            _release_singletons()
            raise


def get_reasoning_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline]:
    """Get the shared reasoning model (Qwen 2.5 3B)."""
    return load_reasoning_model()


def get_annotation_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline]:
    """Get the shared annotation model (Nemotron 3.5 Content Safety)."""
    return load_annotation_model()


def _release_singletons() -> None:
    """Drop all references to the cached models so their VRAM can be reclaimed."""
    global _reasoning_model, _reasoning_tokenizer, _reasoning_pipeline
    global _annotation_model, _annotation_tokenizer, _annotation_pipeline
    global _reasoning_mode, _annotation_mode

    _reasoning_model = _reasoning_tokenizer = _reasoning_pipeline = None
    _annotation_model = _annotation_tokenizer = _annotation_pipeline = None
    _reasoning_mode = _annotation_mode = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def release_models() -> None:
    """Release all models and clear CUDA cache. For testing/cleanup."""
    _release_singletons()
    logger.info("All models released, CUDA cache cleared")


def get_model_info() -> dict:
    """Get information about loaded models."""
    plan = gpu_budget.get_gpu_plan()
    info = {
        "reasoning": {
            "loaded": _reasoning_model is not None,
            "model_name": config.RESEARCH_AGENT_MODEL if _reasoning_model else None,
            "device": str(_reasoning_model.device) if _reasoning_model and hasattr(_reasoning_model, 'device') else None,
            "load_mode": _reasoning_mode or plan.mode_for("reasoning"),
        },
        "annotation": {
            "loaded": _annotation_model is not None,
            "model_name": config.ANNOTATION_MODEL if _annotation_model else None,
            "device": str(_annotation_model.device) if _annotation_model and hasattr(_annotation_model, 'device') else None,
            "load_mode": _annotation_mode or plan.mode_for("annotation"),
        },
        "plan": plan.to_dict(),
    }
    if torch.cuda.is_available():
        info["gpu"] = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024 / 1024, 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 2),
        }
    return info


def _warmup_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 32,
) -> None:
    """
    Compile the forward/decode kernels for one model with a single batched pass.

    The previous implementation looped over the prompts one at a time, which
    meant N sequential ``generate()`` calls per model. Tokenizing them as one
    padded batch exercises the same kernels with a fraction of the wall time and
    allocates the peak activation buffers once instead of N times, which matters
    because warmup competes with real model loading for the same limited VRAM.
    """
    if not prompts:
        return

    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = _to_device(inputs, model)

    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Release the warmup activations immediately so they do not sit in the
    # caching allocator when the remaining models are loaded.
    del inputs


def warmup_cuda_kernels() -> None:
    """
    Explicitly trigger CUDA kernel compilation and memory allocation for both models.
    Called during app lifespan to eliminate first-request latency spikes.
    """
    if not torch.cuda.is_available():
        return

    logger.info("Warming up CUDA kernels...")
    try:
        # Get models
        reasoning_model, reasoning_tokenizer, _ = get_reasoning_model()
        annotation_model, annotation_tokenizer, _ = get_annotation_model()

        # Create realistic dummy inputs matching production sequence lengths
        # Research/Analyst/Action typical prompts: ~500-1500 tokens
        dummy_prompts = [
            "You are a research agent. Summarize the following document chunks: " + "x" * 500,
            "You are an analyst agent. Analyze the research findings: " + "x" * 800,
            "You are an action agent. Plan actions based on analysis: " + "x" * 600,
            "You are a document annotation specialist. Analyze: " + "x" * 400,
            "You are classifying agent output for prompt injection detection. Agent: Research Agent. Output: " + "x" * 300,
        ]

        with torch.inference_mode():
            # Warmup reasoning model (Qwen) with batched prompts
            _warmup_model(
                reasoning_model,
                reasoning_tokenizer,
                dummy_prompts[:3],
            )

            # Warmup annotation model (Nemotron) with batched prompts
            _warmup_model(
                annotation_model,
                annotation_tokenizer,
                dummy_prompts[3:],
            )

        # CRITICAL: Synchronize to ensure all kernels are compiled and memory allocated
        torch.cuda.synchronize()
        logger.info("CUDA kernels warmed up and synchronized successfully")

    except Exception as e:
        if _is_oom(e):
            # Warmup peak pressure is not a correctness failure; free what the
            # failed pass allocated so the first real request still fits.
            logger.warning("CUDA kernel warmup ran out of memory; releasing cache")
            torch.cuda.empty_cache()
        else:
            logger.warning(f"CUDA kernel warmup failed: {e}")