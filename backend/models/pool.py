"""
Model Pool for PromptTrace
Manages model lifecycle, batching, and resource coordination.
Uses CUDA kernel warmup for zero first-request latency.

Invocation telemetry is written by a background thread (see jsonl_writer) rather
than by opening, writing and closing a file inside the shared mutex. Token
counting is opt-in: it duplicates tokenization the generation pipeline already
performs, and doing it unconditionally made the instrumentation cost more than
the inference it was measuring.
"""

import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Generator, Optional, Any, List, Dict, Deque
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch

from config import config
from jsonl_writer import enqueue_jsonl, jsonl_writer_stats
from .loader import (
    get_reasoning_model,
    get_annotation_model,
    release_models,
    get_model_info,
    warmup_cuda_kernels,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInvocationLog:
    """Log entry for a single model invocation."""
    model_name: str
    agent_name: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    gpu_device: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class ModelPool:
    """
    Centralized model pool managing both reasoning and annotation models.

    Features:
    - Thread-safe model access
    - Invocation logging (JSONL, written off the calling thread)
    - GPU memory monitoring
    - Batch inference coordination
    - CUDA kernel warmup on initialization
    """

    _instance: Optional["ModelPool"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # Bounded in-memory ring of recent invocations. The previous list grew
        # without limit for the life of the process, and every get_logs()/
        # get_stats() call then walked all of it. Entries are still appended
        # under the lock; the bound is enforced by the deque itself.
        self._invocation_logs: Deque[ModelInvocationLog] = deque(
            maxlen=max(1, int(config.INVOCATION_LOG_MAX_ENTRIES))
        )
        self._log_lock = threading.Lock()

        # Running aggregates, updated in the same critical section as the
        # append. get_stats() is then O(1) instead of building two filtered lists
        # and running four full passes over the history on every call.
        # "total" is tracked separately because the in-memory buffer is bounded
        # and can no longer be used as the lifetime counter.
        self._stats = {
            "total": 0,
            "reasoning_count": 0,
            "annotation_count": 0,
            "reasoning_latency_ms": 0.0,
            "annotation_latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # Cache the CUDA availability probe. torch.cuda.is_available() walks the
        # driver on first call and is stable for the process lifetime; this runs
        # once per invocation otherwise.
        self._gpu_device: str = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure log directory exists
        log_dir = os.path.join(config.LOGS_DIR, "model_invocations")
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "invocations.jsonl")

        # Trigger CUDA kernel warmup immediately on pool creation. app.py used to
        # call warmup_cuda_kernels() a second time in its lifespan hook, which
        # doubled startup cost and competed with model loading for VRAM.
        warmup_cuda_kernels()

        logger.info("ModelPool initialized with CUDA warmup complete")

    @property
    def reasoning_pipeline(self):
        """Get the reasoning model pipeline."""
        _, _, pipeline = get_reasoning_model()
        return pipeline

    @property
    def annotation_pipeline(self):
        """Get the annotation model pipeline."""
        _, _, pipeline = get_annotation_model()
        return pipeline

    @property
    def reasoning_tokenizer(self):
        """Get the reasoning model tokenizer."""
        _, tokenizer, _ = get_reasoning_model()
        return tokenizer

    @property
    def annotation_tokenizer(self):
        """Get the annotation model tokenizer."""
        _, tokenizer, _ = get_annotation_model()
        return tokenizer

    def log_invocation(self, log: ModelInvocationLog) -> None:
        """
        Record an invocation in memory and hand it to the background writer.

        The memory append and the aggregate update are cheap and stay under the
        lock. The file write does not: it used to open, write and close the JSONL
        file while holding this same process-wide mutex, which meant every
        concurrent inference serialized behind a filesystem round trip and the
        "parallel" action/monitor section could not actually overlap.
        """
        with self._log_lock:
            self._invocation_logs.append(log)

            is_reasoning = "Qwen" in log.model_name
            is_annotation = "Nemotron" in log.model_name

            self._stats["total"] += 1
            if is_reasoning:
                self._stats["reasoning_count"] += 1
                self._stats["reasoning_latency_ms"] += log.latency_ms
            if is_annotation:
                self._stats["annotation_count"] += 1
                self._stats["annotation_latency_ms"] += log.latency_ms

            self._stats["input_tokens"] += log.input_tokens
            self._stats["output_tokens"] += log.output_tokens

        # Persist outside the lock: this only serializes the record and enqueues
        # it. Never raises, so telemetry cannot fail a request.
        enqueue_jsonl(self._log_path, asdict(log))

    def get_logs(self, limit: Optional[int] = None) -> List[ModelInvocationLog]:
        """Get recent invocation logs."""
        with self._log_lock:
            if limit:
                return list(self._invocation_logs)[-limit:]
            return list(self._invocation_logs)

    def clear_logs(self) -> None:
        """Clear invocation logs."""
        with self._log_lock:
            self._invocation_logs.clear()
            for key in self._stats:
                self._stats[key] = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.

        Served from counters maintained as invocations are recorded, so cost no
        longer grows with history length. GPU memory is read outside the lock:
        it is a CUDA API call with its own cost and holding the invocation mutex
        across it would stall every recording thread.
        """
        with self._log_lock:
            stats = dict(self._stats)
            buffered = len(self._invocation_logs)

        if stats["total"] == 0:
            return {"total_invocations": 0}

        reasoning_count = stats["reasoning_count"]
        annotation_count = stats["annotation_count"]

        return {
            "total_invocations": stats["total"],
            "reasoning_invocations": reasoning_count,
            "annotation_invocations": annotation_count,
            "avg_reasoning_latency_ms": (
                stats["reasoning_latency_ms"] / reasoning_count
                if reasoning_count else 0
            ),
            "avg_annotation_latency_ms": (
                stats["annotation_latency_ms"] / annotation_count
                if annotation_count else 0
            ),
            "total_input_tokens": stats["input_tokens"],
            "total_output_tokens": stats["output_tokens"],
            # How many of the lifetime invocations are still held in memory.
            "buffered_invocations": buffered,
            "gpu_memory": get_model_info().get("gpu", {}),
            "log_writers": jsonl_writer_stats(),
        }

    @contextmanager
    def track_invocation(
        self,
        model_name: str,
        agent_name: str,
        input_text: Optional[str] = None,
        count_tokens: Optional[bool] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager to track a model invocation.

        Yields a dict to collect output tokens and confidence.

        Args:
            input_text: Optional prompt text, used only for token accounting.
                Do not pass a deep-rendered structure here: callers used to hand
                in ``str(messages)``, which flattened every retrieved chunk into
                one large string purely so it could then be tokenized again.
            count_tokens: Force exact token counting on or off. Defaults to
                ``config.EXACT_TOKEN_ACCOUNTING`` (false). Latency is always
                recorded regardless.

        The timed window deliberately excludes the input token count, which is
        performed before the yield, and includes only the generation itself.
        """
        start = time.perf_counter()

        exact = (
            config.EXACT_TOKEN_ACCOUNTING
            if count_tokens is None
            else bool(count_tokens)
        )
        input_tokens = self._count_tokens(model_name, input_text) if (
            exact and input_text
        ) else 0

        result = {"output_text": "", "confidence": None, "success": True, "error": None}

        try:
            yield result
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            raise
        finally:
            # This runs in a finally block, so a failure here would replace an
            # otherwise successful invocation with an exception. Token counting
            # and logging are bookkeeping and must never fail the caller.
            try:
                latency_ms = (time.perf_counter() - start) * 1000
                output_tokens = 0
                if exact:
                    output_tokens = self._count_tokens(
                        model_name, result.get("output_text", "")
                    )

                log = ModelInvocationLog(
                    model_name=model_name,
                    agent_name=agent_name,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    gpu_device=self._gpu_device,
                    confidence=result.get("confidence"),
                    success=result["success"],
                    error=result.get("error"),
                )
                self.log_invocation(log)
            except Exception as e:
                logger.warning(
                    f"Failed to record invocation for {agent_name} ({model_name}): {e}"
                )

    def _count_tokens(self, model_name: str, text: Any) -> int:
        """
        Count tokens using the appropriate tokenizer.

        Accepts a single string or a list of strings, because batched
        invocations track a list of generations. Returns a total count rather
        than the token id list, so the ids are not retained for longer than the
        count itself.

        Text is capped at ``config.TOKEN_COUNT_MAX_CHARS`` before tokenizing so a
        pathological payload cannot turn telemetry into a multi-second stall.
        """
        if "Qwen" in model_name:
            tokenizer = self.reasoning_tokenizer
        elif "Nemotron" in model_name:
            tokenizer = self.annotation_tokenizer
        else:
            return 0

        segments = text if isinstance(text, (list, tuple)) else [text]
        budget = max(1, int(config.TOKEN_COUNT_MAX_CHARS))

        total = 0
        for segment in segments:
            if not isinstance(segment, str):
                continue
            if budget <= 0:
                break
            clipped = segment[:budget]
            budget -= len(clipped)
            # add_special_tokens=False and no return tensors: this is a count,
            # so only the ids need to exist, never the attention mask.
            total += len(tokenizer.encode(clipped, add_special_tokens=False))

        return total

    def warm_up(self) -> None:
        """Warm up both models - now delegates to CUDA kernel warmup."""
        warmup_cuda_kernels()

    def shutdown(self) -> None:
        """Shutdown the pool and release models."""
        logger.info("Shutting down ModelPool")
        release_models()
        ModelPool._instance = None


# Global instance
_model_pool: Optional[ModelPool] = None


def get_model_pool() -> ModelPool:
    """Get the global ModelPool instance."""
    global _model_pool
    if _model_pool is None:
        _model_pool = ModelPool()
    return _model_pool