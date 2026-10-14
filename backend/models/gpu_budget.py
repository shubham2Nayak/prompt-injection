"""
GPU VRAM Budget Planner for PromptTrace.

PromptTrace keeps three models resident on the accelerator at the same time:

    embedding  BAAI/bge-large-en-v1.5              ~0.34B params
    annotation nvidia/Nemotron-3.5-Content-Safety   ~4.30B params (Gemma-3 text + SigLIP vision tower)
    reasoning  Qwen/Qwen2.5-3B-Instruct            ~3.09B params

At full fp16 those weights total ~15.0 GiB, which is already larger than the
14.56 GiB of usable VRAM on a T4 - before the CUDA context, the KV cache, or a
single activation buffer. The stack therefore cannot run unquantized on a 16 GB
card no matter how the models are ordered; it OOMs on whichever model is loaded
last.

This module decides, once at startup, which precision each model gets so that
the whole resident set fits in the memory that is actually free at that moment.
It prefers full fp16 and only degrades the models that must be degraded,
quantizing the reasoning model before the annotation model because annotation
is the security-critical part of the pipeline.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GIB = float(2**30)

# Bytes of VRAM per parameter. The fp16 column is the measured resident
# footprint of each checkpoint on this stack; the quantized columns are the
# per-parameter averages measured across both LLMs on a T4 (torch 2.5.1,
# transformers 4.50, bitsandbytes 0.50). They exceed the theoretical ratios
# because bitsandbytes deliberately leaves embeddings and the lm_head in fp16,
# which is a large share of both models.
FP16_BYTES_PER_PARAM = 2.0
INT8_BYTES_PER_PARAM = 1.03
NF4_BYTES_PER_PARAM = 0.67

FP16 = "fp16"
INT8 = "int8"
NF4 = "nf4"
CPU = "cpu"

_QUANTIZED = {INT8, NF4}


@dataclass
class ModelSpec:
    """Static description of one resident model."""

    slot: str
    model_name: str
    params_b: float
    fp16_bytes: float
    is_critical: bool = False
    note: str = ""

    def vram_bytes(self, mode: str) -> float:
        """Estimated resident VRAM for this model at the given load mode."""
        if mode == CPU:
            return 0.0
        if mode == FP16:
            return self.fp16_bytes
        factor = INT8_BYTES_PER_PARAM if mode == INT8 else NF4_BYTES_PER_PARAM
        return self.params_b * factor * GIB


# Measured resident VRAM of each checkpoint on a T4, in GiB.
#   bge-large-en-v1.5              0.335B params, 0.62 fp16 / 1.25 fp32
#   Nemotron-3.5-Content-Safety    4.300B params, 7.23 fp16 / 4.32 int8 / 2.95 nf4
#   Qwen2.5-3B-Instruct            3.086B params, 5.75 fp16 / 3.21 int8 / 2.03 nf4
# Together at fp16 they are 13.60 GiB of weights against 14.56 GiB of usable
# VRAM, which is why the full-fidelity rung of the ladder is unreachable here.
REGISTRY: Dict[str, ModelSpec] = {
    "embedding": ModelSpec(
        slot="embedding",
        model_name="BAAI/bge-large-en-v1.5",
        params_b=0.335,
        fp16_bytes=0.624 * GIB,
        note="RAG embeddings; fp16 halves the fp32 SentenceTransformer default",
    ),
    "annotation": ModelSpec(
        slot="annotation",
        model_name="nvidia/Nemotron-3.5-Content-Safety",
        params_b=4.300,
        fp16_bytes=7.23 * GIB,
        is_critical=True,
        note="Safety classifier; kept at the highest precision the budget allows",
    ),
    "reasoning": ModelSpec(
        slot="reasoning",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        params_b=3.086,
        fp16_bytes=5.75 * GIB,
        note="Research/Analyst/Action; degraded first because its quality is the "
             "least sensitive to quantization here",
    ),
}

# Ordered from highest to lowest fidelity. The first plan that fits wins, so a
# larger GPU automatically gets better precision with no extra configuration.
PLAN_LADDER: List[Dict[str, str]] = [
    {"embedding": FP16, "annotation": FP16, "reasoning": FP16},
    {"embedding": FP16, "annotation": FP16, "reasoning": INT8},
    {"embedding": FP16, "annotation": INT8, "reasoning": INT8},
    {"embedding": FP16, "annotation": NF4, "reasoning": INT8},
    {"embedding": FP16, "annotation": NF4, "reasoning": NF4},
    {"embedding": CPU, "annotation": NF4, "reasoning": NF4},
]

# Downgrade ladders used when a load still OOMs at runtime despite the plan.
DEGRADE: Dict[str, str] = {FP16: INT8, INT8: NF4, NF4: CPU, CPU: CPU}


class GpuBudgetError(RuntimeError):
    """Raised when no plan can fit the resident model set in free VRAM."""


@dataclass
class LoadPlan:
    """The chosen per-model load modes plus the arithmetic behind them."""

    modes: Dict[str, str]
    free_gib: float
    total_gib: float
    reserve_gib: float
    weights_gib: float
    steps: List[str] = field(default_factory=list)

    @property
    def headroom_gib(self) -> float:
        return self.free_gib - self.weights_gib - self.reserve_gib

    def mode_for(self, slot: str) -> str:
        return self.modes.get(slot, CPU)

    def is_quantized(self, slot: str) -> bool:
        return self.mode_for(slot) in _QUANTIZED

    def to_dict(self) -> dict:
        return {
            "modes": dict(self.modes),
            "free_gib": round(self.free_gib, 2),
            "total_gib": round(self.total_gib, 2),
            "reserve_gib": round(self.reserve_gib, 2),
            "weights_gib": round(self.weights_gib, 2),
            "headroom_gib": round(self.headroom_gib, 2),
            "notes": [REGISTRY[s].note for s in self.modes if REGISTRY[s].note],
        }

    def describe(self) -> str:
        parts = ", ".join(
            f"{slot}={mode}" for slot, mode in sorted(self.modes.items())
        )
        return (
            f"{parts} | weights {self.weights_gib:.2f} GiB of "
            f"{self.free_gib:.2f} GiB free "
            f"(reserve {self.reserve_gib:.2f} GiB, headroom {self.headroom_gib:.2f} GiB)"
        )


def cuda_free_memory() -> Tuple[float, float]:
    """Return (free_gib, total_gib) for the active CUDA device, or (0, 0)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return free_bytes / GIB, total_bytes / GIB
    except Exception as exc:  # pragma: no cover - driver/runtime edge cases
        logger.warning(f"Could not query CUDA memory: {exc}")
        return 0.0, 0.0


def build_plan(
    free_gib: float,
    total_gib: float,
    reserve_gib: float,
    overrides: Optional[Dict[str, str]] = None,
) -> LoadPlan:
    """
    Pick the highest-fidelity plan whose weights fit in ``free_gib``.

    ``reserve_gib`` is held back for the CUDA context, the KV cache of both LLMs
    and transient activation buffers, so the weights never fill the card.

    Raises:
        GpuBudgetError: when even the cheapest plan does not fit.
    """
    overrides = overrides or {}
    steps: List[str] = []

    for candidate in PLAN_LADDER:
        modes = dict(candidate)
        applied = {slot: mode for slot, mode in overrides.items() if slot in modes}
        if applied:
            modes.update(applied)
            steps.append(f"forced {applied} from config override")
        weights = sum(
            REGISTRY[slot].vram_bytes(mode) for slot, mode in modes.items()
        ) / GIB
        if weights + reserve_gib <= free_gib:
            return LoadPlan(
                modes=modes,
                free_gib=free_gib,
                total_gib=total_gib,
                reserve_gib=reserve_gib,
                weights_gib=weights,
                steps=steps,
            )
        steps.append(
            f"skipped {modes} ({weights:.2f} GiB weights + {reserve_gib:.2f} GiB "
            f"reserve exceeds {free_gib:.2f} GiB free)"
        )

    cheapest = sum(
        REGISTRY[slot].vram_bytes(mode) for slot, mode in PLAN_LADDER[-1].items()
    ) / GIB
    raise GpuBudgetError(
        f"PromptTrace needs {cheapest + reserve_gib:.2f} GiB of VRAM for its three "
        f"resident models plus runtime headroom, but only {free_gib:.2f} GiB of "
        f"{total_gib:.2f} GiB is free. Another process is holding the GPU, or the "
        f"card is too small. Free the GPU (nvidia-smi) or lower EMBEDDING_DEVICE=cpu."
    )


_plan: Optional[LoadPlan] = None
_plan_lock = threading.Lock()


def get_gpu_plan(refresh: bool = False) -> LoadPlan:
    """Return the process-wide load plan, computing it on first use."""
    global _plan
    if _plan is not None and not refresh:
        return _plan
    with _plan_lock:
        if _plan is not None and not refresh:
            return _plan
        from config import config

        free_gib, total_gib = cuda_free_memory()
        if total_gib == 0.0:
            # CPU-only host: everything runs on CPU, there is no budget to plan.
            _plan = LoadPlan(
                modes={slot: CPU for slot in REGISTRY},
                free_gib=0.0,
                total_gib=0.0,
                reserve_gib=0.0,
                weights_gib=0.0,
                steps=["CUDA unavailable, all models on CPU"],
            )
        else:
            _plan = build_plan(
                free_gib=free_gib,
                total_gib=total_gib,
                reserve_gib=config.GPU_RESERVE_GB,
                overrides=config.GPU_LOAD_MODE_OVERRIDES,
            )
            for step in _plan.steps:
                logger.info(f"GPU budget: {step}")
        logger.info(f"GPU load plan: {_plan.describe()}")
        return _plan


def torch_dtype_for(mode: str, model_name: str = ""):
    """
    Return the torch dtype to load a model with under the given mode.

    Gemma-3 - the backbone of the Nemotron safety model - yields an all-NaN
    logit tensor in float16 because its activations overflow the fp16 range,
    which makes it emit padding tokens forever instead of an answer. Models in
    that family are therefore pinned to bfloat16 whatever the configured dtype
    says. Bitsandbytes handles its own compute dtype for the quantized modes.
    """
    import torch

    if mode == NF4 or mode == INT8:
        return torch.float16
    from config import config

    configured = config.MODEL_DTYPE.lower()
    if any(family in model_name.lower() for family in ("gemma", "nemotron")):
        if configured == "float16":
            logger.warning(
                f"{model_name} cannot run in float16 (NaN logits); "
                f"loading in bfloat16 instead"
            )
        return torch.bfloat16

    if configured == "bfloat16":
        return torch.bfloat16
    # Compared against the lowercased value like every other branch here; the
    # raw config string made this unreachable for "Float32"/"FLOAT32".
    if configured == "float32":
        return torch.float32
    return torch.float16


def is_quantized_mode(mode: str) -> bool:
    """True when the mode needs bitsandbytes to load."""
    return mode in _QUANTIZED


def quantization_available() -> bool:
    """True when bitsandbytes is importable and can be used for quantization."""
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def quantization_config_for(mode: str):
    """
    Return a BitsAndBytesConfig for a quantized mode, or None for fp16/CPU.

    Returns None for a quantized mode when bitsandbytes is unavailable, so the
    caller can skip that rung instead of failing the whole load.
    """
    if mode not in _QUANTIZED:
        return None
    if not quantization_available():
        return None
    from transformers import BitsAndBytesConfig

    if mode == INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    import torch

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def set_mode(slot: str, mode: str) -> None:
    """Downgrade a slot in the active plan after a runtime OOM."""
    plan = get_gpu_plan()
    current = plan.mode_for(slot)
    if current == mode:
        return
    plan.modes[slot] = mode
    plan.weights_gib = sum(
        REGISTRY[s].vram_bytes(m) for s, m in plan.modes.items()
    ) / GIB
    logger.warning(
        f"GPU budget: downgraded {slot} {current} -> {mode} after OOM; "
        f"plan is now {plan.describe()}"
    )


def next_mode(mode: str) -> str:
    """Return the next fallback load mode after ``mode``."""
    return DEGRADE.get(mode, CPU)
