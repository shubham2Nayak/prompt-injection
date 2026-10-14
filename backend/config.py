"""
PromptTrace Backend Configuration
Centralized configuration for models, paths, and system parameters.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency for CPU-only setups
    torch = None

load_dotenv()

logger = logging.getLogger(__name__)

# Avoid fragmentation and allow the allocator to better reuse the limited VRAM
# available on shared T4 environments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _free_gpu_memory_mb() -> float:
    """Return free GPU memory in MB, or 0 when CUDA is unavailable."""
    if torch is None or not torch.cuda.is_available():
        return 0.0

    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        return free_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def _default_device() -> str:
    """Prefer CUDA when available; let model loader handle OOM via device_map=auto."""
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    return "cuda"


# ─── Centralized Model Registry ──────────────────────────────────────────────
# Two-model architecture:
#   1. Reasoning Model: Qwen/Qwen2.5-3B-Instruct (Research, Analyst, Action)
#   2. Safety Annotation Model: nvidia/Nemotron-3.5-Content-Safety (Context Engine, Propagation Monitor)
# Embedding Model: BAAI/bge-large-en-v1.5 (RAG)

RESEARCH_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ANALYST_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ACTION_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ANNOTATION_MODEL = "nvidia/Nemotron-3.5-Content-Safety"
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


def _parse_mode_overrides(raw: str) -> dict:
    """
    Parse "reasoning=nf4,annotation=int8" into a slot -> load mode mapping.

    Empty entries are ignored so an unset env var yields no overrides and the
    GPU budget planner picks the best plan that fits on its own.
    """
    overrides: dict = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        slot, _, mode = item.partition("=")
        slot, mode = slot.strip().lower(), mode.strip().lower()
        if slot and mode:
            overrides[slot] = mode
    return overrides


class Config:
    """Central configuration class for PromptTrace backend."""

    # Environment
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # FastAPI
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"

    # LLM Models (centralized registry)
    RESEARCH_AGENT_MODEL: str = os.getenv("RESEARCH_AGENT_MODEL", RESEARCH_MODEL)
    ANALYST_AGENT_MODEL: str = os.getenv("ANALYST_AGENT_MODEL", ANALYST_MODEL)
    ACTION_AGENT_MODEL: str = os.getenv("ACTION_AGENT_MODEL", ACTION_MODEL)
    PROPAGATION_MONITOR_MODEL: str = os.getenv("PROPAGATION_MONITOR_MODEL", ANNOTATION_MODEL)
    ANNOTATION_MODEL: str = os.getenv("ANNOTATION_MODEL", ANNOTATION_MODEL)

    # Annotation Model Configuration
    ANNOTATION_DEVICE: str = os.getenv("ANNOTATION_DEVICE", _default_device())
    ANNOTATION_TEMPERATURE: float = float(
        os.getenv("ANNOTATION_TEMPERATURE", "0.0")
    )
    # Number of prompts a single TextGenerationPipeline call may push through
    # model.generate() at once. HuggingFace defaults this to 1, which silently
    # turns every "batched" call into N sequential generate() invocations. This
    # value is forwarded to the pipeline constructor so batching is real.
    ANNOTATION_BATCH_SIZE: int = int(
        os.getenv("ANNOTATION_BATCH_SIZE", "8")
    )

    # Reasoning pipeline batch size. The three agents issue single-prompt calls,
    # so this only takes effect if a future change batches agent fan-out; it is
    # still passed explicitly rather than left at the transformers default.
    REASONING_BATCH_SIZE: int = int(
        os.getenv("REASONING_BATCH_SIZE", "1")
    )

    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", _default_device())

    # Model Configuration
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", _default_device())
    # bfloat16 by default: the Nemotron safety model is Gemma-3 based and
    # produces an all-NaN logit tensor in float16, which silently makes it emit
    # padding tokens instead of an answer. models.gpu_budget also pins
    # Gemma-family checkpoints to bfloat16 regardless of this setting.
    MODEL_DTYPE: str = os.getenv("MODEL_DTYPE", "float16")
    MAX_SEQ_LENGTH: int = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "4"))

    # LLM Generation Parameters (per-model temperatures)
    # Sampling temperature gradient, matching .env.example. The research agent
    # is allowed a little variance, the analyst only a little, and action and
    # annotation stay fully deterministic. This default had drifted to 0.0 for
    # the analyst, silently changing generation behaviour for anyone running
    # without an .env file.
    TEMPERATURE_RESEARCH: float = float(os.getenv("TEMPERATURE_RESEARCH", "0.2"))
    TEMPERATURE_ANALYST: float = float(os.getenv("TEMPERATURE_ANALYST", "0.1"))
    TEMPERATURE_ACTION: float = float(os.getenv("TEMPERATURE_ACTION", "0.0"))
    TEMPERATURE_ANNOTATION: float = float(os.getenv("TEMPERATURE_ANNOTATION", "0.0"))
    TOP_P: float = float(os.getenv("TOP_P", "0.9"))
    TOP_K: int = int(os.getenv("TOP_K", "20"))
    MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "512"))
    # Per-agent token limits for structured JSON outputs. Each agent is asked
    # for a multi-field JSON document, so a cap that is too small truncates the
    # object mid-string and the validator falls back to an empty result. These
    # sit at the global hard cap below, which is what actually bounds VRAM.
    MAX_NEW_TOKENS_RESEARCH: int = int(os.getenv("MAX_NEW_TOKENS_RESEARCH", "768"))
    MAX_NEW_TOKENS_ANALYST: int = int(os.getenv("MAX_NEW_TOKENS_ANALYST", "768"))
    MAX_NEW_TOKENS_ACTION: int = int(os.getenv("MAX_NEW_TOKENS_ACTION", "768"))
    # The annotation response is a small {category, confidence, reason} object,
    # so it needs far less than the reporting agents and stays cheap to batch.
    MAX_NEW_TOKENS_ANNOTATION: int = int(os.getenv("MAX_NEW_TOKENS_ANNOTATION", "256"))

    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "256"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "32"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

    # File Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    UPLOAD_DIR: str = os.path.join(DATA_DIR, "uploads")
    INDEX_DIR: str = os.path.join(DATA_DIR, "indices")
    LOGS_DIR: str = os.path.join(DATA_DIR, "logs")
    CACHE_DIR: str = os.path.join(DATA_DIR, "cache")

    # Vector Store Configuration
    FAISS_METRIC: str = os.getenv("FAISS_METRIC", "L2")
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "Flat")
    # GPU ordinal used for the FAISS index. Kept separate from MODEL_DEVICE so
    # the vector store can be moved independently of the LLMs.
    FAISS_GPU_DEVICE: int = int(os.getenv("FAISS_GPU_DEVICE", "0"))
    # When True, a request for a GPU index that cannot be created raises instead
    # of silently degrading to CPU. The previous behaviour logged a warning and
    # continued, which hid the fact that every search was paying a blocking
    # device-to-host copy on a machine believed to be doing GPU retrieval.
    FAISS_REQUIRE_GPU: bool = os.getenv(
        "FAISS_REQUIRE_GPU", "true"
    ).lower() == "true"

    # Concurrency / scheduling
    # Simulations run on a dedicated thread pool so the FastAPI event loop stays
    # responsive. More than one worker on a single GPU usually trades throughput
    # for VRAM thrash, so the default serialises GPU work while still freeing
    # the event loop for health, metrics, logs and graph requests.
    SIMULATION_CONCURRENCY: int = int(os.getenv("SIMULATION_CONCURRENCY", "1"))

    # Metrics retention
    # Workflow states are large (retrieved chunk text, agent outputs, timelines),
    # so retaining every run was an unbounded memory leak. All scalar metrics are
    # aggregated incrementally at record time, which means this window only
    # bounds the per-run detail kept for the timeline/graph views and does not
    # change any reported number.
    METRICS_HISTORY_SIZE: int = int(os.getenv("METRICS_HISTORY_SIZE", "200"))
    # Distinct propagation paths retained for reporting. Bounded because the
    # cardinality is otherwise driven by agent-name permutations.
    METRICS_PATH_CACHE_SIZE: int = int(os.getenv("METRICS_PATH_CACHE_SIZE", "500"))

    # Propagation log retention
    # In-memory ring of recent propagation events. Served directly by the log
    # endpoints so no request has to re-read and re-parse the JSONL file.
    PROPAGATION_EVENT_BUFFER_SIZE: int = int(
        os.getenv("PROPAGATION_EVENT_BUFFER_SIZE", "5000")
    )
    # Lines read from the tail of an existing propagation_events.jsonl when the
    # logger starts, so events recorded before a restart remain visible.
    PROPAGATION_LOG_BOOTSTRAP_LINES: int = int(
        os.getenv("PROPAGATION_LOG_BOOTSTRAP_LINES", "2000")
    )

    # Token accounting / telemetry
    # Exact token counting duplicates work the generation pipeline already does:
    # it re-renders prompts and re-runs the tokenizer on every invocation. It is
    # therefore opt-in. Set to true when exact per-agent token counts are needed
    # for reporting; latency_ms is always recorded either way.
    EXACT_TOKEN_ACCOUNTING: bool = os.getenv(
        "EXACT_TOKEN_ACCOUNTING", "false"
    ).lower() == "true"
    # Hard ceiling on characters handed to the tokenizer for a single count, so a
    # pathological payload cannot turn telemetry into a multi-second stall.
    TOKEN_COUNT_MAX_CHARS: int = int(os.getenv("TOKEN_COUNT_MAX_CHARS", "20000"))

    # Invocation log retention
    # In-memory ring buffer size for per-invocation telemetry.
    INVOCATION_LOG_MAX_ENTRIES: int = int(
        os.getenv("INVOCATION_LOG_MAX_ENTRIES", "5000")
    )
    # Background log writer behaviour.
    INVOCATION_LOG_MAX_BYTES: int = int(
        os.getenv("INVOCATION_LOG_MAX_BYTES", str(16 * 1024 * 1024))
    )
    INVOCATION_LOG_BACKUP_COUNT: int = int(
        os.getenv("INVOCATION_LOG_BACKUP_COUNT", "3")
    )
    # Bounded queue depth. If the writer thread cannot keep up, oldest entries
    # are dropped rather than blocking the inference path.
    INVOCATION_LOG_QUEUE_SIZE: int = int(
        os.getenv("INVOCATION_LOG_QUEUE_SIZE", "20000")
    )

    # GPU Memory Budget
    # VRAM held back for the CUDA context, the KV cache of both LLMs and
    # transient activation buffers, so resident model weights never fill the card.
    GPU_RESERVE_GB: float = float(os.getenv("GPU_RESERVE_GB", "2.0"))
    # Optional per-slot precision overrides, e.g. "reasoning=nf4,annotation=int8".
    # Valid modes: fp16, int8, nf4, cpu. Leave empty to let the planner decide.
    GPU_LOAD_MODE_OVERRIDES: dict = _parse_mode_overrides(
        os.getenv("GPU_LOAD_MODE_OVERRIDES", "")
    )

    # Propagation Monitor Configuration
    PROPAGATION_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("PROPAGATION_CONFIDENCE_THRESHOLD", "0.6")
    )
    MUTATION_DETETION_ENABLED: bool = (
        os.getenv("MUTATION_DETECTION_ENABLED", "true").lower() == "true"
    )

    # Logging
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    )

    # Timeouts
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "300"))
    EMBEDDING_TIMEOUT: int = int(os.getenv("EMBEDDING_TIMEOUT", "120"))

    # System Limits
    MAX_PDF_SIZE_MB: int = int(os.getenv("MAX_PDF_SIZE_MB", "20"))
    MAX_CHUNKS_PER_QUERY: int = int(os.getenv("MAX_CHUNKS_PER_QUERY", "5"))

    # Token Cap Enforcement
    # Global hard cap for all MAX_NEW_TOKENS* settings to prevent GPU waste
    _MAX_TOKEN_CAP: int = 768

    @classmethod
    def _enforce_token_caps(cls) -> None:
        """Enforce global maximum on all MAX_NEW_TOKENS* settings."""
        token_fields = [
            "MAX_NEW_TOKENS",
            "MAX_NEW_TOKENS_RESEARCH",
            "MAX_NEW_TOKENS_ANALYST",
            "MAX_NEW_TOKENS_ACTION",
            "MAX_NEW_TOKENS_ANNOTATION",
        ]
        for field in token_fields:
            current = getattr(cls, field, 0)
            if current > cls._MAX_TOKEN_CAP:
                logger.warning(
                    f"Capping {field} from {current} to {cls._MAX_TOKEN_CAP} "
                    f"(global hard cap)"
                )
                setattr(cls, field, cls._MAX_TOKEN_CAP)

    @classmethod
    def validate_and_create_dirs(cls) -> None:
        """Create necessary directories if they don't exist."""
        # Enforce token caps first
        cls._enforce_token_caps()
        for directory in [cls.UPLOAD_DIR, cls.INDEX_DIR, cls.LOGS_DIR, cls.CACHE_DIR]:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def to_dict(cls) -> dict:
        """Convert config to dictionary for logging."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and key.isupper()
        }


# Initialize config
config = Config()
config.validate_and_create_dirs()