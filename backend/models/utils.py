"""
Model utilities for PromptTrace
Helper functions for model operations, JSON extraction, and validation.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from model output.
    
    Tries multiple strategies:
    1. Direct JSON parse
    2. Find JSON object in text
    3. Strip markdown code fences
    """
    if not text:
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Find JSON object in text
    json_match = re.search(r"\{[^{}]*(?:\"[a-zA-Z_]+\")[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 3: Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    # Strategy 4: Find first { to last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


def validate_json_schema(
    data: Dict[str, Any],
    required_fields: List[str],
    field_types: Optional[Dict[str, type]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate JSON against expected schema.
    
    Returns:
        (is_valid, error_message)
    """
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"

    if field_types:
        for field, expected_type in field_types.items():
            if field in data and not isinstance(data[field], expected_type):
                return False, f"Field {field} has wrong type: expected {expected_type.__name__}, got {type(data[field]).__name__}"

    return True, None


def validate_confidence(value: Any) -> float:
    """Validate and clamp confidence value to [0, 1]."""
    try:
        conf = float(value)
        return max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        return 0.0


def validate_category(category: str, valid_categories: List[str]) -> str:
    """Validate category against known categories."""
    if not category:
        return "UNKNOWN"
    cat_upper = category.strip().upper()
    for valid in valid_categories:
        if valid.upper() == cat_upper:
            return valid
    return "UNKNOWN"


ANNOTATION_CATEGORIES = [
    "ACADEMIC_DESCRIPTION",
    "QUOTED_EXAMPLE",
    "EMBEDDED_PROCESSING_NOTE",
    "MIXED_CONTENT",
    "UNKNOWN",
]


def validate_annotation_output(raw: str) -> Dict[str, Any]:
    """
    Validate and normalize annotation model output.
    
    Returns dict with: category, confidence, reason, annotation_category, infected
    """
    parsed = extract_json(raw)
    if parsed is None:
        logger.warning("Failed to parse annotation JSON")
        return {
            "category": "UNKNOWN",
            "confidence": 0.0,
            "reason": "JSON parsing failed",
            "annotation_category": "UNKNOWN",
            "infected": False,
        }

    category = validate_category(parsed.get("category", "UNKNOWN"), ANNOTATION_CATEGORIES)
    confidence = validate_confidence(parsed.get("confidence", 0.0))
    reason = str(parsed.get("reason", "No reason provided"))[:200]
    infected = parsed.get("infected", category in {"EMBEDDED_PROCESSING_NOTE", "MIXED_CONTENT"})

    return {
        "category": category,
        "confidence": confidence,
        "reason": reason,
        "annotation_category": category,
        "infected": bool(infected),
    }


def validate_reasoning_output(
    raw: str,
    required_fields: List[str],
) -> Dict[str, Any]:
    """
    Validate reasoning model output (Research, Analyst, Action).
    
    Returns parsed dict or minimal valid structure.
    """
    parsed = extract_json(raw)
    if parsed is None:
        logger.warning("Failed to parse reasoning JSON")
        return {field: [] if field.endswith("s") else "" for field in required_fields}

    # Ensure all required fields exist
    for field in required_fields:
        if field not in parsed:
            parsed[field] = [] if field.endswith("s") else ""

    return parsed


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with provided variables."""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"Missing template variable: {e}")
        return template


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens in text using provided tokenizer."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to max tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated = tokens[:max_tokens]
    return tokenizer.decode(truncated, skip_special_tokens=True)


def ensure_cuda() -> bool:
    """Check if CUDA is available and log GPU info."""
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - running on CPU")
        return False

    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return True


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage in MB."""
    import torch
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0}

    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    free = total - reserved

    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "free_mb": round(free, 2),
        "total_mb": round(total, 2),
    }