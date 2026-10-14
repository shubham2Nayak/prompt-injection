"""JSON validator for model-generated annotation outputs.

Validates JSON returned by the model with zero retries.
If validation fails, returns UNKNOWN immediately. Never fabricates
values - if validation fails, the result is categorized as UNKNOWN.
"""

import json
import logging
import re
from typing import Any, Optional

from .taxonomy import AnnotationCategory, UNKNOWN as UNK_CATEGORY
from .schema import ContextAnnotation, AnnotationSpan

logger = logging.getLogger(__name__)


def safe_extract_json(raw_text: str) -> dict:
    """
    Robustly extract and parse JSON from LLM output that may contain
    markdown code fences (```json ... ```) or conversational filler text.
    
    Args:
        raw_text: Raw string output from the LLM
        
    Returns:
        Parsed JSON dictionary
        
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    if not raw_text or not isinstance(raw_text, str):
        raise ValueError("Empty or non-string input received")

    # 1. Strip markdown backticks (```json ... ```)
    # Handle multi-line code fences with content between them
    cleaned = raw_text.strip()
    # Remove opening fence (``` or ```json) and everything before it on that line
    cleaned = re.sub(r"^.*```(?:json)?\s*\n?", "", cleaned, flags=re.MULTILINE)
    # Remove closing fence (```) and everything after it on that line
    cleaned = re.sub(r"\n?```.*$", "", cleaned, flags=re.MULTILINE)

    # 2. Extract JSON object substring if conversational filler surrounds it
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)

    # 3. Parse JSON
    return json.loads(cleaned)


def _try_parse_json(raw: str) -> Optional[dict]:
    """Try to parse a string as JSON using safe_extract_json. Returns dict or None."""
    try:
        return safe_extract_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _normalize_category(value: Any) -> str:
    """Normalize a category value to a taxonomy string."""
    if isinstance(value, AnnotationCategory):
        return str(value)
    if isinstance(value, str):
        # Match against known categories case-insensitively
        for cat in AnnotationCategory:
            if cat.name.upper() == value.upper():
                return cat.name
        # Lowercase fallback
        return value.strip().upper()
    return str(value).strip().upper()


# The Nemotron safety checkpoint is a Llama-Guard style classifier, not a
# general chat model. Its chat template hard-codes the answer format and
# discards the system message, so it answers in plain text:
#
#     User Safety: safe
#     Response Safety: unsafe
#     Safety Categories: S15
#
# Map those verdicts onto the PromptTrace context taxonomy, which classifies
# how retrieved text should be treated rather than whether it is harmful.
SAFETY_CATEGORY_BY_VERDICT = {
    "safe": "ACADEMIC_DESCRIPTION",
    "unsafe": "EMBEDDED_PROCESSING_NOTE",
    "refusal": "MIXED_CONTENT",
}


def _parse_safety_field(text: str, field: str) -> Optional[str]:
    """Return the verdict for one 'X Safety:' field, lowercased, or None."""
    match = re.search(rf"^{field}\s*Safety\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    if not match:
        return None
    verdict = match.group(1).strip().strip(".").lower()
    for known in SAFETY_CATEGORY_BY_VERDICT:
        if verdict.startswith(known):
            return known
    return None or verdict or None


def validate_safety_classifier_output(raw: str) -> Optional[dict]:
    """
    Parse the Nemotron safety classifier's plain-text answer format.

    Returns a dict shaped like :func:`validate_annotation_json` (keys category,
    confidence, reason), or None when the text is not in this format so the
    caller can fall back to JSON parsing.

    The model emits a hard safe/unsafe label and no numeric score, so the
    confidence below is a fixed weight on how explicit the verdict was, not a
    probability produced by the model.
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw.strip()
    if not re.search(r"^\w+\s*Safety\s*:", text, re.MULTILINE | re.IGNORECASE):
        return None

    user_verdict = _parse_safety_field(text, "User")
    response_verdict = _parse_safety_field(text, "Response")

    categories_match = re.search(
        r"^Safety Categories\s*:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE
    )
    safety_categories = categories_match.group(1).strip() if categories_match else ""

    verdicts = [v for v in (user_verdict, response_verdict) if v]
    if not verdicts:
        return {
            "category": UNK_CATEGORY,
            "confidence": 0.0,
            "reason": f"Could not read a safety verdict from: {text[:200]}",
        }

    if "unsafe" in verdicts:
        category = "MIXED_CONTENT" if "safe" in verdicts else "EMBEDDED_PROCESSING_NOTE"
    elif all(v == "refusal" for v in verdicts):
        category = "MIXED_CONTENT"
    else:
        category = "ACADEMIC_DESCRIPTION"

    # An explicit unsafe verdict with named hazard categories is the strongest
    # signal the model gives; a bare "safe" is next; ambiguity scores lowest.
    if "unsafe" in verdicts and safety_categories:
        confidence = 0.9
    elif "unsafe" in verdicts:
        confidence = 0.8
    elif verdicts:
        confidence = 0.85
    else:
        confidence = 0.5

    reason = safety_categories or f"model verdict: {', '.join(verdicts)}"
    return {"category": category, "confidence": confidence, "reason": reason}


def validate_annotation_output(raw: str) -> dict:
    """
    Validate annotation output from the annotation model.

    Handles both response formats: the Nemotron safety classifier's plain-text
    verdicts and a JSON object. Returns the UNKNOWN annotation on failure.
    """
    safety = validate_safety_classifier_output(raw)
    if safety is not None:
        return safety

    parsed = validate_annotation_json(raw)
    if parsed.get("category", UNK_CATEGORY) == UNK_CATEGORY:
        logger.debug(
            "Annotation output matched neither the safety verdict format nor "
            "a valid JSON annotation"
        )
    return parsed


def validate_annotation_json(
    raw: str,
    allow_retry: bool = False,
) -> dict:
    """
    Validate and normalize a JSON annotation string with ZERO retries.

    Steps:
    1. Try to parse the raw string as JSON using safe_extract_json (handles markdown, filler).
    2. If parsing fails, IMMEDIATELY return {category: UNKNOWN, confidence: 0.0}.
    3. If validation succeeds, normalize the category field.
    4. The LLM is NEVER queried a second time for a failed JSON parse.

    Returns:
        dict with keys: category, confidence, reason
    """
    # Step 1: try robust parse (handles markdown code fences, filler text)
    parsed = _try_parse_json(raw)

    if parsed is None:
        # IMMEDIATE failure - return UNKNOWN, NO RETRIES
        logger.warning("Annotation JSON validation failed - zero retry policy enforced")
        return {"category": UNK_CATEGORY, "confidence": 0.0, "reason": "JSON validation failed"}

    # Step 2: normalize category
    category = parsed.get("category", UNK_CATEGORY)
    normalized_cat = _normalize_category(category)

    # Validate against taxonomy
    # Check that the normalized category is a known category name
    valid_categories = {cat.name for cat in AnnotationCategory}
    if normalized_cat not in valid_categories:
        normalized_cat = UNK_CATEGORY

    confidence = parsed.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        confidence = 0.0

    reason = parsed.get("reason", "No reason provided")

    return {
        "category": normalized_cat,
        "confidence": float(confidence),
        "reason": str(reason),
    }


def validate_context_annotation(
    raw: str,
) -> ContextAnnotation:
    """
    Validate a full ContextAnnotation JSON string.

    Returns a ContextAnnotation Pydantic model instance.
    If validation fails, returns a ContextAnnotation with UNKNOWN category.
    """
    parsed = validate_annotation_json(raw)
    try:
        # Try to build the full model
        # We only have category and confidence at the top level from JSON;
        # other fields need additional structure
        return ContextAnnotation(
            chunk_id=parsed.get("chunk_id", ""),
            paragraph_offset=parsed.get("paragraph_offset", 0),
            character_offset=parsed.get("character_offset", 0),
            annotation_category=parsed.get("annotation_category", UNK_CATEGORY),
            confidence=parsed.get("confidence", 0.0),
            spans=AnnotationSpan.validate_spans(parsed.get("spans", [])),
            model_used=parsed.get("model_used", "unknown"),
            latency_ms=parsed.get("latency_ms", 0.0),
        )
    except Exception as e:
        logger.error(f"Failed to build ContextAnnotation: {str(e)}")
        # Return minimal viable model
        return ContextAnnotation(
            chunk_id="",
            paragraph_offset=0,
            character_offset=0,
            annotation_category=UNK_CATEGORY,
            confidence=0.0,
            spans=[],
            model_used="unknown",
            latency_ms=0.0,
        )