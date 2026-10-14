"""Repository-wide runtime diagnostics for the PromptTrace pipeline.

This module inspects live objects at pipeline boundaries and prints a
compact report that makes contract violations obvious: the function that
expected a shape, the type it actually received, the field names present,
and the fields that are missing or unexpected. It is intentionally
read-only and safe to call from any stage of the pipeline.

Usage:
    from debug.contracts import check_contract, diagnose_object

    check_contract(
        "rag.chunking.TextChunker.batch_chunk_pages",
        expected_fields=["text", "chunk_id", "page_number"],
        actual=chunk,
    )
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, get_args, get_origin

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field introspection
# ---------------------------------------------------------------------------

def _field_names(obj: Any) -> List[str]:
    """Return the set of accessible field/attribute names for *obj*."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        return list(obj.keys())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return [f.name for f in dataclasses.fields(obj)]
    # Pydantic models and other objects with __dict__ / model_fields
    fields_attr = getattr(obj, "model_fields", None)
    if isinstance(fields_attr, dict):
        return list(fields_attr.keys())
    if hasattr(obj, "__dict__"):
        return [k for k in vars(obj).keys() if not k.startswith("_")]
    # Tuples/lists expose indices, not names
    if isinstance(obj, (tuple, list)):
        return [f"idx_{i}" for i in range(len(obj))]
    return []


def _type_name(obj: Any) -> str:
    """A short, human-readable type name for *obj*."""
    if obj is None:
        return "None"
    if isinstance(obj, dict):
        return "dict"
    if isinstance(obj, list):
        return "list"
    if isinstance(obj, tuple):
        return "tuple"
    if isinstance(obj, set):
        return "set"
    return type(obj).__name__


# ---------------------------------------------------------------------------
# Public diagnostics
# ---------------------------------------------------------------------------

def diagnose_object(
    label: str,
    expected_fields: Optional[Sequence[str]] = None,
    actual: Any = None,
    raise_on_violation: bool = False,
    allow_extra: bool = True,
) -> Dict[str, Any]:
    """
    Inspect *actual* and report on its fields versus *expected_fields*.

    Args:
        label: Human-readable name of the boundary being checked
            (e.g. "Parser -> Chunker").
        expected_fields: Field names the consumer expects to find.
        actual: The object that was actually produced.
        raise_on_violation: If True, raise ValueError on a missing/extra field.
        allow_extra: If True, extra fields in actual are not treated as violations.

    Returns:
        Dict with keys: label, actual_type, field_names, missing_fields,
        extra_fields, ok.

    Raises:
        ValueError: If *raise_on_violation* is True and the contract is
            violated.
    """
    actual_fields = _field_names(actual)
    expected = list(expected_fields) if expected_fields else []

    missing = [f for f in expected if f not in actual_fields]
    extra = [f for f in actual_fields if f not in expected] if expected else []

    # Only treat extra as violation if allow_extra is False
    ok = not missing and (allow_extra or not extra)

    report = {
        "label": label,
        "actual_type": _type_name(actual),
        "field_names": actual_fields,
        "missing_fields": missing,
        "extra_fields": extra,
        "ok": ok,
    }

    if not ok:
        logger.warning(
            "Contract violation at %s: actual_type=%s missing=%s extra=%s",
            label, report["actual_type"], missing, extra,
        )
        if raise_on_violation:
            raise ValueError(
                f"Contract violation at {label}: "
                f"actual_type={report['actual_type']}, "
                f"missing={missing}, extra={extra}"
            )

    return report


def check_contract(
    stage: str,
    expected_fields: Sequence[str],
    actual: Any,
    raise_on_violation: bool = False,
    allow_extra: bool = True,
) -> Dict[str, Any]:
    """
    Convenience wrapper around diagnose_object for stage boundaries.

    Args:
        stage: Name of the pipeline stage (e.g. "Parser -> Chunker").
        expected_fields: Fields the next stage expects.
        actual: Object produced by the previous stage.
        raise_on_violation: If True, raise on contract violation.
        allow_extra: If True, extra fields in actual are not treated as violations.

    Returns:
        See diagnose_object.
    """
    return diagnose_object(
        label=stage,
        expected_fields=expected_fields,
        actual=actual,
        raise_on_violation=raise_on_violation,
        allow_extra=allow_extra,
    )


def validate_sequence(
    stage: str,
    expected_fields: Sequence[str],
    actual: Any,
    raise_on_violation: bool = False,
    allow_extra: bool = True,
) -> List[Dict[str, Any]]:
    """
    Validate every element of a sequence against the expected field set.

    Useful when a stage returns a list of records (chunks, annotations) and
    each record must satisfy the downstream contract.

    Args:
        stage: Name of the pipeline stage.
        expected_fields: Fields each record must carry.
        actual: A sequence of records.
        raise_on_violation: If True, raise on the first violation.
        allow_extra: If True, extra fields in records are not treated as violations.

    Returns:
        List of per-record diagnostics.

    Raises:
        ValueError: If *raise_on_violation* is True and any record violates.
    """
    if actual is None:
        return []
    if not isinstance(actual, (list, tuple)):
        raise TypeError(
            f"validate_sequence expects a sequence, got {type(actual).__name__}"
        )

    reports: List[Dict[str, Any]] = []
    for idx, item in enumerate(actual):
        report = diagnose_object(
            label=f"{stage}[{idx}]",
            expected_fields=expected_fields,
            actual=item,
            raise_on_violation=False,
            allow_extra=allow_extra,
        )
        report["index"] = idx
        reports.append(report)
        if not report["ok"] and raise_on_violation:
            raise ValueError(
                f"Contract violation at {stage}[{idx}]: "
                f"actual_type={report['actual_type']}, "
                f"missing={report['missing_fields']}, "
                f"extra={report['extra_fields']}"
            )
    return reports


def summarize(reports: Sequence[Dict[str, Any]]) -> str:
    """Render a compact human-readable summary of diagnostic reports."""
    if not reports:
        return "No diagnostics collected."
    lines = []
    for r in reports:
        status = "OK" if r.get("ok") else "FAIL"
        lines.append(
            f"[{status}] {r.get('label', '?')}: "
            f"type={r.get('actual_type')} "
            f"missing={r.get('missing_fields', [])} "
            f"extra={r.get('extra_fields', [])}"
        )
    return "\n".join(lines)


__all__ = [
    "diagnose_object",
    "check_contract",
    "validate_sequence",
    "summarize",
    "diagnose_stage",
]

def diagnose_stage(
    function_name: str,
    expected_type: type,
    actual: Any,
    expected_fields: Optional[Sequence[str]] = None,
    allow_extra: bool = True,
) -> Dict[str, Any]:
    """
    Inspect a pipeline stage and report type and contract information.

    Args:
        function_name: Name of the function/stage being inspected
        expected_type: The expected Python type of the actual parameter
        actual: The actual object received
        expected_fields: Optional field names expected for dicts/dataclasses
        allow_extra: If True, extra fields are not treated as violations.

    Returns:
        Dict with keys: function_name, expected_type, actual_type,
        field_names, missing_fields, extra_fields, ok
    """
    actual_type = _type_name(actual)
    field_names = _field_names(actual)
    expected = list(expected_fields) if expected_fields else []

    missing = [f for f in expected if f not in field_names] if expected else []
    extra = [f for f in field_names if f not in expected] if expected else []

    ok = not missing and (allow_extra or not extra)

    report = {
        "function_name": function_name,
        "expected_type": _type_name(expected_type) if expected_type else None,
        "actual_type": actual_type,
        "field_names": field_names,
        "missing_fields": missing,
        "extra_fields": extra,
        "ok": ok,
    }

    if not ok:
        logger.warning(
            "Stage violation at %s: expected_type=%s actual_type=%s missing=%s extra=%s",
            function_name,
            report["expected_type"],
            report["actual_type"],
            missing,
            extra,
        )

    return report
