"""Pydantic models for annotation data structures.

All annotation-related models are defined here so that validation is
centralized and consistent throughout the pipeline.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AnnotationSpan(BaseModel):
    """A span of text within a document that has been annotated."""

    start_char: int = Field(..., description="Starting character offset")
    end_char: int = Field(..., description="Ending character offset")
    text: str = Field(..., description="The annotated text segment")
    category: str = Field(..., description="Annotation category label")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence score"
    )

    @staticmethod
    def validate_spans(spans_data: List[Any]) -> List["AnnotationSpan"]:
        """Validate and convert spans data to AnnotationSpan instances."""
        valid_spans = []
        for s in spans_data:
            try:
                valid_spans.append(AnnotationSpan(
                    start_char=s.get("start_char", 0),
                    end_char=s.get("end_char", 0),
                    text=s.get("text", ""),
                    category=s.get("category", "UNKNOWN"),
                    confidence=float(s.get("confidence", 0.0)),
                ))
            except Exception:
                continue
        return valid_spans


class ContextAnnotation(BaseModel):
    """A complete annotation for a document chunk."""

    chunk_id: str = Field(..., description="Identifier of the chunk being annotated")
    paragraph_offset: int = Field(
        ..., description="Paragraph offset within the source document"
    )
    character_offset: int = Field(
        ..., description="Character offset within the paragraph"
    )
    annotation_category: str = Field(
        ..., description="High-level annotation category"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Model confidence score"
    )
    spans: List[AnnotationSpan] = Field(
        default_factory=list, description="Detailed text spans"
    )
    model_used: str = Field(..., description="Model that produced the annotation")
    latency_ms: float = Field(
        description="Annotation inference latency in ms", ge=0.0
    )


class WorkflowAnnotation(BaseModel):
    """Annotation attached to a workflow step output."""

    step: str = Field(..., description="Workflow step name")
    chunk_ids: List[str] = Field(
        default_factory=list, description="Chunk IDs affected"
    )
    overall_category: str = Field(
        ..., description="Overall annotation category for this step"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence score"
    )
    annotation_summary: str = Field(
        ..., description="Human-readable summary of annotations"
    )
    latency_ms: float = Field(
        description="Annotation processing latency in ms", ge=0.0
    )