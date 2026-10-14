"""Context annotation module for PromptTrace.

This package provides the central semantic annotation engine for the
PromptTrace workflow, replacing scattered keyword-based logic with one
model-first annotation pipeline.
"""

from .context_engine import ContextEngine
from .classifier import AnnotationClassifier
from .schema import ContextAnnotation
from .cache import AnnotationCache
from .validator import validate_annotation_json, validate_context_annotation
from .taxonomy import AnnotationCategory, ACADEMIC_DESCRIPTION, QUOTED_EXAMPLE, UNKNOWN