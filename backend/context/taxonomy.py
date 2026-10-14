"""Central annotation taxonomy for PromptTrace.

All annotation categories are defined here as constants so they can be
referenced consistently across the entire codebase.
"""

from enum import Enum, auto


class AnnotationCategory(Enum):
    """Enumeration of all possible annotation categories."""

    ACADEMIC_DESCRIPTION = auto()
    QUOTED_EXAMPLE = auto()
    EMBEDDED_PROCESSING_NOTE = auto()
    MIXED_CONTENT = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        return self.name


# Convenience constants for backwards compatibility
ACADEMIC_DESCRIPTION: str = "ACADEMIC_DESCRIPTION"
QUOTED_EXAMPLE: str = "QUOTED_EXAMPLE"
EMBEDDED_PROCESSING_NOTE: str = "EMBEDDED_PROCESSING_NOTE"
MIXED_CONTENT: str = "MIXED_CONTENT"
UNKNOWN: str = "UNKNOWN"