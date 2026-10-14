"""
Data contract adapters for the PromptTrace RAG pipeline.

The architecture refactor introduced richer metadata objects (ChunkMetadata,
ContextAnnotation) while several downstream stages still consume the legacy
flat-dict / 2-tuple shapes. Rather than rewriting every consumer, this module
provides a single set of normalization helpers that translate between the old
and new representations.

Public API:
    normalize_parser_output(full_text, page_contents) -> (str, List[PageContent])
    normalize_chunk_output(chunks) -> List[ChunkDocument]
    normalize_chunk_dict(chunk) -> ChunkDocument
    normalize_vector_output(results) -> List[Tuple[dict, float]]

All adapters are idempotent: passing an already-normalized object through
returns it unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rich output dataclasses (the "new" schema)
# ---------------------------------------------------------------------------

@dataclass
class ParagraphOffsets:
    """Character offsets of paragraph starts within a page."""
    offsets: List[int] = field(default_factory=list)


@dataclass
class PageContent:
    """
    Normalized representation of a single parsed page.

    Attributes:
        text: Raw page text (as returned by PyMuPDF).
        page_number: 1-based page number.
        paragraph_offsets: Character offset of each paragraph start.
    """
    text: str
    page_number: int
    paragraph_offsets: List[int] = field(default_factory=list)

    @classmethod
    def from_tuple(cls, item: Tuple[Any, ...]) -> "PageContent":
        """Build from a (text, page_number, paragraph_offsets) tuple."""
        text = str(item[0]) if len(item) > 0 else ""
        page_number = int(item[1]) if len(item) > 1 else 0
        offsets_raw = item[2] if len(item) > 2 else []
        offsets = [int(o) for o in offsets_raw] if offsets_raw is not None else []
        return cls(text=text, page_number=page_number, paragraph_offsets=offsets)

    def to_tuple(self) -> Tuple[str, int, List[int]]:
        """Serialize back to the legacy 3-tuple shape."""
        return (self.text, self.page_number, self.paragraph_offsets)


@dataclass
class ChunkDocument:
    """
    Normalized representation of a single chunk.

    This is the canonical type that every downstream stage (embeddings, FAISS,
    context engine, retrieval) should consume.
    """
    text: str = ""
    chunk_id: str = ""
    page_number: int = 0
    source_document: str = ""
    paragraph_offset: int = 0
    character_offset: int = 0
    retrieval_score: float = 0.0
    # Optional richer metadata, populated when available.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (legacy embedding/FAISS shape)."""
        return {
            "text": self.text,
            "chunk_id": self.chunk_id,
            "page_number": self.page_number,
            "source_document": self.source_document,
            "paragraph_offset": self.paragraph_offset,
            "character_offset": self.character_offset,
            "retrieval_score": self.retrieval_score,
        }


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def normalize_parser_output(
    full_text: Any,
    page_contents: Any,
) -> Tuple[str, List[PageContent]]:
    """
    Normalize the parser output into (full_text, List[PageContent]).

    Accepts either the new 3-tuple-per-page shape or the legacy 2-tuple shape
    so that callers never break when the parser contract changes.

    Args:
        full_text: Concatenated document text.
        page_contents: Iterable of page records, each either a 3-tuple
            (text, page_number, paragraph_offsets) or a 2-tuple
            (text, page_number).

    Returns:
        Tuple of (full_text, List[PageContent]).

    Raises:
        TypeError: If a page record is not a tuple/list of length 2 or 3.
    """
    text = str(full_text) if full_text is not None else ""
    normalized: List[PageContent] = []
    for item in page_contents:
        if isinstance(item, PageContent):
            normalized.append(item)
        elif isinstance(item, (tuple, list)):
            if len(item) == 3:
                normalized.append(PageContent.from_tuple(item))
            elif len(item) == 2:
                normalized.append(PageContent(text=str(item[0]), page_number=int(item[1])))
            else:
                raise TypeError(
                    f"Parser page record must be a 2- or 3-tuple, got len={len(item)}"
                )
        else:
            raise TypeError(
                f"Unsupported parser page record type: {type(item).__name__}"
            )
    logger.debug(f"Normalized parser output: {len(normalized)} pages")
    return text, normalized


def normalize_chunk_dict(chunk: Any) -> ChunkDocument:
    """
    Convert a single chunk dict (or ChunkDocument) into a ChunkDocument.

    Args:
        chunk: A dict-like chunk with keys such as 'text', 'chunk_id',
            'page_number', 'source_document', 'paragraph_offset',
            'character_offset', 'retrieval_score'. May also be a
            ChunkDocument already.

    Returns:
        A ChunkDocument instance.
    """
    if isinstance(chunk, ChunkDocument):
        return chunk
    if not isinstance(chunk, dict):
        raise TypeError(
            f"normalize_chunk_dict expects dict or ChunkDocument, got {type(chunk).__name__}"
        )
    known = {
        "text", "chunk_id", "page_number", "source_document",
        "paragraph_offset", "character_offset", "retrieval_score",
    }
    metadata = {k: v for k, v in chunk.items() if k not in known}
    return ChunkDocument(
        text=str(chunk.get("text", "")),
        chunk_id=str(chunk.get("chunk_id", "")),
        page_number=int(chunk.get("page_number", 0)),
        source_document=str(chunk.get("source_document", "")),
        paragraph_offset=int(chunk.get("paragraph_offset", 0)),
        character_offset=int(chunk.get("character_offset", 0)),
        retrieval_score=float(chunk.get("retrieval_score", 0.0)),
        metadata=metadata,
    )


def normalize_chunk_output(chunks: Any) -> List[ChunkDocument]:
    """
    Normalize chunker output into a list of ChunkDocument objects.

    Accepts either the new dict-per-chunk shape or any already-normalized
    ChunkDocument list.

    Args:
        chunks: Iterable of chunk dicts or ChunkDocument objects.

    Returns:
        List of ChunkDocument instances.
    """
    if chunks is None:
        return []
    normalized: List[ChunkDocument] = []
    for chunk in chunks:
        normalized.append(normalize_chunk_dict(chunk))
    logger.debug(f"Normalized chunk output: {len(normalized)} chunks")
    return normalized


def normalize_vector_output(results: Any) -> List[Tuple[Dict[str, Any], float]]:
    """
    Normalize FAISS search results into (metadata_dict, similarity) tuples.

    Accepts either the legacy shape (already tuples) or a richer shape where
    each result is (ChunkDocument, float) or (dict, float).

    Args:
        results: Iterable of result pairs from FAISSVectorStore.search().

    Returns:
        List of (metadata_dict, float) tuples suitable for downstream
        consumers that index into the dict by key.
    """
    if results is None:
        return []
    normalized: List[Tuple[Dict[str, Any], float]] = []
    for result in results:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError(
                f"Vector result must be a 2-tuple (metadata, similarity), got {result!r}"
            )
        metadata, similarity = result
        if isinstance(metadata, ChunkDocument):
            meta_dict = metadata.to_dict()
        elif isinstance(metadata, dict):
            meta_dict = dict(metadata)
        else:
            # Last-resort: convert arbitrary object to dict via __dict__
            meta_dict = getattr(metadata, "__dict__", {"value": metadata})
        try:
            sim_value = float(similarity)
        except (TypeError, ValueError):
            sim_value = 0.0
        normalized.append((meta_dict, sim_value))
    logger.debug(f"Normalized vector output: {len(normalized)} results")
    return normalized


__all__ = [
    "ParagraphOffsets",
    "PageContent",
    "ChunkDocument",
    "normalize_parser_output",
    "normalize_chunk_dict",
    "normalize_chunk_output",
    "normalize_vector_output",
]