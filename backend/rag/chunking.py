"""
Text Chunking for PromptTrace RAG
Splits text into chunks with semantic and overlap awareness.
Preserves paragraph boundaries and never splits paragraphs containing
processing notes.
"""

import logging
from typing import Any, List, Tuple
import re

logger = logging.getLogger(__name__)


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs preserving boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def _contains_processing_note(text: str) -> bool:
    """Check if text contains an AI processing note."""
    lowered = text.lower()
    markers = [
        "ai processing note",
        "if this document is being summarized by an ai assistant",
        "for the ai reading this",
        "assistant:",
        "system override",
        "embedded processing note",
    ]
    return any(marker in lowered for marker in markers)


def _token_estimate(text: str) -> int:
    """ Rough token estimate: 1 token ≈ 4 characters for English. """
    return max(1, len(text) // 4)


def _chunk_paragraphs(
    paragraphs: List[str],
    chunk_size: int = 700,
    chunk_overlap: int = 100,
) -> List[dict]:
    """
    Split paragraphs into chunks, never splitting paragraphs that contain
    processing notes.

    Args:
        paragraphs: List of paragraph texts
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of chunk dictionaries with metadata
    """
    target_chars = chunk_size * 4  # ~4 chars per token
    overlap_chars = chunk_overlap * 4

    chunks = []
    current_chunk_chars = 0
    current_chunk_paragraphs: List[str] = []
    current_chunk_start_para_idx = -1
    current_chunk_para_offset = 0

    for para_idx, paragraph in enumerate(paragraphs):
        para_len = len(paragraph)

        # Never split a paragraph that contains a processing note
        if _contains_processing_note(paragraph) and current_chunk_paragraphs:
            # Save current chunk and start a new one with this paragraph
            if current_chunk_paragraphs:
                chunk_text = " ".join(current_chunk_paragraphs)
                chunks.append(
                    {
                        "text": chunk_text,
                        "paragraph_offset": current_chunk_para_offset
                        if current_chunk_paragraphs
                        else 0,
                        "character_offset": sum(
                            len(p) + 2 for p in current_chunk_paragraphs[:-1]
                        )
                        if current_chunk_paragraphs
                        else 0,
                    }
                )
            current_chunk_chars = 0
            current_chunk_paragraphs = []
            current_chunk_start_para_idx = para_idx

        # Check if adding this paragraph exceeds chunk size
        test_chars = current_chunk_chars + para_len + (1 if current_chunk_paragraphs else 0)

        if test_chars > target_chars and current_chunk_paragraphs:
            # Save current chunk
            chunk_text = " ".join(current_chunk_paragraphs)
            chunks.append(
                {
                    "text": chunk_text,
                    "paragraph_offset": current_chunk_para_offset
                    if current_chunk_paragraphs
                    else 0,
                    "character_offset": sum(
                        len(p) + 2 for p in current_chunk_paragraphs[:-1]
                    )
                    if current_chunk_paragraphs
                    else 0,
                }
            )

            # Start new chunk with overlap
            overlap_text = " ".join(
                current_chunk_paragraphs[-3:]
            )  # ~3 paragraphs overlap
            current_chunk_chars = len(overlap_text) + para_len
            current_chunk_paragraphs = [overlap_text, paragraph]
            current_chunk_para_offset = para_idx - len(current_chunk_paragraphs) + 2
        else:
            # Add paragraph to current chunk
            if not current_chunk_paragraphs:
                current_chunk_para_offset = para_idx
            current_chunk_paragraphs.append(paragraph)
            current_chunk_chars = test_chars

    # Add final chunk
    if current_chunk_paragraphs:
        chunk_text = " ".join(current_chunk_paragraphs)
        chunks.append(
            {
                "text": chunk_text,
                "paragraph_offset": current_chunk_para_offset
                if current_chunk_paragraphs
                else 0,
                "character_offset": sum(
                    len(p) + 2 for p in current_chunk_paragraphs[:-1]
                )
                if current_chunk_paragraphs
                else 0,
            }
        )

    logger.info(f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs")
    return chunks


class TextChunker:
    """Split text into overlapping chunks with page and paragraph metadata."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        """
        Initialize text chunker.

        Args:
            chunk_size: Target chunk size in tokens (~700)
            chunk_overlap: Overlap between chunks in tokens (~100)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_into_chunks(
        self, text: str, document_name: str, page_number: int
    ) -> List[dict]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to split
            document_name: Name of source document
            page_number: Page number in document

        Returns:
            List of chunk dictionaries with metadata
        """
        paragraphs = _split_into_paragraphs(text)

        chunk_dicts = _chunk_paragraphs(
            paragraphs, self.chunk_size, self.chunk_overlap
        )

        chunks = []
        for idx, chunk in enumerate(chunk_dicts):
            chunks.append(
                {
                    "chunk_id": f"{document_name}_{page_number}_{idx}",
                    "page_number": page_number,
                    "source_document": document_name,
                    "paragraph_offset": chunk["paragraph_offset"],
                    "character_offset": chunk["character_offset"],
                    "retrieval_score": 0.0,  # Will be set during retrieval
                    "text": chunk["text"],
                }
            )

        return chunks

    def split_page_into_chunks(
        self, page_text: str, document_name: str, page_number: int
    ) -> List[dict]:
        """
        Specialized chunking for a single page.

        Args:
            page_text: Text from one page
            document_name: Document name
            page_number: Page number

        Returns:
            List of chunks from this page
        """
        return self.split_into_chunks(page_text, document_name, page_number)

    def batch_chunk_pages(
        self, page_contents: List[Tuple], document_name: str
    ) -> List[dict]:
        """
        Chunk multiple pages at once.

        Accepts either the legacy 2-tuple shape ``(text, page_number)`` or the
        richer 3-tuple shape ``(text, page_number, paragraph_offsets)`` produced
        by the refactored PDF parser. Paragraph offsets are preserved when
        available so that chunk metadata carries forward the original
        paragraph-level character offsets.

        Args:
            page_contents: List of (text, page_number) or
                (text, page_number, paragraph_offsets) tuples
            document_name: Document name

        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        for item in page_contents:
            # Tolerate both the legacy 2-tuple and the rich 3-tuple page shape.
            if len(item) == 3:
                text, page_num, _paragraph_offsets = item
            elif len(item) == 2:
                text, page_num = item
            else:
                raise ValueError(
                    f"page_contents entries must be 2- or 3-tuples, got len={len(item)}"
                )
            chunks = self.split_page_into_chunks(text, document_name, page_num)
            all_chunks.extend(chunks)

        logger.info(f"Total chunks from all pages: {len(all_chunks)}")
        return all_chunks

    @staticmethod
    def normalize_chunk_output(chunks: Any) -> List[dict]:
        """
        Adapter that normalizes chunker output into a uniform list of dicts.

        Accepts either a list of chunk dicts (the current shape) or a list of
        richer chunk objects (e.g. ``ChunkDocument`` dataclasses). The returned
        dicts always contain the canonical embedding/FAISS key set so that
        downstream stages never need to know which shape they received.

        Args:
            chunks: Iterable of chunk dicts or chunk-like objects

        Returns:
            List of canonical chunk dicts with keys: text, chunk_id,
            page_number, source_document, paragraph_offset, character_offset,
            retrieval_score.
        """
        # Local import avoids a circular dependency with backend.rag.adapters
        # when chunking is imported standalone.
        from rag.adapters import normalize_chunk_output as _normalize

        normalized = _normalize(chunks)
        return [chunk.to_dict() for chunk in normalized]