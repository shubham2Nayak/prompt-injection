"""
PDF Parser for PromptTrace RAG
Handles PDF parsing and text extraction with page and paragraph tracking.
Supports both synchronous and asynchronous (thread-pooled) operations.
"""

import logging
import asyncio
from typing import List, Tuple, Optional
from pathlib import Path
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs preserving boundaries."""
    # Split on double newlines, keeping the delimiter information
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


class PDFParser:
    """Parse PDF documents and extract text with page and paragraph metadata."""

    def __init__(self, max_file_size_mb: int = 50) -> None:
        """
        Initialize PDF parser.

        Args:
            max_file_size_mb: Maximum file size in megabytes
        """
        self.max_file_size_mb = max_file_size_mb

    def load_pdf(self, pdf_path: str) -> Tuple[str, List[Tuple[str, int, List[int]]]]:
        """
        Load and parse PDF file (synchronous).

        The page_contents list uses a 3-tuple shape
        ``(text, page_number, paragraph_offsets)`` so that paragraph-level
        character offsets are preserved for downstream chunk metadata. This
        is the canonical rich output of the refactored pipeline.

        Consumers that still expect the legacy 2-tuple shape
        ``(text, page_number)`` should call ``normalize_parser_output`` from
        ``backend.rag.adapters`` rather than unpacking directly.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (full_text, page_contents)
            where page_contents is list of (text, page_number, paragraph_offsets)
            paragraph_offsets gives character offset of each paragraph start

        Raises:
            FileNotFoundError: If PDF not found
            ValueError: If file too large or invalid
        """
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Check file size
        file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(
                f"PDF file too large: {file_size_mb:.2f}MB (max {self.max_file_size_mb}MB)"
            )

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF: {str(e)}")

        full_text_parts = []
        page_contents: List[Tuple[str, int, List[int]]] = []

        try:
            for page_num, page in enumerate(doc):
                try:
                    text = page.get_text()
                    paragraphs = _split_into_paragraphs(text)
                    # Compute character offsets for each paragraph
                    char_offsets = []
                    offset = 0
                    for p in paragraphs:
                        char_offsets.append(offset)
                        offset += len(p) + 2  # +2 for the \n\n separator
                    # Rejoin with \n\n for full text
                    joined = "\n\n".join(paragraphs)
                    full_text_parts.append(joined)
                    page_contents.append((text, page_num + 1, char_offsets))
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
        finally:
            doc.close()

        full_text = "\n".join(full_text_parts)
        logger.info(f"Parsed PDF: {pdf_file.name}, {len(page_contents)} pages")
        return full_text, page_contents

    async def load_pdf_async(self, pdf_path: str) -> Tuple[str, List[Tuple[str, int, List[int]]]]:
        """
        Load and parse PDF file asynchronously using thread pool.

        This offloads the blocking PyMuPDF operations to a thread pool,
        preventing event loop blocking during PDF ingestion.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (full_text, page_contents)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.load_pdf, pdf_path)

    def get_document_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with metadata
        """
        pdf_file = Path(pdf_path)

        try:
            doc = fitz.open(pdf_path)
            metadata = {
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "size_mb": pdf_file.stat().st_size / (1024 * 1024),
                "page_count": len(doc),
                "document_title": doc.metadata.get("title", "Unknown"),
                "document_author": doc.metadata.get("author", "Unknown"),
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Failed to extract metadata: {str(e)}")
            return {
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "error": str(e),
            }

    async def get_document_metadata_async(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF asynchronously.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_document_metadata, pdf_path)


class ChunkMetadata:
    """Metadata for a text chunk."""

    def __init__(
        self,
        chunk_id: str,
        page_number: int,
        source_document: str,
        paragraph_offset: int,
        character_offset: int,
        retrieval_score: float,
        text: str,
    ) -> None:
        self.chunk_id = chunk_id
        self.page_number = page_number
        self.source_document = source_document
        self.paragraph_offset = paragraph_offset
        self.character_offset = character_offset
        self.retrieval_score = retrieval_score
        self.text = text