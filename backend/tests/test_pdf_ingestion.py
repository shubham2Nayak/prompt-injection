"""
Regression tests for PDF ingestion pipeline.

Tests PDF upload workflow from parser through vector store.
Verifies compatibility between refactored modules.
"""

import sys
import unittest
from pathlib import Path

from conftest import get_sample_pdf_path
from rag.parser import PDFParser
from rag.chunking import TextChunker
from rag.adapters import normalize_chunk_output
from rag.embeddings import EmbeddingGenerator
from rag.chunking import TextChunker as TC
from debug.contracts import diagnose_stage, diagnose_object, check_contract
from state import RAGMetadata, WorkflowState
from context.schema import ContextAnnotation


class TestPDFIngestion(unittest.TestCase):
    """Test PDF ingestion pipeline end-to-end."""

    def setUp(self):
        self.parser = PDFParser(max_file_size_mb=50)
        self.chunker = TextChunker(chunk_size=700, chunk_overlap=100)
        self.embedding_gen = EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2", device="cpu"
        )
        # The suite referenced a "sample1.pdf" that was never committed, so every
        # case below aborted in the parser before asserting anything. A real PDF
        # is generated once and reused.
        self.pdf_path = get_sample_pdf_path()
        self.pdf_name = Path(self.pdf_path).name

    def test_safe_pdf_upload_successfully(self):
        """Test 1: Safe PDF uploads successfully."""
        from graph import PromptTraceWorkflow

        # Just verify the workflow can be initialized and parser works
        wf = PromptTraceWorkflow()
        self.assertIsNotNone(wf.pdf_parser)
        self.assertIsNotNone(wf.text_chunker)
        self.assertIsNotNone(wf.embedding_generator)
        self.assertIsNotNone(wf.vector_store)

        # Parse a PDF
        text, page_contents = self.parser.load_pdf(self.pdf_path)
        self.assertIsInstance(text, str)
        self.assertIsInstance(page_contents, list)
        self.assertGreater(len(page_contents), 0)

    def test_metadata_rich_parser_output_accepted(self):
        """Test 2: Metadata-rich parser output is accepted."""
        text, page_contents = self.parser.load_pdf(self.pdf_path)

        # Chunk the pages - handles both 2-tuple and 3-tuple shapes
        chunks = self.chunker.batch_chunk_pages(page_contents, "sample1")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

        # Normalize chunk output - should return dicts
        normalized = TC.normalize_chunk_output(chunks)
        self.assertIsInstance(normalized, list)
        self.assertGreater(len(normalized), 0)

        # Verify first chunk is a dict with expected keys
        first = normalized[0]
        self.assertIsInstance(first, dict)
        expected_keys = {
            "text",
            "chunk_id",
            "page_number",
            "source_document",
            "paragraph_offset",
            "character_offset",
            "retrieval_score",
        }
        self.assertTrue(expected_keys.issubset(set(first.keys())))

    def test_chunk_metadata_survives_into_faiss(self):
        """Test 3: Chunk metadata survives into FAISS."""
        text, page_contents = self.parser.load_pdf(self.pdf_path)
        chunks = self.chunker.batch_chunk_pages(page_contents, "sample1")
        normalized = TC.normalize_chunk_output(chunks)

        # Embed chunks
        embedded = self.embedding_gen.embed_chunks(normalized)
        self.assertEqual(len(embedded), len(normalized))

        # Verify embeddings were added.
        # Embeddings come back as torch tensors so the vector store can consume
        # them on-device without a NumPy round trip; they are no longer plain
        # Python lists.
        import numpy as np
        import torch

        for chunk in embedded:
            self.assertIn("embedding", chunk)
            embedding = chunk["embedding"]
            self.assertIsInstance(embedding, (torch.Tensor, np.ndarray))
            self.assertGreater(len(embedding), 0)
            if isinstance(embedding, torch.Tensor):
                self.assertGreater(embedding.numel(), 0)

        # Add to vector store
        embedding_dim = self.embedding_gen.get_embedding_dimension()
        from rag.vector_store import FAISSVectorStore

        vector_store = FAISSVectorStore(embedding_dim=embedding_dim, metric="L2")
        vector_store.add_chunks(embedded)
        stats = vector_store.get_stats()
        self.assertGreater(stats["total_chunks"], 0)

    def test_context_annotations_attach_correctly(self):
        """Test 4: Context annotations attach correctly."""
        from context.context_engine import ContextEngine
        from config import config

        text, page_contents = self.parser.load_pdf(self.pdf_path)
        chunks = self.chunker.batch_chunk_pages(page_contents, "sample1")
        normalized = TC.normalize_chunk_output(chunks)
        embedded = self.embedding_gen.embed_chunks(normalized)

        # Create a context engine and annotate a chunk
        engine = ContextEngine(
            annotation_model_name=config.ANNOTATION_MODEL,
            annotation_device=config.ANNOTATION_DEVICE,
            cache_size=500,
        )

        # annotate_chunk expects text string, not dict
        chunk_dict = embedded[0] if embedded else {}
        text_content = chunk_dict.get("text", "") if isinstance(chunk_dict, dict) else str(chunk_dict)

        annotation = engine.annotate_chunk(
            chunk_id="test_chunk_1",
            text=text_content,
            paragraph_offset=0,
            character_offset=0,
        )

        self.assertIsInstance(annotation, ContextAnnotation)
        self.assertTrue(hasattr(annotation, "annotation_category"))
        self.assertTrue(hasattr(annotation, "confidence"))
        self.assertTrue(hasattr(annotation, "chunk_id"))

    def test_workflow_state_remains_valid(self):
        """Test 5: WorkflowState remains valid through pipeline."""
        # Create a valid workflow state
        state = WorkflowState(
            original_query="test query",
            uploaded_pdf_path=self.pdf_path,
            workflow_status="in_progress",
        )
        self.assertIsInstance(state, WorkflowState)
        self.assertEqual(state.workflow_status, "in_progress")
        self.assertEqual(state.original_query, "test query")

        # Add retrieved chunks
        chunks = [
            RAGMetadata(
                document_name="sample1",
                page_number=1,
                chunk_id="chunk_1",
                trust_level="untrusted",
                retrieved_text="test text",
                chunk_index=0,
                paragraph_offset=0,
                character_offset=0,
            )
        ]
        state.retrieved_chunks = chunks
        self.assertEqual(len(state.retrieved_chunks), 1)

        # Add chunk annotations
        state.chunk_annotations = [
            ContextAnnotation(
                chunk_id="chunk_1",
                paragraph_offset=0,
                character_offset=0,
                annotation_category="ACADEMIC_DESCRIPTION",
                confidence=0.95,
                spans=[],
                model_used="test_model",
                latency_ms=5.0,
            )
        ]
        self.assertEqual(len(state.chunk_annotations), 1)

    def test_no_unpacking_errors(self):
        """Test 6: No unpacking errors occur during PDF ingestion."""
        text, page_contents = self.parser.load_pdf(self.pdf_path)

        # This should not raise "too many values to unpack" error
        chunks = self.chunker.batch_chunk_pages(page_contents, "sample1")
        self.assertIsInstance(chunks, list)

        # Normalize should work without errors
        normalized = TC.normalize_chunk_output(chunks)
        self.assertIsInstance(normalized, list)

        # Embed should work without TypeError
        embedded = self.embedding_gen.embed_chunks(normalized)
        self.assertIsInstance(embedded, list)

        # Vector store add should work
        embedding_dim = self.embedding_gen.get_embedding_dimension()
        from rag.vector_store import FAISSVectorStore

        vector_store = FAISSVectorStore(embedding_dim=embedding_dim, metric="L2")
        vector_store.add_chunks(embedded)
        stats = vector_store.get_stats()
        self.assertGreaterEqual(stats["total_chunks"], len(embedded))


if __name__ == "__main__":
    unittest.main()