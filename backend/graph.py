"""
LangGraph Workflow for PromptTrace
Orchestrates the multi-agent propagation simulation.
Optimized for GPU-native tensor flow with zero CPU-GPU sync points.
"""

import logging
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

from langgraph.graph import StateGraph, END

from config import config
from state import WorkflowState
from rag import PDFParser, TextChunker, EmbeddingGenerator, FAISSVectorStore
from rag.adapters import normalize_vector_output
from agents import ResearchAgent, AnalystAgent, ActionAgent, PropagationMonitor
from context import ContextEngine, ContextAnnotation
from models import get_gpu_plan, GPU_CPU
from tools import propagation_logger
from debug.contracts import check_contract

logger = logging.getLogger(__name__)

# Contract expected of every ContextAnnotation leaving the retrieval node. The
# field list is fixed for the process, so it is resolved once rather than
# rebuilt and re-introspected on every simulation.
_ANNOTATION_CONTRACT_FIELDS = (
    "chunk_id", "paragraph_offset", "character_offset",
    "annotation_category", "confidence", "spans",
    "model_used", "latency_ms",
)
# Set built once from the schema; membership tests in the diagnostic are then
# O(1) instead of a linear scan per expected field.
_ANNOTATION_CONTRACT_SET = frozenset(_ANNOTATION_CONTRACT_FIELDS)


def _validate_annotation_sequence(annotations: list) -> None:
    """
    Check every annotation carries the retrieval contract's fields.

    Equivalent to ``validate_sequence("Retrieval -> Research", ...)`` but resolves
    the field names once against the model schema instead of once per record, and
    logs a single violation summary rather than allocating a report dict per
    annotation. Results are diagnostic only, so nothing is returned.
    """
    model_fields = getattr(type(annotations[0]), "model_fields", None)
    if isinstance(model_fields, dict):
        present = model_fields.keys()
    else:
        present = dir(annotations[0])

    missing = _ANNOTATION_CONTRACT_SET.difference(present)
    if missing:
        logger.warning(
            "Contract violation at Retrieval -> Research: "
            "annotation is missing fields %s",
            sorted(missing),
        )


class PromptTraceWorkflow:
    """Main workflow orchestrator for PromptTrace."""

    def __init__(self) -> None:
        """Initialize workflow components."""
        self.graph: Optional[Any] = None

        # RAG components
        self.pdf_parser: Optional[PDFParser] = None
        self.text_chunker: Optional[TextChunker] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.vector_store: Optional[FAISSVectorStore] = None

        # Agents
        self.research_agent: Optional[ResearchAgent] = None
        self.analyst_agent: Optional[AnalystAgent] = None
        self.action_agent: Optional[ActionAgent] = None
        self.propagation_monitor: Optional[PropagationMonitor] = None

        # Serializes assembly of a completed WorkflowState. A single workflow
        # instance is shared by every request, so two simulations finishing at
        # the same moment must not interleave their merges.
        self._merge_lock = threading.Lock()

        self._initialize_components()
        self._build_graph()

    def _initialize_components(self) -> None:
        """Initialize all workflow components with GPU acceleration."""
        logger.info("Initializing PromptTrace workflow components")

        try:
            self.pdf_parser = PDFParser(max_file_size_mb=config.MAX_PDF_SIZE_MB)
            self.text_chunker = TextChunker(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP,
            )

            # The GPU budget planner may have moved embeddings to CPU when the
            # two LLMs need the whole card.
            embedding_device = config.EMBEDDING_DEVICE
            if get_gpu_plan().mode_for("embedding") == GPU_CPU:
                if embedding_device.startswith("cuda"):
                    logger.warning(
                        "GPU budget placed the embedding model on CPU; "
                        "retrieval will run embeddings on the host"
                    )
                embedding_device = "cpu"

            # EmbeddingGenerator now uses GPU-native tensors by default
            self.embedding_generator = EmbeddingGenerator(
                model_name=config.EMBEDDING_MODEL,
                device=embedding_device,
                batch_size=config.EMBEDDING_BATCH_SIZE,
            )

            # Initialize vector store with GPU if embeddings are on GPU
            embedding_dim = self.embedding_generator.get_embedding_dimension()
            use_gpu = embedding_device == "cuda"
            # FAISS_REQUIRE_GPU (default true) turns an unavailable GPU index
            # into a startup error instead of a silent CPU fallback, so a host
            # that believes it is doing GPU retrieval cannot quietly stop doing
            # so. Set FAISS_REQUIRE_GPU=false for a CPU deployment.
            self.vector_store = FAISSVectorStore(
                embedding_dim=embedding_dim,
                metric=config.FAISS_METRIC,
                use_gpu=use_gpu,
                gpu_device=config.FAISS_GPU_DEVICE,
            )
            if use_gpu and not self.vector_store.use_gpu:
                logger.warning(
                    "Vector store fell back to a CPU index; retrieval will pay a "
                    "device-to-host copy per query"
                )

            # Initialize context annotation engine
            self.context_engine = ContextEngine(
                annotation_model_name=config.ANNOTATION_MODEL,
                annotation_device=config.ANNOTATION_DEVICE,
                cache_size=500,
            )

            # Initialize agents
            self.research_agent = ResearchAgent()
            self.analyst_agent = AnalystAgent()
            self.action_agent = ActionAgent()
            self.propagation_monitor = PropagationMonitor(context_engine=self.context_engine)

            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""
        logger.info("Building LangGraph workflow")

        workflow = StateGraph(WorkflowState)

        # Define nodes
        workflow.add_node("retrieval", self._node_retrieval)
        workflow.add_node("research", self._node_research)
        workflow.add_node("analyst", self._node_analyst)
        # Combined action + monitor node for parallel execution
        workflow.add_node("action_monitor", self._node_action_monitor)
        workflow.add_node("end", self._node_end)

        # Define edges
        workflow.set_entry_point("retrieval")
        workflow.add_edge("retrieval", "research")
        workflow.add_edge("research", "analyst")
        workflow.add_edge("analyst", "action_monitor")
        workflow.add_edge("action_monitor", "end")
        workflow.add_edge("end", END)

        self.graph = workflow.compile()
        logger.info("Workflow graph built successfully")

    def _node_retrieval(self, state: WorkflowState) -> WorkflowState:
        """RAG retrieval node - GPU-accelerated."""
        logger.info(f"[NODE] Retrieval: Processing query")
        state.agents_status["retrieval"] = "running"

        start_time = time.time()

        try:
            if not self.vector_store or self.vector_store.index.ntotal == 0:
                logger.warning("Vector store is empty")
                state.retrieved_chunks = []
                state.agents_status["retrieval"] = "completed"
                return state

            # Embed query - returns GPU tensor directly, no CPU copy
            query_embedding = self.embedding_generator.embed_text(
                state.original_query, return_tensor=True
            )

            # Search vector store - accepts GPU tensor directly
            raw_results = self.vector_store.search(
                query_embedding, top_k=config.RETRIEVAL_TOP_K
            )
            # Normalize results through the contract adapter
            results = normalize_vector_output(raw_results)
            # Validate the vector store -> retrieval contract
            if results:
                check_contract(
                    "VectorStore -> Retrieval",
                    expected_fields=[
                        "chunk_id", "page_number", "source_document",
                        "paragraph_offset", "character_offset",
                        "retrieval_score", "text",
                    ],
                    actual=results[0][0],
                )

            # Extract chunks with metadata
            from state import RAGMetadata
            retrieved_chunks = []
            for metadata, similarity in results:
                # Map new metadata fields to RAGMetadata (backward-compatible)
                doc_name = metadata.get("source_document", metadata.get("document_name", ""))
                chunk = RAGMetadata(
                    document_name=doc_name,
                    page_number=metadata.get("page_number", 0),
                    chunk_id=metadata.get("chunk_id", ""),
                    trust_level=metadata.get("trust_level", "untrusted"),
                    retrieved_text=metadata.get("text", metadata.get("retrieved_text", "")),
                    chunk_index=metadata.get("chunk_index", 0),
                    paragraph_offset=metadata.get("paragraph_offset", 0),
                    character_offset=metadata.get("character_offset", 0),
                )
                retrieved_chunks.append(chunk)

            # Enforce MAX_CHUNKS_PER_QUERY limit
            if len(retrieved_chunks) > config.MAX_CHUNKS_PER_QUERY:
                retrieved_chunks = retrieved_chunks[:config.MAX_CHUNKS_PER_QUERY]

            state.retrieved_chunks = retrieved_chunks

            # Annotate chunks using context engine (batched) - GPU-accelerated
            chunk_ids = [chunk.chunk_id for chunk in retrieved_chunks]
            texts = [chunk.retrieved_text[:1200] for chunk in retrieved_chunks]
            paragraph_offsets = [chunk.paragraph_offset for chunk in retrieved_chunks]
            character_offsets = [chunk.character_offset for chunk in retrieved_chunks]

            annotations = self.context_engine.annotate_chunks_batch(
                chunk_ids=chunk_ids,
                texts=texts,
                paragraph_offsets=paragraph_offsets,
                character_offsets=character_offsets,
            )

            annotated_chunks = list(zip(retrieved_chunks, annotations))

            # Attach a per-chunk annotation summary to each RAGMetadata
            for chunk, annotation in annotated_chunks:
                if annotation and annotation.annotation_category != "UNKNOWN":
                    chunk.annotation_summary = (
                        f"Category: {annotation.annotation_category}, "
                        f"Confidence: {annotation.confidence:.2f}"
                    )

            state.chunk_annotations = annotations
            # Validate the retrieval -> research contract. The field set is
            # identical for every annotation, so it is resolved once here instead
            # of per record inside validate_sequence().
            if state.chunk_annotations:
                _validate_annotation_sequence(state.chunk_annotations)

            retrieval_time = (time.time() - start_time) * 1000
            state.agents_status["retrieval"] = "completed"
            state.individual_agent_times["retrieval"] = retrieval_time

            logger.info(
                f"[NODE] Retrieval completed: {len(retrieved_chunks)} chunks "
                f"in {retrieval_time:.2f}ms"
            )
            propagation_logger.log_retrieval(
                state.start_timestamp.isoformat(),
                state.original_query,
                len(retrieved_chunks),
                retrieval_time,
            )

        except Exception as e:
            logger.error(f"Retrieval node error: {str(e)}")
            state.agents_status["retrieval"] = "failed"
            state.error_message = str(e)

        return state

    def _node_research(self, state: WorkflowState) -> WorkflowState:
        """Research agent node."""
        logger.info("[NODE] Research Agent")
        state.agents_status["research_agent"] = "running"

        start_time = time.time()
        try:
            state = self.research_agent.process(state)
            # Validate research -> analyst contract
            if state.research_output:
                check_contract(
                    "Research -> Analyst",
                    expected_fields=[
                        "summary", "key_findings", "sources_cited",
                        "raw_text", "processing_time_ms",
                    ],
                    # Pass the model itself. _field_names() reads model_fields
                    # directly, so model_dump() would recursively serialize
                    # raw_text plus a full retrieved_text per used chunk only
                    # for the result to be discarded unused.
                    actual=state.research_output,
                )
            propagation_logger.log_agent_end(
                state.start_timestamp.isoformat(),
                "Research Agent",
                state.individual_agent_times.get("research_agent", 0),
                len(state.research_output.summary) if state.research_output else 0,
            )
        except Exception as e:
            logger.error(f"Research node error: {str(e)}")
            state.agents_status["research_agent"] = "failed"
            state.error_message = str(e)

        return state

    def _node_analyst(self, state: WorkflowState) -> WorkflowState:
        """Analyst agent node."""
        logger.info("[NODE] Analyst Agent")
        state.agents_status["analyst_agent"] = "running"

        try:
            state = self.analyst_agent.process(state)
            # Validate analyst -> action contract
            if state.analyst_output:
                check_contract(
                    "Analyst -> Action",
                    expected_fields=[
                        "executive_summary", "key_findings", "recommendations",
                        "annotation_analysis", "confidence", "processing_time_ms",
                    ],
                    # Field-name check only; no deep serialization needed.
                    actual=state.analyst_output,
                )
            # Log using new AnalystAgentOutput fields
            if state.analyst_output:
                report_size = len(
                    state.analyst_output.executive_summary
                ) + sum(len(f) for f in state.analyst_output.key_findings)
                propagation_logger.log_agent_end(
                    state.start_timestamp.isoformat(),
                    "Analyst Agent",
                    state.individual_agent_times.get("analyst_agent", 0),
                    report_size,
                )
            else:
                propagation_logger.log_agent_end(
                    state.start_timestamp.isoformat(),
                    "Analyst Agent",
                    state.individual_agent_times.get("analyst_agent", 0),
                    0,
                )
        except Exception as e:
            logger.error(f"Analyst node error: {str(e)}")
            state.agents_status["analyst_agent"] = "failed"
            state.error_message = str(e)

        return state

    def _run_action_and_monitor(self, state: WorkflowState) -> WorkflowState:
        """
        Run Action Agent and Propagation Monitor concurrently, then serialize the
        one part of monitoring that genuinely depends on the Action Agent.

        The previous implementation handed the *same* ``WorkflowState`` instance
        to both workers. That was not merely a lost-update problem: the monitor
        reads ``state.action_output`` while the action agent is still assigning
        it, so whether the Action Agent appeared in the monitoring output at all
        depended on thread timing. Both also mutated ``agents_status`` and
        ``individual_agent_times`` concurrently, and the final "merge" was a
        self-assignment because both futures returned the same object.

        The fix is copy-on-write:

        * each worker receives its own deep copy, so no attribute is shared and
          no read can observe a half-written value;
        * the monitor pre-classifies only the Research and Analyst outputs, which
          exist before the Action Agent runs and are therefore genuinely
          independent work that overlaps with its generation;
        * the Action Agent's own output is classified afterwards, on the merged
          state, because it does not exist until that agent has produced it.

        Net effect: the two expensive model calls still overlap, the results are
        deterministic, and no lock is needed on the hot path. A merge lock is
        retained for the final assembly because a state may be completed by two
        different requests at once.
        """
        # Snapshot for the monitor's independent prefix. It only reads, so this
        # copy needs no action_output and is taken before the action agent runs.
        monitor_input = state.model_copy(deep=True)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="action-monitor") as pool:
            action_future = pool.submit(self.action_agent.process, state.model_copy(deep=True))
            monitor_future = pool.submit(
                self.propagation_monitor.preclassify_prefix, monitor_input
            )

            # The action agent writes to its own copy, so the original state is
            # untouched and safe to read here.
            action_state = action_future.result()
            precomputed = monitor_future.result()

        # Merge the action agent's results into the authoritative state. Only the
        # fields the action agent actually writes are copied: a blanket field
        # sweep would reassign identical values and could clobber the timing and
        # status dictionaries that were just merged.
        with self._merge_lock:
            if action_state.action_output is not None:
                state.action_output = action_state.action_output

            state.agents_status["action_agent"] = action_state.agents_status.get(
                "action_agent", state.agents_status.get("action_agent", "completed")
            )

            action_time = action_state.individual_agent_times.get("action_agent")
            if action_time is not None:
                state.individual_agent_times["action_agent"] = action_time

            if action_state.error_message:
                state.error_message = action_state.error_message

        # Dependent pass: the Action Agent output now exists, so classify it and
        # assemble the full timeline. Runs outside the lock because it is the
        # only consumer of the state at this point.
        state = self.propagation_monitor.monitor_workflow(state, precomputed=precomputed)

        # Log both
        propagation_logger.log_agent_end(
            state.start_timestamp.isoformat(),
            "Action Agent",
            state.individual_agent_times.get("action_agent", 0),
            len(state.action_output.action_summary) if state.action_output else 0,
        )

        for event in state.propagation_history:
            propagation_logger.log_propagation_event(
                {
                    "agent": event.agent,
                    "infected": event.infected,
                    "attack_detected": event.attack_detected,
                    "attack_phrase": event.attack_phrase,
                    "propagation_depth": event.propagation_depth,
                    "mutation_detected": event.mutation_detected,
                    "confidence": event.confidence,
                }
            )

        return state

    def _node_action_monitor(self, state: WorkflowState) -> WorkflowState:
        """Combined Action Agent + Propagation Monitor node (runs in parallel)."""
        logger.info("[NODE] Action Agent + Propagation Monitor (parallel)")
        state.agents_status["action_agent"] = "running"

        try:
            state = self._run_action_and_monitor(state)
            state.agents_status["action_agent"] = "completed"
        except Exception as e:
            logger.error(f"Action+Monitor node error: {str(e)}")
            state.agents_status["action_agent"] = "failed"
            state.error_message = str(e)

        return state

    def _node_end(self, state: WorkflowState) -> WorkflowState:
        """End node - finalize state."""
        state.end_timestamp = datetime.utcnow()
        state.workflow_status = "completed"
        state.total_processing_time_ms = sum(
            state.individual_agent_times.values()
        )

        logger.info(
            f"[NODE] Workflow completed in {state.total_processing_time_ms:.2f}ms"
        )
        propagation_logger.log_workflow_end(
            state.start_timestamp.isoformat(),
            "completed",
            state.total_processing_time_ms,
        )

        return state

    def run(self, query: str, pdf_path: Optional[str] = None) -> WorkflowState:
        """
        Run the complete workflow.

        Args:
            query: Input query/prompt
            pdf_path: Path to PDF file (if available)

        Returns:
            Completed workflow state
        """
        logger.info(f"Starting workflow with query: {query[:100]}")
        propagation_logger.log_workflow_start(
            str(time.time()), query
        )

        # Initialize state
        state = WorkflowState(
            original_query=query,
            uploaded_pdf_path=pdf_path,
            workflow_status="in_progress",
        )

        # Initialize agent statuses
        state.agents_status = {
            "retrieval": "pending",
            "research_agent": "pending",
            "analyst_agent": "pending",
            "action_agent": "pending",
        }

        try:
            # LangGraph can return a raw dict representation of the state after
            # invoking the compiled graph. Convert it back to the typed model so
            # downstream API and metrics logic can access attributes safely.
            final_state = self.graph.invoke(state.model_dump())
            if isinstance(final_state, dict):
                return WorkflowState.model_validate(final_state)
            if isinstance(final_state, WorkflowState):
                return final_state
            return WorkflowState.model_validate(final_state.model_dump())
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            state.workflow_status = "failed"
            state.error_message = str(e)
            state.end_timestamp = datetime.utcnow()
            return state

    def load_pdf_to_vector_store(self, pdf_path: str) -> dict:
        """
        Load PDF and build vector store (synchronous).

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with loading information
        """
        return self._load_pdf_to_vector_store_impl(pdf_path)

    async def load_pdf_to_vector_store_async(self, pdf_path: str) -> dict:
        """
        Load PDF and build vector store asynchronously.

        Uses thread pool for PDF parsing to avoid blocking the event loop.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with loading information
        """
        loop = asyncio.get_event_loop()
        # Run the synchronous implementation in a thread pool
        return await loop.run_in_executor(None, self._load_pdf_to_vector_store_impl, pdf_path)

    def _load_pdf_to_vector_store_impl(self, pdf_path: str) -> dict:
        """
        Internal implementation of PDF loading and vector store creation.
        GPU-accelerated embedding generation.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with loading information
        """
        logger.info(f"Loading PDF to vector store: {pdf_path}")

        try:
            # Parse PDF (returns (full_text, [(text, page, paragraph_offsets), ...]))
            # This runs in thread pool via load_pdf_to_vector_store_async
            full_text, page_contents = self.pdf_parser.load_pdf(pdf_path)

            # Get document name
            doc_name = Path(pdf_path).stem

            # Chunk text
            chunks = self.text_chunker.batch_chunk_pages(page_contents, doc_name)
            # Normalize through the contract adapter
            chunks = TextChunker.normalize_chunk_output(chunks)
            # Validate the chunker -> vector store contract before embedding
            if chunks:
                check_contract(
                    "Parser -> Chunker",
                    expected_fields=[
                        "text", "chunk_id", "page_number", "source_document",
                        "paragraph_offset", "character_offset", "retrieval_score",
                    ],
                    actual=chunks[0],
                )
            logger.info(f"Created {len(chunks)} chunks for embedding")

            # Embed chunks - GPU-native tensor flow, no CPU copies
            chunks = self.embedding_generator.embed_chunks(chunks)
            # Validate that the embedding stage preserved the canonical fields
            if chunks:
                check_contract(
                    "Embeddings -> VectorStore",
                    expected_fields=[
                        "text", "chunk_id", "page_number", "source_document",
                        "paragraph_offset", "character_offset", "retrieval_score",
                        "embedding",
                    ],
                    actual=chunks[0],
                )

            # Add to vector store - accepts GPU tensors directly
            self.vector_store.add_chunks(chunks)

            stats = self.vector_store.get_stats()
            logger.info(f"Vector store stats: {stats}")

            return {
                "status": "success",
                "document_name": doc_name,
                "chunks_created": len(chunks),
                "vector_store_stats": stats,
            }
        except Exception as e:
            logger.error(f"Failed to load PDF: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
            }