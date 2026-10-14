"""
Research Agent for PromptTrace
Summarizes retrieved chunks faithfully, preserves annotation metadata,
and returns JSON-only output.
"""

import logging
import time
import torch
from typing import List, Optional

from config import config
from state import WorkflowState, ResearchAgentOutput, RAGMetadata
from context.schema import ContextAnnotation
from models import get_reasoning_model, get_model_pool, validate_reasoning_output
from prompts import prompts

logger = logging.getLogger(__name__)


class ResearchAgent:
    """Agent for faithful chunk summarization and analysis.

    Responsibilities:
    - Summarize retrieved chunks
    - Preserve annotation metadata (never performs annotation itself)
    - Return JSON-only output
    """

    def __init__(self) -> None:
        """Initialize research agent with shared reasoning model."""
        self.model_name = config.RESEARCH_AGENT_MODEL
        self._model, self._tokenizer, self.pipeline = get_reasoning_model()
        self._pool = get_model_pool()

    def _create_chat_messages(self, chunks: list[RAGMetadata]) -> List[dict]:
        """
        Create chat messages for summarization using system/user separation.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of chat messages for tokenizer.apply_chat_template
        """
        chunk_text = "\n---\n".join(
            f"[Chunk {i+1}] {chunk.retrieved_text}" for i, chunk in enumerate(chunks)
        )

        return prompts.build_chat_messages(
            system_template=prompts.RESEARCH_SYSTEM,
            user_template=prompts.RESEARCH_USER,
            chunk_text=chunk_text,
        )

    def _extract_annotation_summary(
        self, chunk_annotations: List[ContextAnnotation]
    ) -> dict:
        """Extract a summary of the chunk annotations.

        Args:
            chunk_annotations: List of ContextAnnotation objects

        Returns:
            Dictionary with annotation summary statistics
        """
        if not chunk_annotations:
            return {
                "annotation_summary": "No annotations available",
                "average_confidence": 0.0,
                "category_distribution": {},
            }

        category_counts: dict[str, int] = {}
        total_confidence = 0.0
        total_chunks = len(chunk_annotations)

        for ann in chunk_annotations:
            cat = ann.annotation_category
            category_counts[cat] = category_counts.get(cat, 0) + 1
            total_confidence += ann.confidence

        # Determine majority category
        if category_counts:
            majority_category = max(category_counts, key=category_counts.get)
            majority_percentage = category_counts[majority_category] / total_chunks * 100
        else:
            majority_category = "UNKNOWN"
            majority_percentage = 0.0

        return {
            "annotation_summary": (
                f"Annotated {total_chunks} chunks; "
                f"majority category: {majority_category} "
                f"({majority_percentage:.1f}%); "
                f"average confidence: {total_confidence / total_chunks:.2f}"
            ),
            "average_confidence": round(total_confidence / total_chunks, 3),
            "category_distribution": {
                cat: count / total_chunks for cat, count in category_counts.items()
            },
        }

    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Process retrieved chunks and generate summary.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with research output
        """
        start_time = time.time()

        if not state.retrieved_chunks:
            logger.warning("No retrieved chunks provided to Research Agent")
            state.research_output = ResearchAgentOutput(
                summary="No chunks retrieved",
                citations=[],
                used_chunks=[],
                annotation_summary="No chunks retrieved",
                confidence=0.0,
                processing_time_ms=0,
                raw_text="No chunks retrieved",
            )
            return state

        try:
            logger.info(f"Processing {len(state.retrieved_chunks)} chunks")

            # Create chat messages
            messages = self._create_chat_messages(state.retrieved_chunks)

            # Generate summary with tracking
            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name="Research Agent",
                # Pass the already-formatted prompt, not str(messages). Deep
                # rendering the message list flattened every retrieved chunk into
                # one large string purely so it could be tokenized a second time
                # for telemetry.
                input_text=None,
            ) as tracker:
                # Apply chat template
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                with torch.inference_mode():
                    outputs = self.pipeline(
                        formatted_prompt,
                        max_new_tokens=config.MAX_NEW_TOKENS_RESEARCH,
                        truncation=True,
                        temperature=config.TEMPERATURE_RESEARCH,
                        top_p=config.TOP_P,
                        do_sample=config.TEMPERATURE_RESEARCH > 0.0,
                    )

                generated = outputs[0]["generated_text"]
                # Extract only the generated portion after the formatted prompt
                if generated.startswith(formatted_prompt):
                    summary_text = generated[len(formatted_prompt):].strip()
                else:
                    summary_text = generated
                tracker["output_text"] = summary_text

            # Extract key findings (simple line-based extraction)
            key_findings = [
                line.strip()
                for line in summary_text.split("\n")
                if line.strip() and len(line.strip()) > 20
            ][:5]

            # Extract sources (chunk IDs used)
            sources_cited = [chunk.chunk_id for chunk in state.retrieved_chunks]

            # Build used_chunks with metadata
            used_chunks = []
            for chunk in state.retrieved_chunks:
                used_chunks.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "page_number": chunk.page_number,
                        "document_name": chunk.document_name,
                        "retrieved_text": chunk.retrieved_text,
                    }
                )

            # Extract annotation summary from chunk annotations
            annotation_summary_data = self._extract_annotation_summary(
                state.chunk_annotations if hasattr(state, 'chunk_annotations') else []
            )

            # Determine confidence from annotations
            if state.chunk_annotations and hasattr(state, 'chunk_annotations'):
                valid_annotations = [
                    a for a in state.chunk_annotations
                    if a and a.annotation_category != "UNKNOWN"
                ]
                if valid_annotations:
                    confidence = round(
                        sum(a.confidence for a in valid_annotations) / len(valid_annotations), 3
                    )
                else:
                    confidence = round(
                        sum(a.confidence for a in state.chunk_annotations) / len(state.chunk_annotations), 3
                    )
            else:
                confidence = 0.0

            # Create output
            processing_time = (time.time() - start_time) * 1000
            state.research_output = ResearchAgentOutput(
                summary=summary_text,
                key_findings=key_findings,
                sources_cited=sources_cited,
                raw_text=summary_text,  # No filtering - preserve all content
                processing_time_ms=processing_time,
                # New fields for the refactored pipeline
                citations=sources_cited,
                used_chunks=used_chunks,
                annotation_summary=annotation_summary_data["annotation_summary"],
                confidence=confidence,
            )

            state.agents_status["research_agent"] = "completed"
            state.individual_agent_times["research_agent"] = processing_time

            logger.info(
                f"Research Agent completed in {processing_time:.2f}ms, "
                f"output size: {len(summary_text)} chars"
            )

        except Exception as e:
            logger.error(f"Research Agent error: {str(e)}")
            state.agents_status["research_agent"] = "failed"
            state.error_message = f"Research Agent failed: {str(e)}"
            state.research_output = ResearchAgentOutput(
                summary="",
                citations=[],
                used_chunks=[],
                annotation_summary="",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                raw_text="",
            )

        return state