"""
Analyst Agent for PromptTrace
Creates structured analysis reports from research output and chunk annotations.
"""

import json
import logging
import time
import torch
from typing import Dict, Any, List

from config import config
from state import WorkflowState, AnalystAgentOutput
from context.schema import ContextAnnotation
from context.validator import safe_extract_json
from models import get_reasoning_model, get_model_pool, validate_reasoning_output
from prompts import prompts

logger = logging.getLogger(__name__)


class AnalystAgent:
    """Agent for structured analysis and report generation.

    Responsibilities:
    - Generate structured report from research summary and chunk annotations
    - Input: Research summary, Chunk annotations
    - Output JSON: executive_summary, key_findings, recommendations, annotation_analysis, confidence
    - No regex parsing
    - No keyword extraction
    - Output must be raw valid JSON only - no markdown, no preamble
    """

    def __init__(self) -> None:
        """Initialize analyst agent with shared reasoning model."""
        self.model_name = config.ANALYST_AGENT_MODEL
        self._model, self._tokenizer, self.pipeline = get_reasoning_model()
        self._pool = get_model_pool()

    def _create_chat_messages(self, research_summary: str, annotation_summary: str) -> List[dict]:
        """
        Create chat messages for analysis using system/user separation.

        Args:
            research_summary: Summary from research agent
            annotation_summary: Summary of chunk annotations

        Returns:
            List of chat messages for tokenizer.apply_chat_template
        """
        return prompts.build_chat_messages(
            system_template=prompts.ANALYST_SYSTEM,
            user_template=prompts.ANALYST_USER,
            research_summary=research_summary,
            annotation_summary=annotation_summary,
        )

    def _extract_annotation_analysis(
        self, chunk_annotations: List[ContextAnnotation]
    ) -> str:
        """Extract annotation analysis from chunk annotations.

        Args:
            chunk_annotations: List of ContextAnnotation objects

        Returns:
            String describing the annotation analysis
        """
        if not chunk_annotations:
            return "No chunk annotations available for analysis."

        # Count categories
        category_counts: Dict[str, int] = {}
        for ann in chunk_annotations:
            cat = ann.annotation_category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Build analysis
        analysis_parts: List[str] = []
        total = len(chunk_annotations)

        # Majority category
        if category_counts:
            majority_cat = max(category_counts, key=category_counts.get)
            majority_pct = category_counts[majority_cat] / total * 100
            analysis_parts.append(
                f"Majority of {majority_pct:.1f}% of chunks were annotated as '{majority_cat}'."
            )

        # Distribution
        for cat, count in category_counts.items():
            pct = count / total * 100
            analysis_parts.append(f"{count} chunks ({pct:.1f}%) fell into category '{cat}'.")

        # Low confidence chunks
        low_confidence = [a for a in chunk_annotations if a.confidence < 0.5]
        if low_confidence:
            analysis_parts.append(
                f"{len(low_confidence)} chunks had low confidence annotations (< 0.5)."
            )

        return " ".join(analysis_parts) if analysis_parts else "Annotations present but distribution unclear."

    def process(self, state: WorkflowState) -> WorkflowState:
        """
        Process research output and generate structured report.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with analyst output
        """
        start_time = time.time()

        if not state.research_output:
            logger.warning("No research output provided to Analyst Agent")
            state.analyst_output = AnalystAgentOutput(
                executive_summary="No research output available",
                key_findings=[],
                recommendations=[],
                annotation_analysis="No research output provided for analysis.",
                confidence=0.0,
                processing_time_ms=0,
            )
            return state

        try:
            logger.info("Analyst Agent processing research output")

            # Safely extract research summary regardless of schema variations
            # Use .get() for dict-like objects, fallback to attribute access, then stringify
            research_summary = ""
            if hasattr(state.research_output, 'summary'):
                research_summary = state.research_output.summary
            elif isinstance(state.research_output, dict):
                research_summary = state.research_output.get("summary", "")
            else:
                # Fallback: stringify the entire research_output object
                research_summary = str(state.research_output)
            
            if not research_summary or not research_summary.strip():
                logger.warning("Research output missing summary field")
                research_summary = "No summary available"

            # Create prompt with research summary and chunk annotations
            annotation_summary_data = ""
            if hasattr(state, 'chunk_annotations') and state.chunk_annotations:
                # Build a concise annotation summary
                category_counts: Dict[str, int] = {}
                for ann in state.chunk_annotations:
                    if ann and ann.annotation_category != "UNKNOWN":
                        category_counts[ann.annotation_category] = (
                            category_counts.get(ann.annotation_category, 0) + 1
                        )

                parts: List[str] = []
                total = len([a for a in state.chunk_annotations if a and a.annotation_category != "UNKNOWN"])
                if total > 0:
                    for cat, count in category_counts.items():
                        parts.append(f"{count} as '{cat}'")
                    annotation_summary_data = (
                        f"Chunk annotations: {', '.join(parts)} of {total} chunks. "
                    )

            # Create chat messages
            messages = self._create_chat_messages(
                research_summary=research_summary,
                annotation_summary=annotation_summary_data,
            )

            # Generate analysis with tracking
            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name="Analyst Agent",
                # See research_agent.process: str(messages) was built only to be
                # tokenized again for telemetry.
                input_text=None,
            ) as tracker:
                # Apply chat template
                formatted_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                with torch.inference_mode():
                    outputs = self.pipeline(
                        formatted_prompt,
                        max_new_tokens=config.MAX_NEW_TOKENS_ANALYST,
                        truncation=True,
                        temperature=None,
                        top_p=None,
                        do_sample=False,
                    )

                analysis_text = outputs[0]["generated_text"]
                # Extract only the generated portion after the formatted prompt
                if analysis_text.startswith(formatted_prompt):
                    analysis_text = analysis_text[len(formatted_prompt):].strip()
                tracker["output_text"] = analysis_text

            # Parse JSON using centralized safe_extract_json (zero-retries, handles markdown/filler)
            try:
                validated = safe_extract_json(analysis_text)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Analyst Agent JSON parse failed: {e}. Using fallback.")
                validated = {
                    "executive_summary": "Analysis parsing failed",
                    "key_findings": [],
                    "recommendations": [],
                    "annotation_analysis": "Failed to parse structured output",
                    "confidence": 0.0,
                }

            # Validate required fields exist and have correct types
            required_fields = ["executive_summary", "key_findings", "recommendations", "annotation_analysis", "confidence"]
            for field in required_fields:
                if field not in validated:
                    validated[field] = [] if field in ("key_findings", "recommendations") else ("" if field != "confidence" else 0.0)

            # Extract fields with safe defaults and type enforcement
            executive_summary = validated.get("executive_summary", "Analysis complete")
            if not isinstance(executive_summary, str) or not executive_summary:
                executive_summary = "Analysis complete"

            key_findings = validated.get("key_findings", [])
            if not isinstance(key_findings, list):
                key_findings = []

            recommendations = validated.get("recommendations", [])
            if not isinstance(recommendations, list):
                recommendations = []

            annotation_analysis = validated.get(
                "annotation_analysis",
                self._extract_annotation_analysis(
                    state.chunk_annotations if hasattr(state, 'chunk_annotations') else []
                )
            )
            if not isinstance(annotation_analysis, str) or not annotation_analysis:
                annotation_analysis = self._extract_annotation_analysis(
                    state.chunk_annotations if hasattr(state, 'chunk_annotations') else []
                )

            confidence = validated.get("confidence", 0.0)
            try:
                confidence = float(confidence)
                if confidence < 0.0:
                    confidence = 0.0
                if confidence > 1.0:
                    confidence = 1.0
            except (ValueError, TypeError):
                confidence = 0.0

            # Create output
            processing_time = (time.time() - start_time) * 1000
            state.analyst_output = AnalystAgentOutput(
                executive_summary=executive_summary,
                key_findings=key_findings,
                recommendations=recommendations,
                annotation_analysis=annotation_analysis,
                confidence=confidence,
                processing_time_ms=processing_time,
            )

            state.agents_status["analyst_agent"] = "completed"
            state.individual_agent_times["analyst_agent"] = processing_time

            logger.info(
                f"Analyst Agent completed in {processing_time:.2f}ms, "
                f"confidence={confidence:.2f}"
            )

        except Exception as e:
            logger.error(f"Analyst Agent error: {str(e)}")
            state.agents_status["analyst_agent"] = "failed"
            state.error_message = f"Analyst Agent failed: {str(e)}"
            state.analyst_output = AnalystAgentOutput(
                executive_summary="Analysis failed",
                key_findings=[],
                recommendations=[],
                annotation_analysis="",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        return state