"""Model wrapper for semantic annotation using Nemotron 3.5 Content Safety.

Responsibilities:
* Load annotation model once via shared loader.
* Expose classify_chunk().
* Expose classify_agent_output().
* Return structured JSON only.

Do not perform keyword matching.
Do not modify text.
"""

import logging
import time
import torch
from typing import Any, Dict, Optional

from config import config
from models import get_annotation_model, get_model_pool, validate_annotation_output

from .taxonomy import AnnotationCategory, UNKNOWN
from .schema import ContextAnnotation
from .validator import validate_annotation_json

logger = logging.getLogger(__name__)

# Resolved once: the previous code rebuilt this list for every annotation.
_VALID_CATEGORY_NAMES = frozenset(cat.name for cat in AnnotationCategory)


class AnnotationClassifier:
    """Wrapper for Nemotron 3.5 Content Safety model that classifies document chunks."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 64,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the annotation classifier using shared Nemotron model.

        Args:
            batch_size: Prompts per ``generate()`` call. Defaults to
                ``config.ANNOTATION_BATCH_SIZE``; the pipeline's own default of 1
                serializes multi-prompt calls.
        """
        self.model_name = model_name or config.ANNOTATION_MODEL
        self.device = device or config.ANNOTATION_DEVICE
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, int(batch_size or config.ANNOTATION_BATCH_SIZE))
        self._model, self._tokenizer, self.pipeline = get_annotation_model()
        self._pool = get_model_pool()

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply tokenizer's chat template if available, otherwise return as-is."""
        try:
            # Try to use the tokenizer's chat template
            messages = [{"role": "user", "content": prompt}]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: return prompt as-is
            return prompt

    def _create_annotation_prompt(self, text: str) -> str:
        """Create a prompt for the annotation model."""
        return f"""You are a document annotation specialist. Analyze the text and output ONLY valid JSON.

Text to annotate:
---
{text}
---

Output ONLY this JSON format (no other text, no markdown, no explanations):
{{
  "category": "ACADEMIC_DESCRIPTION",
  "confidence": 0.95,
  "reason": "Text describes academic concepts",
  "annotation_category": "ACADEMIC_DESCRIPTION",
  "paragraph_offset": 0,
  "character_offset": 0,
  "spans": []
}}

Valid categories: ACADEMIC_DESCRIPTION, QUOTED_EXAMPLE, EMBEDDED_PROCESSING_NOTE, MIXED_CONTENT, UNKNOWN

JSON response:"""

    def _create_agent_output_prompt(self, agent_name: str, output_text: str) -> str:
        """Create a prompt for classifying agent output."""
        return f"""You are classifying agent output for prompt injection detection. Output ONLY valid JSON.

Agent: {agent_name}

Output to classify:
---
{output_text}
---

Output ONLY this JSON format (no other text, no markdown, no explanations):
{{
  "category": "UNKNOWN",
  "confidence": 0.5,
  "reason": "No injection detected",
  "annotation_category": "UNKNOWN",
  "infected": false
}}

Valid categories: ACADEMIC_DESCRIPTION, QUOTED_EXAMPLE, EMBEDDED_PROCESSING_NOTE, MIXED_CONTENT, UNKNOWN

JSON response:"""

    def classify_chunk(self, text: str) -> ContextAnnotation:
        """
        Classify a document chunk and return structured annotation.

        Args:
            text: The chunk text to annotate

        Returns:
            ContextAnnotation with category, confidence, and metadata
        """
        start_ms = time.time()

        try:
            # Truncate input text to avoid context window issues
            safe_text = text[:1200].strip()
            prompt = self._create_annotation_prompt(safe_text)

            # Generate annotation with tracking
            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name="ContextEngine.chunk",
                input_text=prompt,
            ) as tracker:
                # Apply chat template for proper model prompting
                formatted_prompt = self._apply_chat_template(prompt)
                with torch.inference_mode():
                    outputs = self.pipeline(
                        formatted_prompt,
                        batch_size=self.batch_size,
                        max_new_tokens=self.max_new_tokens,
                        truncation=True,
                        temperature=self.temperature,
                        do_sample=False,
                        top_p=None,
                        top_k=None,
                    )

                generated_text = outputs[0]["generated_text"]
                tracker["output_text"] = generated_text

            # Use centralized validator with zero-retries (handles markdown/filler extraction)
            annotation = validate_annotation_json(generated_text)

            # Raw model output used to be dumped to stdout with a bare print()
            # on every annotation, which blocked on the console and cost more than
            # the classification itself. It is available at DEBUG level instead.
            logger.debug(
                "Raw Nemotron output (first 500 chars): %s", generated_text[:500]
            )

            # Ensure category is valid
            if annotation["category"] not in _VALID_CATEGORY_NAMES:
                annotation["category"] = UNKNOWN

            latency_ms = round((time.time() - start_ms) * 1000.0, 3)

            return ContextAnnotation(
                chunk_id=annotation.get("chunk_id", ""),
                paragraph_offset=annotation.get("paragraph_offset", 0),
                character_offset=annotation.get("character_offset", 0),
                annotation_category=annotation["category"],
                confidence=annotation["confidence"],
                spans=[],  # Will be populated by caller if needed
                model_used=self.model_name,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Chunk classification error: {str(e)}")
            return ContextAnnotation(
                chunk_id="",
                paragraph_offset=0,
                character_offset=0,
                annotation_category=UNKNOWN,
                confidence=0.0,
                spans=[],
                model_used=self.model_name,
                latency_ms=0.0,
            )

    def classify_agent_output(
        self, agent_name: str, output_text: str
    ) -> ContextAnnotation:
        """
        Classify an agent's output text.

        Args:
            agent_name: Name of the agent that produced the output
            output_text: The agent's output text

        Returns:
            ContextAnnotation with category and confidence
        """
        start_ms = time.time()

        try:
            # Truncate input text to avoid context window issues
            safe_text = output_text[:1200].strip()
            prompt = self._create_agent_output_prompt(agent_name, safe_text)

            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name=f"ContextEngine.{agent_name}",
                input_text=prompt,
            ) as tracker:
                # Apply chat template for proper model prompting
                formatted_prompt = self._apply_chat_template(prompt)
                with torch.inference_mode():
                    outputs = self.pipeline(
                        formatted_prompt,
                        batch_size=self.batch_size,
                        max_new_tokens=self.max_new_tokens,
                        truncation=True,
                        temperature=self.temperature,
                        do_sample=False,
                        top_p=None,
                        top_k=None,
                    )

                generated_text = outputs[0]["generated_text"]
                tracker["output_text"] = generated_text

            # Use centralized validator with zero-retries (handles markdown/filler extraction)
            annotation = validate_annotation_json(generated_text)

            # Bare print() in the classification path replaced with lazy logging.
            logger.debug(
                "Raw Nemotron output (first 500 chars): %s", generated_text[:500]
            )
            logger.debug("Validated annotation: %s", annotation)

            if annotation["category"] not in _VALID_CATEGORY_NAMES:
                annotation["category"] = UNKNOWN

            latency_ms = round((time.time() - start_ms) * 1000.0, 3)

            return ContextAnnotation(
                chunk_id=annotation.get("chunk_id", ""),
                paragraph_offset=annotation.get("paragraph_offset", 0),
                character_offset=annotation.get("character_offset", 0),
                annotation_category=annotation["category"],
                confidence=annotation["confidence"],
                spans=[],
                model_used=self.model_name,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.error(f"Agent output classification error: {str(e)}")
            return ContextAnnotation(
                chunk_id="",
                paragraph_offset=0,
                character_offset=0,
                annotation_category=UNKNOWN,
                confidence=0.0,
                spans=[],
                model_used=self.model_name,
                latency_ms=0.0,
            )