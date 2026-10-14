"""Central Context Annotation Engine for PromptTrace.

This is the single component responsible for semantic annotation of
document chunks and agent outputs. All annotation flows through this
engine - no scattered keyword-based logic exists elsewhere.

The engine coordinates model inference, caching, and metadata attachment
while never modifying the original text content.
Optimized for GPU-native tensor flow with zero CPU-GPU sync points.
"""

import json
import logging
import os
import time
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any

from config import config
from jsonl_writer import enqueue_jsonl as _enqueue_jsonl
from models import get_model_pool, get_annotation_model
from prompts import prompts

from .taxonomy import AnnotationCategory, ACADEMIC_DESCRIPTION, QUOTED_EXAMPLE, UNKNOWN
from .cache import AnnotationCache
from .schema import ContextAnnotation, WorkflowAnnotation

logger = logging.getLogger(__name__)

# JSONL log file path
_ANNOTATION_LOG_DIR = os.path.join(config.LOGS_DIR, "annotations")
_ANNOTATION_LOG_FILE = os.path.join(_ANNOTATION_LOG_DIR, "annotations.jsonl")


def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp, cached so callers stop importing datetime."""
    return datetime.utcnow().isoformat()


class AnnotationClassifier:
    """Wrapper for annotation model (Nemotron) that classifies document chunks."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the classifier using the annotation model (Nemotron).

        Args:
            batch_size: Prompts per ``generate()`` call for the batched paths.
                Defaults to ``config.ANNOTATION_BATCH_SIZE``. The pipeline's own
                default is 1, which serializes the batch into N separate
                generations, so this must be forwarded deliberately.
        """
        self.model_name = model_name or config.ANNOTATION_MODEL
        self.device = device or config.ANNOTATION_DEVICE
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = max(1, int(batch_size or config.ANNOTATION_BATCH_SIZE))
        self._model, self._tokenizer, self.pipeline = get_annotation_model()
        self._pool = get_model_pool()

    # Taxonomy membership is fixed for the process; rebuilding the name list on
    # every annotation showed up in the batch loops as a per-item allocation.
    _VALID_CATEGORY_NAMES: Optional[frozenset] = None

    @classmethod
    def _valid_category_names(cls) -> frozenset:
        """Cached set of taxonomy names used to validate model output."""
        if cls._VALID_CATEGORY_NAMES is None:
            cls._VALID_CATEGORY_NAMES = frozenset(
                cat.name for cat in AnnotationCategory
            )
        return cls._VALID_CATEGORY_NAMES

    def _apply_chat_template(self, system_prompt: str, user_prompt: str) -> str:
        """Apply tokenizer's chat template with system/user separation."""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to simple concatenation if chat template not available
            return f"{system_prompt}\n\n{user_prompt}"

    def _generate(
        self,
        prompts: List[str],
    ) -> List[str]:
        """
        Run one batched pipeline call and return the generated text per prompt.

        ``batch_size`` is forwarded explicitly because ``TextGenerationPipeline``
        defaults it to 1. With the default, passing a list of prompts here would
        quietly issue one ``generate()`` call per prompt, so none of the GPU work
        overlaps and the "batch" gains nothing over a Python loop.

        Sampling parameters are only passed when temperature > 0 so the greedy
        path cannot be perturbed by a stray ``top_p``/``top_k``.
        """
        if self.temperature > 0.0:
            with torch.inference_mode():
                return self._generated_texts(
                    self.pipeline(
                        prompts,
                        batch_size=self.batch_size,
                        max_new_tokens=self.max_new_tokens,
                        truncation=True,
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.9,
                        top_k=20,
                    ),
                    len(prompts),
                )

        with torch.inference_mode():
            return self._generated_texts(
                self.pipeline(
                    prompts,
                    batch_size=self.batch_size,
                    max_new_tokens=self.max_new_tokens,
                    truncation=True,
                    temperature=None,
                    do_sample=False,
                    top_p=None,
                    top_k=None,
                ),
                len(prompts),
            )

    def _build_annotation(self, annotation: dict, latency_ms: float = 0.0) -> ContextAnnotation:
        """Normalize a validated model answer into a ContextAnnotation."""
        if annotation["category"] not in self._valid_category_names():
            annotation["category"] = UNKNOWN

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

    def _unknown_annotation(self) -> ContextAnnotation:
        """Uniform UNKNOWN annotation used for every failure path."""
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

    def _create_annotation_prompt(self, text: str) -> tuple[str, str]:
        """Create system and user prompts for chunk annotation."""
        system = prompts.CONTEXT_ANNOTATION_SYSTEM
        user = prompts.CONTEXT_ANNOTATION_USER.format(text=text)
        return system, user

    def _create_agent_output_prompt(self, agent_name: str, output_text: str) -> tuple[str, str]:
        """Create system and user prompts for agent output classification."""
        system = prompts.AGENT_OUTPUT_CLASSIFICATION_SYSTEM
        user = prompts.AGENT_OUTPUT_CLASSIFICATION_USER.format(
            agent_name=agent_name, output_text=output_text
        )
        return system, user

    def _extract_and_validate_json(self, generated_only: str) -> dict:
        """
        Validate the annotation model's answer.

        The Nemotron checkpoint answers in its own plain-text verdict format and
        a JSON annotation in others, so the validator handles both and returns
        the same {category, confidence, reason} shape either way.
        """
        from .validator import validate_annotation_output

        return validate_annotation_output(generated_only)

    @staticmethod
    def _generated_texts(outputs: Any, expected: int) -> List[str]:
        """
        Normalize TextGenerationPipeline output into a flat list of generated text.

        A generation pipeline returns a different shape depending on the input:
        a single prompt yields ``[{"generated_text": ...}]``, while a list of
        prompts yields a list of those - ``[[{"generated_text": ...}], ...]``.
        Indexing a batched result with a string key raises a TypeError, so both
        shapes are unwrapped here.
        """
        texts: List[str] = []
        for item in outputs:
            if isinstance(item, list):
                # One inner list per prompt; take the first generation.
                item = item[0] if item else {}
            if isinstance(item, dict):
                texts.append(item.get("generated_text", ""))
            else:
                texts.append(str(item))

        if len(texts) != expected:
            raise ValueError(
                f"Expected {expected} generations from the annotation model, "
                f"got {len(texts)}"
            )
        return texts

    def classify_chunk(self, text: str) -> ContextAnnotation:
        """Classify a document chunk and return structured annotation."""
        start_ms = time.time()

        try:
            safe_text = text[:1200].strip()
            system_prompt, user_prompt = self._create_annotation_prompt(safe_text)
            formatted_prompt = self._apply_chat_template(system_prompt, user_prompt)

            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name="ContextEngine.chunk",
                input_text=formatted_prompt,
            ) as tracker:
                generated_text = self._generate([formatted_prompt])[0]
                tracker["output_text"] = generated_text

            # Extract only the generated portion (after the formatted prompt)
            if generated_text.startswith(formatted_prompt):
                generated_only = generated_text[len(formatted_prompt):].strip()
            else:
                generated_only = generated_text

            # Use centralized validator with zero-retries
            annotation = self._extract_and_validate_json(generated_only)

            latency_ms = round((time.time() - start_ms) * 1000.0, 3)
            return self._build_annotation(annotation, latency_ms)

        except Exception as e:
            logger.error(f"Chunk classification error: {str(e)}")
            return self._unknown_annotation()

    def classify_agent_output(
        self, agent_name: str, output_text: str
    ) -> ContextAnnotation:
        """Classify an agent's output text."""
        start_ms = time.time()

        try:
            safe_text = output_text[:1200].strip()
            system_prompt, user_prompt = self._create_agent_output_prompt(
                agent_name, safe_text
            )
            formatted_prompt = self._apply_chat_template(system_prompt, user_prompt)

            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name=f"ContextEngine.{agent_name}",
                input_text=formatted_prompt,
            ) as tracker:
                generated_text = self._generate([formatted_prompt])[0]
                tracker["output_text"] = generated_text

            # Extract only the generated portion (after the formatted prompt)
            if generated_text.startswith(formatted_prompt):
                generated_only = generated_text[len(formatted_prompt):].strip()
            else:
                generated_only = generated_text

            # Use centralized validator with zero-retries
            annotation = self._extract_and_validate_json(generated_only)

            latency_ms = round((time.time() - start_ms) * 1000.0, 3)
            return self._build_annotation(annotation, latency_ms)

        except Exception as e:
            logger.error(f"Agent output classification error: {str(e)}")
            return self._unknown_annotation()

    def _annotations_from_generated(
        self,
        formatted_prompts: List[str],
        generated_texts: List[str],
        latency_ms: float,
    ) -> List[ContextAnnotation]:
        """
        Turn per-prompt generations into ContextAnnotations in input order.

        Shared by both batch paths so the prompt-echo stripping, validation and
        latency distribution stay identical. ``generated_texts`` is walked exactly
        once; the previous implementation re-derived it from the raw pipeline
        output a second time to drive this loop.
        """
        annotations: List[ContextAnnotation] = []
        for formatted_prompt, generated_text in zip(formatted_prompts, generated_texts):
            # Extract generated portion. The pipeline echoes the prompt because
            # return_full_text=False is not set on the annotation pipeline.
            if generated_text.startswith(formatted_prompt):
                generated_only = generated_text[len(formatted_prompt):].strip()
            else:
                generated_only = generated_text

            # Use centralized validator with zero-retries
            annotations.append(
                self._build_annotation(self._extract_and_validate_json(generated_only))
            )

        if annotations:
            # Distribute the batched latency across the annotations it produced.
            per_item_latency = latency_ms / len(annotations)
            for ann in annotations:
                ann.latency_ms = per_item_latency

        return annotations

    def classify_chunks_batch(self, texts: List[str]) -> List[ContextAnnotation]:
        """
        Classify multiple document chunks in a single batched pipeline call.

        The prompts are handed to the pipeline together with an explicit
        ``batch_size``, so the model performs one forward pass per batch rather
        than one ``generate()`` call per chunk.

        Args:
            texts: List of chunk text contents

        Returns:
            List of ContextAnnotation objects in same order as input
        """
        start_ms = time.time()

        if not texts:
            return []

        try:
            # Prepare and format every prompt up front so the batch is uniform.
            formatted_prompts = [
                self._apply_chat_template(
                    *self._create_annotation_prompt(text[:1200].strip())
                )
                for text in texts
            ]

            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name="ContextEngine.chunk.batch",
                input_text=None,
            ) as tracker:
                generated_texts = self._generate(formatted_prompts)
                tracker["output_text"] = generated_texts

            latency_ms = round((time.time() - start_ms) * 1000.0, 3)
            return self._annotations_from_generated(
                formatted_prompts, generated_texts, latency_ms
            )

        except Exception as e:
            logger.error(
                f"Batch chunk classification error: {str(e)}",
                exc_info=True,
            )
            # Fallback: return UNKNOWN for all
            return [self._unknown_annotation() for _ in texts]

    def classify_agent_outputs_batch(
        self, agent_names: List[str], output_texts: List[str]
    ) -> List[ContextAnnotation]:
        """
        Classify multiple agent outputs in a single batched pipeline call.

        The prompts are handed to the pipeline together with an explicit
        ``batch_size``, so the model performs one forward pass per batch rather
        than one ``generate()`` call per agent output.

        Args:
            agent_names: List of agent names
            output_texts: List of agent output text contents

        Returns:
            List of ContextAnnotation objects in same order as input
        """
        start_ms = time.time()

        if not agent_names or not output_texts:
            return []

        if len(agent_names) != len(output_texts):
            raise ValueError("agent_names and output_texts must have same length")

        try:
            formatted_prompts = [
                self._apply_chat_template(
                    *self._create_agent_output_prompt(name, text[:1200].strip())
                )
                for name, text in zip(agent_names, output_texts)
            ]

            with self._pool.track_invocation(
                model_name=self.model_name,
                agent_name="ContextEngine.agent_output.batch",
                input_text=None,
            ) as tracker:
                generated_texts = self._generate(formatted_prompts)
                tracker["output_text"] = generated_texts

            latency_ms = round((time.time() - start_ms) * 1000.0, 3)
            return self._annotations_from_generated(
                formatted_prompts, generated_texts, latency_ms
            )

        except Exception as e:
            logger.error(f"Batch agent output classification error: {str(e)}")
            # Fallback: return UNKNOWN for all
            return [self._unknown_annotation() for _ in agent_names]


class ContextEngine:
    """Central annotation engine for PromptTrace workflow."""

    def __init__(
        self,
        annotation_model_name: Optional[str] = None,
        annotation_device: Optional[str] = None,
        cache_size: int = 500,
    ) -> None:
        """
        Initialize the context annotation engine.

        Args:
            annotation_model_name: Name of the annotation model (uses config default if None)
            annotation_device: Device for model inference (uses config default if None)
            cache_size: Maximum cache entries
        """
        self.cache = AnnotationCache(max_size=cache_size)
        self.classifier = AnnotationClassifier(
            model_name=annotation_model_name or config.ANNOTATION_MODEL,
            device=annotation_device or config.ANNOTATION_DEVICE,
            temperature=config.TEMPERATURE_ANNOTATION,
            max_new_tokens=config.MAX_NEW_TOKENS_ANNOTATION,
            batch_size=config.ANNOTATION_BATCH_SIZE,
        )
        self.annotation_model_name = annotation_model_name or config.ANNOTATION_MODEL
        self.annotation_device = annotation_device or config.ANNOTATION_DEVICE
        self._pool = get_model_pool()
        self._annotation_latency_ms = 0.0
        self._total_annotations = 0
        # Ensure log directory exists
        os.makedirs(_ANNOTATION_LOG_DIR, exist_ok=True)

    def annotate_chunk(
        self,
        chunk_id: str,
        text: str,
        paragraph_offset: int = 0,
        character_offset: int = 0,
    ) -> ContextAnnotation:
        """
        Annotate a document chunk.

        The original text is never modified. Annotation metadata is computed
        and returned as a ContextAnnotation object.

        Args:
            chunk_id: Identifier of the chunk being annotated
            text: The chunk text content
            paragraph_offset: Paragraph offset within source document
            character_offset: Character offset within the paragraph

        Returns:
            ContextAnnotation with category, confidence, and metadata
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            # Update chunk_id and offsets in the cached result
            cached.chunk_id = chunk_id
            cached.paragraph_offset = paragraph_offset
            cached.character_offset = character_offset
            return cached

        # Perform model-based classification
        annotation = self.classifier.classify_chunk(text)

        # Update metadata
        annotation.chunk_id = chunk_id
        annotation.paragraph_offset = paragraph_offset
        annotation.character_offset = character_offset

        # Cache the result
        self.cache.set(text, annotation)

        # Track stats
        self._annotation_latency_ms += annotation.latency_ms
        self._total_annotations += 1

        # Log annotation event as JSONL
        self._log_annotation_event(
            agent="chunk",
            chunk_id=chunk_id,
            annotation_category=annotation.annotation_category,
            confidence=annotation.confidence,
            model_used=self.annotation_model_name,
            latency_ms=annotation.latency_ms,
            propagation_depth=0,
        )

        logger.debug(
            f"Annotated chunk {chunk_id}: category={annotation.annotation_category}, "
            f"confidence={annotation.confidence:.2f}"
        )

        return annotation

    def _log_annotation_event(
        self,
        agent: str,
        chunk_id: str,
        annotation_category: str,
        confidence: float,
        model_used: str,
        latency_ms: float,
        propagation_depth: int,
    ) -> None:
        """
        Queue an annotation event for the JSONL writer.

        The previous implementation opened, wrote and closed the log file once
        per annotation, inside the retrieval hot path. That is three syscalls per
        chunk plus a full flush, and it also meant the log could not be written
        without stalling inference. Events are now handed to a background writer
        thread so the caller only pays an in-memory enqueue.
        """
        event = {
            "agent": agent,
            "chunk_id": chunk_id,
            "annotation_category": annotation_category,
            "confidence": confidence,
            "model_used": model_used,
            "latency_ms": latency_ms,
            "propagation_depth": propagation_depth,
            "timestamp": _utcnow_iso(),
        }
        _enqueue_jsonl(_ANNOTATION_LOG_FILE, event)

    def annotate_agent_output(
        self,
        agent_name: str,
        text: str,
    ) -> ContextAnnotation:
        """
        Annotate an agent's output text.

        Args:
            agent_name: Name of the agent that produced the output
            text: The agent's output text

        Returns:
            ContextAnnotation with category and confidence
        """
        # Check cache using normalized text
        cached = self.cache.get(text)
        if cached is not None:
            cached.paragraph_offset = 0
            cached.character_offset = 0
            return cached

        annotation = self.classifier.classify_agent_output(agent_name, text)

        # Cache the result
        self.cache.set(text, annotation)

        self._annotation_latency_ms += annotation.latency_ms
        self._total_annotations += 1

        # Log annotation event as JSONL
        self._log_annotation_event(
            agent=agent_name,
            chunk_id="",
            annotation_category=annotation.annotation_category,
            confidence=annotation.confidence,
            model_used=self.annotation_model_name,
            latency_ms=annotation.latency_ms,
            propagation_depth=0,
        )

        logger.debug(
            f"Annotated agent output ({agent_name}): category={annotation.annotation_category}, "
            f"confidence={annotation.confidence:.2f}"
        )

        return annotation

    def annotate_chunks_batch(
        self,
        chunk_ids: List[str],
        texts: List[str],
        paragraph_offsets: Optional[List[int]] = None,
        character_offsets: Optional[List[int]] = None,
    ) -> List[ContextAnnotation]:
        """
        Annotate multiple document chunks in a single batched call.

        This replaces the sequential per-chunk loop with a single pipeline
        invocation for all chunks, significantly reducing GPU kernel launch
        overhead and total annotation latency.

        Args:
            chunk_ids: List of chunk identifiers
            texts: List of chunk text contents
            paragraph_offsets: Optional list of paragraph offsets
            character_offsets: Optional list of character offsets

        Returns:
            List of ContextAnnotation objects in same order as input
        """
        if not chunk_ids or not texts:
            return []

        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must have same length")

        if paragraph_offsets is None:
            paragraph_offsets = [0] * len(chunk_ids)
        if character_offsets is None:
            character_offsets = [0] * len(chunk_ids)

        # Partition into cache hits and misses in a single pass. The five
        # uncached_* lists are kept positionally aligned so the merge below is
        # a zip rather than a search.
        cached_results = {}
        uncached_texts: List[str] = []
        uncached_chunk_ids: List[str] = []
        uncached_para_offsets: List[int] = []
        uncached_char_offsets: List[int] = []

        for i, (chunk_id, text) in enumerate(zip(chunk_ids, texts)):
            cached = self.cache.get(text)
            if cached is not None:
                # Update metadata in cached result
                cached.chunk_id = chunk_id
                cached.paragraph_offset = paragraph_offsets[i]
                cached.character_offset = character_offsets[i]
                cached_results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_chunk_ids.append(chunk_id)
                uncached_para_offsets.append(paragraph_offsets[i])
                uncached_char_offsets.append(character_offsets[i])

        # Batch classify uncached texts
        new_annotations: List[ContextAnnotation] = []
        if uncached_texts:
            new_annotations = self.classifier.classify_chunks_batch(uncached_texts)

            # Update metadata, populate the cache and emit one log event per
            # annotation. enumerate(zip(...)) supplies the position directly;
            # the previous code called uncached_indices.index(idx) inside this
            # loop, which rescanned the list on every iteration and made the
            # whole method quadratic in the number of uncached chunks.
            for position, annotation in enumerate(new_annotations):
                annotation.chunk_id = uncached_chunk_ids[position]
                annotation.paragraph_offset = uncached_para_offsets[position]
                annotation.character_offset = uncached_char_offsets[position]

                # Cache the result
                self.cache.set(uncached_texts[position], annotation)

                # Track stats
                self._annotation_latency_ms += annotation.latency_ms
                self._total_annotations += 1

                # Log annotation event
                self._log_annotation_event(
                    agent="chunk",
                    chunk_id=annotation.chunk_id,
                    annotation_category=annotation.annotation_category,
                    confidence=annotation.confidence,
                    model_used=self.annotation_model_name,
                    latency_ms=annotation.latency_ms,
                    propagation_depth=0,
                )

        # Merge cached and new results back into the original input order. The
        # miss cursor advances only on a miss, so order is preserved without
        # mutating new_annotations.
        results: List[ContextAnnotation] = []
        new_cursor = 0
        for i in range(len(chunk_ids)):
            hit = cached_results.get(i)
            if hit is not None:
                results.append(hit)
            else:
                results.append(new_annotations[new_cursor])
                new_cursor += 1

        logger.debug(
            "Batch annotated %d chunks: %d cached, %d new",
            len(chunk_ids),
            len(cached_results),
            len(new_annotations),
        )

        return results

    def annotate_agent_outputs_batch(
        self,
        agent_names: List[str],
        texts: List[str],
    ) -> List[ContextAnnotation]:
        """
        Annotate multiple agent outputs in a single batched call.

        This replaces sequential per-agent calls with a single pipeline
        invocation for all agent outputs, significantly reducing GPU kernel
        launch overhead and total annotation latency.

        Args:
            agent_names: List of agent names
            texts: List of agent output text contents

        Returns:
            List of ContextAnnotation objects in same order as input
        """
        if not agent_names or not texts:
            return []

        if len(agent_names) != len(texts):
            raise ValueError("agent_names and texts must have same length")

        # Partition into cache hits and misses in a single pass, keeping the two
        # uncached lists positionally aligned.
        cached_results = {}
        uncached_texts: List[str] = []
        uncached_agent_names: List[str] = []

        for i, (agent_name, text) in enumerate(zip(agent_names, texts)):
            cached = self.cache.get(text)
            if cached is not None:
                cached.paragraph_offset = 0
                cached.character_offset = 0
                cached_results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_agent_names.append(agent_name)

        # Batch classify uncached texts
        new_annotations: List[ContextAnnotation] = []
        if uncached_texts:
            new_annotations = self.classifier.classify_agent_outputs_batch(
                uncached_agent_names, uncached_texts
            )

            # Populate the cache and emit one log event per annotation. The
            # position comes from enumerate directly; the previous code called
            # uncached_indices.index(idx) twice per iteration, making this loop
            # quadratic in the number of misses.
            for position, annotation in enumerate(new_annotations):
                # Cache the result
                self.cache.set(uncached_texts[position], annotation)

                # Track stats
                self._annotation_latency_ms += annotation.latency_ms
                self._total_annotations += 1

                # Log annotation event
                self._log_annotation_event(
                    agent=uncached_agent_names[position],
                    chunk_id="",
                    annotation_category=annotation.annotation_category,
                    confidence=annotation.confidence,
                    model_used=self.annotation_model_name,
                    latency_ms=annotation.latency_ms,
                    propagation_depth=0,
                )

        # Merge cached and new results back into the original input order.
        results: List[ContextAnnotation] = []
        new_cursor = 0
        for i in range(len(agent_names)):
            hit = cached_results.get(i)
            if hit is not None:
                results.append(hit)
            else:
                results.append(new_annotations[new_cursor])
                new_cursor += 1

        logger.debug(
            "Batch annotated %d agent outputs: %d cached, %d new",
            len(agent_names),
            len(cached_results),
            len(new_annotations),
        )

        return results

    def annotate_workflow_step(
        self,
        step_name: str,
        chunk_ids: List[str],
        texts: List[str],
    ) -> WorkflowAnnotation:
        """
        Annotate an entire workflow step.

        Args:
            step_name: Name of the workflow step
            chunk_ids: List of chunk IDs being processed
            texts: List of chunk text contents

        Returns:
            WorkflowAnnotation with overall category and summary
        """
        categories: Dict[str, int] = {}
        total_confidence = 0.0

        for chunk_id, text in zip(chunk_ids, texts):
            annotation = self.annotate_chunk(
                chunk_id=chunk_id,
                text=text,
                paragraph_offset=0,
                character_offset=0,
            )
            cat = annotation.annotation_category
            categories[cat] = categories.get(cat, 0) + 1
            total_confidence += annotation.confidence

        # Determine overall category (majority vote)
        if categories:
            overall_category = max(categories, key=categories.get)
            overall_confidence = total_confidence / len(chunk_ids) if chunk_ids else 0.0
        else:
            overall_category = UNKNOWN
            overall_confidence = 0.0

        annotation_summary = (
            f"Step {step_name}: {len(chunk_ids)} chunks annotated, "
            f"majority category: {overall_category}"
        )

        latency_ms = self._annotation_latency_ms / self._total_annotations if self._total_annotations > 0 else 0.0

        return WorkflowAnnotation(
            step=step_name,
            chunk_ids=chunk_ids,
            overall_category=overall_category,
            confidence=overall_confidence,
            annotation_summary=annotation_summary,
            latency_ms=latency_ms,
        )

    def get_stats(self) -> dict:
        """Return engine statistics."""
        cache_stats = self.cache.stats()
        return {
            "total_annotations": self._total_annotations,
            "average_latency_ms": (
                self._annotation_latency_ms / self._total_annotations
                if self._total_annotations > 0 else 0.0
            ),
            "cache_stats": cache_stats,
        }