"""Propagation monitor for PromptTrace.

Observes instruction-like content in agent messages using the centralized
Context Engine with Nemotron 3.5 Content Safety. No heuristic fallbacks.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from config import config
from state import WorkflowState, PropagationEvent, PropagationMonitorOutput
from context import ContextEngine
from models import get_model_pool

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Semantic classification for a single message."""

    model_config = ConfigDict(extra="allow")

    category: str = Field(..., description="Classification label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(..., description="Human-readable rationale")
    infected: bool = Field(default=False)


class PropagationHistoryEvent(BaseModel):
    """Single propagation annotation event."""

    source_agent: str
    destination_agent: Optional[str] = None
    classification: str
    confidence: float
    propagation_depth: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _normalize_text(value: Optional[str]) -> str:
    """Normalize text before semantic analysis."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def update_propagation_history(
    history: List[Dict[str, Any]],
    source_agent: Optional[str],
    destination_agent: Optional[str],
    classification: str,
    confidence: float,
    propagation_depth: int = 0,
    timestamp: Optional[str] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Store propagation annotation metadata without changing the message content."""
    event = {
        "source_agent": source_agent,
        "destination_agent": destination_agent,
        "classification": str(classification),
        "confidence": float(confidence),
        "propagation_depth": int(propagation_depth),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    event.update(kwargs)
    history.append(event)
    return history


class PropagationMonitor:
    """Propagation monitor using Context Engine with Nemotron 3.5 Content Safety."""

    def __init__(self, context_engine: Optional[ContextEngine] = None) -> None:
        self.context_engine = context_engine or ContextEngine()
        self._pool = get_model_pool()

    def analyze_agent_output(
        self,
        agent_name: str,
        current_output: str,
        previous_output: Optional[str] = None,
        propagation_depth: int = 0,
    ) -> PropagationMonitorOutput:
        """Analyze agent output for propagation indicators using Nemotron."""
        start_ms = time.perf_counter()

        # Classify using Context Engine (Nemotron)
        annotation = self.context_engine.annotate_agent_output(agent_name, current_output)

        # Extract instruction spans from annotation (if available)
        spans = []
        if annotation.spans:
            spans = [s.text for s in annotation.spans if s.category in {"EMBEDDED_PROCESSING_NOTE", "MIXED_CONTENT"}]

        # Determine infection status based on Nemotron classification
        infected = (
            annotation.confidence > config.PROPAGATION_CONFIDENCE_THRESHOLD
            and annotation.annotation_category in {"EMBEDDED_PROCESSING_NOTE", "MIXED_CONTENT"}
        )
        depth = propagation_depth + 1 if infected else 0

        # Compute mutation detection
        mutation_detected = False
        if infected and spans and previous_output:
            for span in spans:
                if span[:20].lower() not in previous_output.lower():
                    mutation_detected = True
                    break

        latency_ms = round((time.perf_counter() - start_ms) * 1000.0, 3)

        return PropagationMonitorOutput(
            infected=infected,
            agent=agent_name,
            attack_detected=infected,
            attack_phrase=spans[0] if spans else None,
            propagation_depth=depth,
            mutation_detected=mutation_detected,
            confidence=annotation.confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            analysis_details={
                "category": annotation.annotation_category,
                "reason": "Nemotron 3.5 Content Safety classification",
                "instruction_spans": spans,
                "latency_ms": latency_ms,
                "classification_source": "nemotron",
                "propagation_depth": depth,
                "persistence_score": round(annotation.confidence if infected else 0.0, 3),
                "mutation_detected": mutation_detected,
            },
        )

    def analyze_agent_outputs_batch(
        self,
        agent_names: List[str],
        outputs: List[str],
        previous_outputs: Optional[List[Optional[str]]] = None,
        propagation_depths: Optional[List[int]] = None,
    ) -> List[PropagationMonitorOutput]:
        """Analyze multiple agent outputs in a single batched call."""
        if not agent_names or not outputs:
            return []

        if len(agent_names) != len(outputs):
            raise ValueError("agent_names and outputs must have same length")

        if previous_outputs is None:
            previous_outputs = [None] * len(agent_names)
        if propagation_depths is None:
            propagation_depths = [0] * len(agent_names)

        start_ms = time.perf_counter()

        # Batch classify using Context Engine
        annotations = self.context_engine.annotate_agent_outputs_batch(agent_names, outputs)

        latency_ms = round((time.perf_counter() - start_ms) * 1000.0, 3)
        # Divide by the number of requested agents, not by len(annotations).
        # The requested count is known and stable, whereas the classifier's
        # return length is not guaranteed to be sized at all, and zip below
        # already truncates safely to whatever came back.
        per_item_latency = latency_ms / len(agent_names) if agent_names else 0.0

        results = []
        for i, (agent_name, output, annotation, prev_output, depth) in enumerate(
            zip(agent_names, outputs, annotations, previous_outputs, propagation_depths)
        ):
            # Extract instruction spans from annotation (if available)
            spans = []
            if annotation.spans:
                spans = [s.text for s in annotation.spans if s.category in {"EMBEDDED_PROCESSING_NOTE", "MIXED_CONTENT"}]

            # Determine infection status based on Nemotron classification
            infected = (
                annotation.confidence > config.PROPAGATION_CONFIDENCE_THRESHOLD
                and annotation.annotation_category in {"EMBEDDED_PROCESSING_NOTE", "MIXED_CONTENT"}
            )
            new_depth = depth + 1 if infected else 0

            # Compute mutation detection
            mutation_detected = False
            if infected and spans and prev_output:
                for span in spans:
                    if span[:20].lower() not in prev_output.lower():
                        mutation_detected = True
                        break

            results.append(PropagationMonitorOutput(
                infected=infected,
                agent=agent_name,
                attack_detected=infected,
                attack_phrase=spans[0] if spans else None,
                propagation_depth=new_depth,
                mutation_detected=mutation_detected,
                confidence=annotation.confidence,
                timestamp=datetime.now(timezone.utc).isoformat(),
                analysis_details={
                    "category": annotation.annotation_category,
                    "reason": "Nemotron 3.5 Content Safety classification",
                    "instruction_spans": spans,
                    "latency_ms": per_item_latency,
                    "classification_source": "nemotron",
                    "propagation_depth": new_depth,
                    "persistence_score": round(annotation.confidence if infected else 0.0, 3),
                    "mutation_detected": mutation_detected,
                },
            ))

        return results

    # The agent labels in pipeline order. The Action Agent is last because its
    # output does not exist until that agent has run, which is what makes the
    # Action/Monitor pairing below only partially parallelizable.
    _AGENT_OUTPUT_ACCESSORS = (
        ("Research Agent", "research_output", "summary"),
        ("Analyst Agent", "analyst_output", "executive_summary"),
        ("Action Agent", "action_output", "action_summary"),
    )

    def collect_agent_outputs(self, state: WorkflowState) -> List[tuple]:
        """
        Gather the (agent_name, output_text) pairs currently present on a state.

        Reads the state once, in pipeline order. Agents whose output has not been
        produced yet are simply absent from the result.

        Args:
            state: Workflow state to read

        Returns:
            List of (agent_name, output_text) tuples
        """
        collected: List[tuple] = []

        for label, output_field, text_field in self._AGENT_OUTPUT_ACCESSORS:
            output = getattr(state, output_field, None)
            if output is None:
                continue
            text = getattr(output, text_field, None)
            if text is None:
                continue
            collected.append((label, text))

        return collected

    def preclassify_prefix(
        self,
        state: WorkflowState,
        exclude: tuple = ("Action Agent",),
    ) -> Dict[str, PropagationMonitorOutput]:
        """
        Classify every available agent output except the named ones.

        This is the part of monitoring that has no dependency on the Action
        Agent, so it can run concurrently with it. The Action Agent's own output
        cannot be classified until that agent has produced it, which is precisely
        why the two were previously sharing one mutable state object: the monitor
        was reading ``state.action_output`` while the action agent was still
        writing it, so whether the Action Agent appeared in the monitoring output
        depended on thread timing.

        Exclusion is by agent name rather than by position. During the concurrent
        phase the Action Agent's output does not exist yet, so excluding "the
        last available output" would silently drop the Analyst Agent instead and
        leave work that could have overlapped.

        Args:
            state: Workflow state to read (treated as read-only)
            exclude: Agent labels to leave for the dependent pass

        Returns:
            Mapping of agent_name -> PropagationMonitorOutput for the classified prefix
        """
        excluded = set(exclude or ())

        agent_outputs = [
            (name, text)
            for name, text in self.collect_agent_outputs(state)
            if name not in excluded
        ]

        if not agent_outputs:
            return {}

        agent_names = [name for name, _ in agent_outputs]
        outputs = [text for _, text in agent_outputs]
        results = self.analyze_agent_outputs_batch(agent_names, outputs)

        return dict(zip(agent_names, results))

    def monitor_workflow(
        self,
        state: WorkflowState,
        precomputed: Optional[Dict[str, PropagationMonitorOutput]] = None,
    ) -> WorkflowState:
        """
        Monitor an entire workflow while preserving the existing state schema.

        Args:
            state: Workflow state to annotate in place
            precomputed: Results already produced by ``preclassify_prefix``,
                keyed by agent name. Only the agents actually present on the state
                and absent from this mapping are classified here, so passing the
                prefix from a concurrent pass avoids redoing that GPU work.

        Returns:
            The same state instance, with propagation_timeline and
            propagation_history populated.
        """
        precomputed = precomputed or {}

        agent_outputs = self.collect_agent_outputs(state)

        if not agent_outputs:
            logger.info("No agent outputs to monitor")
            return state

        # Classify whatever is not already available. Agents already classified
        # concurrently are reused rather than sent through the model again.
        pending = [(n, t) for n, t in agent_outputs if n not in precomputed]

        by_agent: Dict[str, PropagationMonitorOutput] = {}
        if pending:
            pending_names = [n for n, _ in pending]
            pending_texts = [t for _, t in pending]
            batch_results = self.analyze_agent_outputs_batch(pending_names, pending_texts)
            by_agent.update(zip(pending_names, batch_results))

        for agent_name, result in precomputed.items():
            if agent_name not in by_agent:
                by_agent[agent_name] = result

        # Assemble in pipeline order. An agent the classifier did not return a
        # verdict for is skipped rather than raising IndexError.
        monitor_outputs = [by_agent[name] for name, _ in agent_outputs if name in by_agent]

        if len(monitor_outputs) != len(agent_outputs):
            logger.warning(
                "Monitor returned %d result(s) for %d agent(s); missing verdicts: %s",
                len(monitor_outputs),
                len(agent_outputs),
                [n for n, _ in agent_outputs if n not in by_agent],
            )

        # Process results sequentially to maintain propagation depth logic
        previous_output = None
        propagation_depth = 0

        for i, (agent_name, output) in enumerate(agent_outputs):
            if agent_name not in by_agent:
                previous_output = output
                continue

            monitor_output = by_agent[agent_name]

            # Override propagation depth based on sequential logic
            if monitor_output.infected:
                monitor_output.propagation_depth = propagation_depth + 1
                propagation_depth = monitor_output.propagation_depth
            else:
                monitor_output.propagation_depth = 0

            # Recompute mutation detection with correct previous_output
            if monitor_output.infected:
                spans = monitor_output.analysis_details.get("instruction_spans", [])
                mutation_detected = False
                if spans and previous_output:
                    for span in spans:
                        if span[:20].lower() not in previous_output.lower():
                            mutation_detected = True
                            break
                monitor_output.mutation_detected = mutation_detected
                monitor_output.analysis_details["mutation_detected"] = mutation_detected
                monitor_output.analysis_details["propagation_depth"] = monitor_output.propagation_depth

            state.propagation_timeline.append(monitor_output)
            if monitor_output.infected:
                state.propagation_history.append(
                    PropagationEvent(
                        agent=monitor_output.agent,
                        infected=monitor_output.infected,
                        attack_detected=monitor_output.attack_detected,
                        attack_phrase=monitor_output.attack_phrase,
                        propagation_depth=monitor_output.propagation_depth,
                        mutation_detected=monitor_output.mutation_detected,
                        confidence=monitor_output.confidence,
                        category=monitor_output.analysis_details.get("category", "UNKNOWN"),
                    )
                )
            previous_output = output

        logger.info(
            "Propagation monitoring complete. Detected events: %s",
            len(state.propagation_history),
        )
        return state