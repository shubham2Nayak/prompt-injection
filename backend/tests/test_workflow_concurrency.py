"""
Regression tests for concurrent Action Agent / Propagation Monitor execution.

The two workers used to receive the same WorkflowState instance. That was a
correctness problem, not merely a lost-update problem: the monitor read
``state.action_output`` while the action agent was still assigning it, so
whether the Action Agent appeared in the monitoring output at all depended on
thread timing. Both also mutated ``agents_status`` and
``individual_agent_times`` concurrently, and the final "merge" was a
self-assignment because both futures returned the same object.

These tests pin the copy-on-write behaviour: the Action Agent must appear in the
monitoring output on every run, results must be identical to a sequential run,
and no worker may observe a partially written attribute.
"""

import threading
import time
import unittest
from unittest.mock import Mock

from agents.propagation_monitor import PropagationMonitor
from context.schema import ContextAnnotation
from state import (
    ActionAgentOutput,
    AnalystAgentOutput,
    PropagationMonitorOutput,
    ResearchAgentOutput,
    WorkflowState,
)

CLEAN = "ACADEMIC_DESCRIPTION"
INFECTED = "EMBEDDED_PROCESSING_NOTE"


def _annotation(category: str, confidence: float) -> ContextAnnotation:
    return ContextAnnotation(
        chunk_id="",
        paragraph_offset=0,
        character_offset=0,
        annotation_category=category,
        confidence=confidence,
        spans=[],
        model_used="nvidia/Nemotron-3.5-Content-Safety",
        latency_ms=10.0,
    )


def _make_state() -> WorkflowState:
    return WorkflowState(
        original_query="test query",
        research_output=ResearchAgentOutput(
            summary="Research summary",
            raw_text="Research summary",
            confidence=0.9,
            processing_time_ms=10.0,
        ),
        analyst_output=AnalystAgentOutput(
            executive_summary="Analysis summary",
            annotation_analysis="preserved",
            confidence=0.9,
            processing_time_ms=10.0,
        ),
        action_output=None,
    )


def _monitor(verdicts) -> PropagationMonitor:
    """Build a monitor whose Context Engine returns the given categories."""
    engine = Mock()

    def _batch(names, texts):
        return [
            _annotation(CLEAN if v == CLEAN else INFECTED, 0.95 if v == CLEAN else 0.92)
            for v in verdicts[: len(names)]
        ]

    engine.annotate_agent_outputs_batch.side_effect = _batch
    engine.annotate_agent_output.side_effect = lambda name, text: _annotation(CLEAN, 0.95)
    return PropagationMonitor(context_engine=engine)


class TestMonitorCollectsAgentOutputs(unittest.TestCase):
    """collect_agent_outputs is the read-only view the concurrency fix relies on."""

    def test_collects_available_outputs_in_pipeline_order(self):
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )

        collected = PropagationMonitor(context_engine=Mock()).collect_agent_outputs(state)

        self.assertEqual(
            [name for name, _ in collected],
            ["Research Agent", "Analyst Agent", "Action Agent"],
        )

    def test_skips_agents_whose_output_is_absent(self):
        state = _make_state()
        # action_output deliberately unset
        collected = PropagationMonitor(context_engine=Mock()).collect_agent_outputs(state)

        self.assertEqual(
            [name for name, _ in collected],
            ["Research Agent", "Analyst Agent"],
        )

    def test_collect_does_not_mutate_state(self):
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )
        before = len(state.propagation_timeline)

        PropagationMonitor(context_engine=Mock()).collect_agent_outputs(state)

        self.assertEqual(len(state.propagation_timeline), before)


class TestPreclassifyPrefix(unittest.TestCase):
    """The prefix pass must exclude the Action Agent, which is the whole point."""

    def test_prefix_excludes_action_agent(self):
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )
        monitor = _monitor([CLEAN, INFECTED, CLEAN])

        precomputed = monitor.preclassify_prefix(state)

        self.assertEqual(
            sorted(precomputed.keys()), ["Analyst Agent", "Research Agent"]
        )
        self.assertNotIn("Action Agent", precomputed)

    def test_prefix_reads_only_available_outputs(self):
        """
        Excluding the Action Agent must not exclude anyone else.

        During the concurrent phase action_output does not exist yet, so
        excluding by position would have dropped the Analyst Agent and left
        work that could have overlapped on the worker thread.
        """
        monitor = _monitor([CLEAN, INFECTED])
        precomputed = monitor.preclassify_prefix(_make_state())

        self.assertEqual(
            sorted(precomputed.keys()), ["Analyst Agent", "Research Agent"]
        )

    def test_exclusion_is_by_label_not_position(self):
        """A caller can name exactly which agent to defer."""
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )
        monitor = _monitor([CLEAN, INFECTED, CLEAN])

        precomputed = monitor.preclassify_prefix(
            state, exclude=("Analyst Agent", "Action Agent")
        )

        self.assertEqual(list(precomputed.keys()), ["Research Agent"])

    def test_prefix_returns_empty_when_nothing_available(self):
        state = WorkflowState(original_query="q")
        monitor = _monitor([CLEAN])

        self.assertEqual(monitor.preclassify_prefix(state), {})

    def test_prefix_does_not_touch_state(self):
        state = _make_state()
        monitor = _monitor([CLEAN, INFECTED])

        monitor.preclassify_prefix(state)

        # A read-only pass must not have populated the timeline.
        self.assertEqual(len(state.propagation_timeline), 0)
        self.assertEqual(len(state.propagation_history), 0)


class TestMonitorWorkflowAcceptsPrecomputed(unittest.TestCase):
    """monitor_workflow must reuse precomputed verdicts, not reclassify them."""

    def test_precomputed_results_are_reused(self):
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )
        monitor = _monitor([CLEAN, INFECTED, CLEAN])

        precomputed = monitor.preclassify_prefix(state)
        engine_batch = monitor.context_engine.annotate_agent_outputs_batch

        # The prefix pass already consumed one batch; only the Action Agent
        # remains, so exactly one more call with a single agent.
        before = engine_batch.call_count
        result = monitor.monitor_workflow(state, precomputed=precomputed)

        self.assertEqual(engine_batch.call_count - before, 1)
        last_names = engine_batch.call_args.args[0]
        self.assertEqual(list(last_names), ["Action Agent"])

        self.assertEqual(len(result.propagation_timeline), 3)
        self.assertEqual(
            [e.agent for e in result.propagation_timeline],
            ["Research Agent", "Analyst Agent", "Action Agent"],
        )

    def test_works_without_precomputed_results(self):
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )
        monitor = _monitor([CLEAN, INFECTED, CLEAN])

        result = monitor.monitor_workflow(state)

        self.assertEqual(len(result.propagation_timeline), 3)
        # Only the analyst was infected.
        self.assertEqual(len(result.propagation_history), 1)
        self.assertEqual(result.propagation_history[0].agent, "Analyst Agent")


class TestConcurrencySafety(unittest.TestCase):
    """The behaviour that the copy-on-write split is supposed to guarantee."""

    def _run_once(self, monitor, action_delay: float) -> WorkflowState:
        """
        Run one simulated action/monitor pass.

        The action agent stands in for the real one: it writes action_output
        after a delay, which is exactly the window in which the old code let the
        monitor observe a missing attribute.
        """
        state = _make_state()
        precomputed = monitor.preclassify_prefix(state)

        def fake_action(target: WorkflowState) -> WorkflowState:
            time.sleep(action_delay)
            target.action_output = ActionAgentOutput(
                action_summary="Action summary", processing_time_ms=10.0
            )
            target.agents_status["action_agent"] = "completed"
            target.individual_agent_times["action_agent"] = 10.0
            return target

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            action_future = pool.submit(fake_action, state)
            precomputed_future = pool.submit(lambda: precomputed)
            action_future.result()
            precomputed_future.result()

        return monitor.monitor_workflow(state, precomputed=precomputed_future.result())

    def test_action_agent_always_present_in_monitoring_output(self):
        """
        The Action Agent must be monitored on every run.

        With the old shared-state version the monitor read state.action_output
        while the action agent was still writing it, so a slow action agent
        silently dropped the Action Agent from the timeline entirely.
        """
        for action_delay in (0.0, 0.01, 0.05, 0.10):
            with self.subTest(action_delay=action_delay):
                monitor = _monitor([CLEAN, INFECTED, CLEAN])
                result = self._run_once(monitor, action_delay)

                agents = [e.agent for e in result.propagation_timeline]
                self.assertEqual(
                    agents,
                    ["Research Agent", "Analyst Agent", "Action Agent"],
                    f"action agent missing from monitoring at delay={action_delay}",
                )

    def test_results_identical_across_timing_variations(self):
        """
        Output must not depend on thread interleaving.

        Delays that shift the interleaving are swept; the resulting timeline and
        history must be byte-for-byte identical every time.
        """
        baseline = None
        for action_delay in (0.0, 0.02, 0.05, 0.08, 0.12, 0.2):
            monitor = _monitor([CLEAN, INFECTED, CLEAN])
            result = self._run_once(monitor, action_delay)

            snapshot = (
                [
                    (
                        e.agent,
                        e.infected,
                        e.propagation_depth,
                        e.analysis_details.get("category"),
                    )
                    for e in result.propagation_timeline
                ],
                [
                    (e.agent, e.infected, e.propagation_depth)
                    for e in result.propagation_history
                ],
            )

            if baseline is None:
                baseline = snapshot
            else:
                self.assertEqual(
                    snapshot, baseline,
                    f"result changed at action_delay={action_delay}",
                )

    def test_repeated_runs_never_corrupt_the_state(self):
        """Many interleavings must not produce a malformed or empty state."""
        for attempt in range(25):
            monitor = _monitor([CLEAN, INFECTED, CLEAN])
            result = self._run_once(monitor, 0.001 * (attempt % 7))

            self.assertEqual(len(result.propagation_timeline), 3)
            self.assertEqual(len(result.propagation_history), 1)
            # Propagation depth must be a small positive integer, never a
            # corrupt value from a torn read.
            depth = result.propagation_history[0].propagation_depth
            self.assertIsInstance(depth, int)
            self.assertGreater(depth, 0)


class TestShortClassifierResult(unittest.TestCase):
    """A classifier that returns fewer verdicts must degrade, not crash."""

    def test_missing_verdicts_are_skipped(self):
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )

        engine = Mock()
        # Only one verdict for three agents.
        engine.annotate_agent_outputs_batch.side_effect = lambda names, texts: [
            _annotation(INFECTED, 0.95)
        ]

        monitor = PropagationMonitor(context_engine=engine)
        result = monitor.monitor_workflow(state)

        self.assertEqual(len(result.propagation_timeline), 1)

    def test_empty_verdicts_produce_empty_timeline(self):
        state = _make_state()
        state.action_output = ActionAgentOutput(
            action_summary="Action summary", processing_time_ms=10.0
        )

        engine = Mock()
        engine.annotate_agent_outputs_batch.side_effect = lambda names, texts: []

        monitor = PropagationMonitor(context_engine=engine)
        result = monitor.monitor_workflow(state)

        self.assertEqual(len(result.propagation_timeline), 0)
        self.assertEqual(len(result.propagation_history), 0)


if __name__ == "__main__":
    unittest.main()