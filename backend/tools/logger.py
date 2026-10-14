"""
Centralized Logging for PromptTrace
Structured logging for propagation tracking and debugging.

Propagation events are served from a bounded in-memory ring rather than by
re-reading ``propagation_events.jsonl``. Every log endpoint previously opened the
whole file and ran ``json.loads`` over every line, six times per dashboard
refresh, from inside ``async def`` handlers - so each poll blocked the event loop
for work proportional to the entire history.

Writes go through the shared asynchronous JSONL writer, which keeps one
rotating file handle open on a background thread instead of opening, writing and
closing the file per event.
"""

import logging
import json
import threading
from typing import Any, Dict, List, Optional
from collections import deque
from datetime import datetime
from pathlib import Path

from config import config
from jsonl_writer import enqueue_jsonl, jsonl_writer_stats, close_jsonl_writers


class PropagationLogger:
    """Logger for propagation events and workflow state."""

    def __init__(self, log_dir: str = None):
        """
        Initialize propagation logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir or config.LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create main logger
        self.logger = logging.getLogger("PromptTrace")
        self.logger.setLevel(getattr(logging, config.LOG_LEVEL))

        # File handler. Guarded so repeated construction (tests, reloads) does
        # not stack duplicate handlers and duplicate every record.
        log_file = self.log_dir / "prompttrace.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, config.LOG_LEVEL))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.LOG_LEVEL))

        # Formatter
        formatter = logging.Formatter(config.LOG_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers, replacing any this logger already owns.
        existing = {type(h) for h in self.logger.handlers}
        if logging.FileHandler not in existing:
            self.logger.addHandler(file_handler)
        if logging.StreamHandler not in existing:
            self.logger.addHandler(console_handler)

        # Propagation events log
        self.events_file = self.log_dir / "propagation_events.jsonl"

        # Bounded in-memory ring of recent events, served directly by the API.
        self._events: deque = deque(
            maxlen=max(1, int(config.PROPAGATION_EVENT_BUFFER_SIZE))
        )
        self._events_lock = threading.Lock()

        # Running totals so /logs/status and /logs/summary are O(1) instead of
        # scanning the buffer.
        self._total_events = 0
        self._total_infected = 0
        self._confidence_sum = 0.0
        self._agents_seen: set = set()
        self._threat_phrases: set = set()

        # One-shot tail load of pre-existing events, deferred to the first read so
        # importing this module does not scan the log directory.
        self._bootstrapped = False
        self._bootstrap_lock = threading.Lock()

    # -- event recording --------------------------------------------------

    def log_workflow_start(self, workflow_id: str, query: str) -> None:
        """Log workflow start."""
        self.logger.info(f"[WORKFLOW START] ID: {workflow_id}, Query: {query[:100]}...")

    def log_workflow_end(
        self, workflow_id: str, status: str, duration_ms: float
    ) -> None:
        """Log workflow end."""
        self.logger.info(
            f"[WORKFLOW END] ID: {workflow_id}, Status: {status}, Duration: {duration_ms:.2f}ms"
        )

    def log_agent_start(self, workflow_id: str, agent_name: str) -> None:
        """Log agent execution start."""
        self.logger.info(f"[AGENT START] {agent_name} (Workflow: {workflow_id})")

    def log_agent_end(
        self, workflow_id: str, agent_name: str, duration_ms: float, output_size: int
    ) -> None:
        """Log agent execution end."""
        self.logger.info(
            f"[AGENT END] {agent_name}, Duration: {duration_ms:.2f}ms, "
            f"Output size: {output_size} chars"
        )

    def log_propagation_event(self, event: Dict[str, Any]) -> None:
        """
        Record a propagation detection event.

        The event is appended to the in-memory ring and the running totals under
        a lock, then handed to the background writer. Nothing here touches the
        filesystem, so this stays cheap enough for the inference hot path.

        Args:
            event: Propagation event dictionary
        """
        event["timestamp"] = datetime.utcnow().isoformat()

        with self._events_lock:
            self._events.append(event)
            self._total_events += 1
            if event.get("infected"):
                self._total_infected += 1
            self._confidence_sum += float(event.get("confidence", 0) or 0)

            agent = event.get("agent")
            if agent:
                self._agents_seen.add(agent)

            phrase = event.get("attack_phrase")
            if phrase:
                self._threat_phrases.add(phrase)

        # Persist asynchronously. Never raises, so telemetry cannot fail a run.
        enqueue_jsonl(str(self.events_file), event)

        # Log to console
        self.logger.warning(
            f"[PROPAGATION] Agent: {event.get('agent')}, "
            f"Infected: {event.get('infected')}, "
            f"Confidence: {event.get('confidence', 0):.2f}"
        )

    def log_retrieval(
        self, workflow_id: str, query: str, num_chunks: int, time_ms: float
    ) -> None:
        """Log RAG retrieval."""
        self.logger.info(
            f"[RETRIEVAL] {num_chunks} chunks in {time_ms:.2f}ms "
            f"(Query: {query[:50]}..., Workflow: {workflow_id})"
        )

    def log_error(
        self, workflow_id: str, component: str, error: str, details: Optional[str] = None
    ) -> None:
        """Log error."""
        self.logger.error(
            f"[ERROR] {component} (Workflow: {workflow_id}): {error}\n"
            f"Details: {details or 'None'}"
        )

    # -- reads ------------------------------------------------------------

    def _ensure_bootstrapped(self) -> None:
        """
        Seed the ring from an existing log file exactly once, and only when this
        process has not already recorded anything.

        The second condition matters: without it, a process that has been writing
        events through ``log_propagation_event`` would also re-read its own output
        from disk on the first read and count every event twice.

        Bounded to the tail of the file: this runs before the first request is
        served, and reading an unbounded history would defeat the point of the
        buffer. Anything older than the window remains on disk.
        """
        if self._bootstrapped:
            return

        with self._bootstrap_lock:
            if self._bootstrapped:
                return

            # Claim the bootstrap before releasing any lock so concurrent first
            # readers cannot both seed the ring.
            self._bootstrapped = True

            # Already have events from this process: the file already holds
            # them, and re-reading would duplicate the ring and the counters.
            if self._total_events > 0:
                return

            if not self.events_file.exists():
                return

            try:
                for event in self._read_tail(self.events_file):
                    with self._events_lock:
                        self._events.append(event)
                        self._total_events += 1
                        if event.get("infected"):
                            self._total_infected += 1
                        self._confidence_sum += float(event.get("confidence", 0) or 0)
                        if event.get("agent"):
                            self._agents_seen.add(event["agent"])
                        if event.get("attack_phrase"):
                            self._threat_phrases.add(event["attack_phrase"])
            except Exception as e:
                self.logger.warning(f"Could not preload propagation log: {e}")

    @staticmethod
    def _read_tail(path: Path, max_lines: int = None) -> List[Dict[str, Any]]:
        """
        Read at most the last ``max_lines`` JSONL records from ``path``.

        Reads backwards in fixed-size blocks so a large log costs memory
        proportional to the window rather than to the file.
        """
        if max_lines is None:
            max_lines = max(1, int(config.PROPAGATION_LOG_BOOTSTRAP_LINES))

        if not path.exists():
            return []

        block_size = 8192
        events: List[Dict[str, Any]] = []
        collected = ""

        with open(path, "rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()

            while position > 0 and len(events) < max_lines:
                read_size = min(block_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                collected = chunk.decode("utf-8", errors="replace") + collected

                lines = collected.split("\n")
                # The first element is a partial line (or empty); keep it for the
                # next iteration since it continues into the preceding block.
                collected = lines[0]
                complete = [line for line in lines[1:] if line.strip()]

                for line in reversed(complete):
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(events) >= max_lines:
                        break

        return events

    def get_propagation_log(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Read recent propagation events from memory.

        Args:
            limit: Return at most this many of the most recent events

        Returns:
            List of propagation event dictionaries, oldest first
        """
        self._ensure_bootstrapped()

        with self._events_lock:
            if limit is not None and limit > 0:
                events = list(self._events)[-limit:]
            else:
                events = list(self._events)
        return events

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return the ``limit`` most recent events, newest first.

        Avoids sorting the whole buffer per request; the ring is already in
        chronological order, so the newest events are the tail.
        """
        self._ensure_bootstrapped()

        with self._events_lock:
            if limit is not None and limit > 0:
                events = list(self._events)[-limit:]
            else:
                events = list(self._events)
        events.reverse()
        return events

    def get_events_by_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        """Return buffered events for one agent, oldest first."""
        self._ensure_bootstrapped()
        with self._events_lock:
            return [e for e in self._events if e.get("agent") == agent_name]

    def get_timeline(self, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return buffered events in chronological order.

        Args:
            hours: Accepted for API compatibility. The buffer is already
                chronological and bounded, so no time filtering is applied.
        """
        return self.get_propagation_log()

    def get_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics from running totals.

        Returns:
            Dictionary of aggregate propagation statistics
        """
        self._ensure_bootstrapped()

        with self._events_lock:
            total = self._total_events
            infected = self._total_infected
            confidence_sum = self._confidence_sum
            agents = sorted(self._agents_seen)
            threats = sorted(self._threat_phrases)

        average_confidence = confidence_sum / total if total else 0.0

        return {
            "total_events": total,
            "infected_events": infected,
            "agents_affected": agents,
            "average_confidence": average_confidence,
            "detection_rate": infected / total if total else 0.0,
            "buffered_events": len(self._events),
            "buffer_limit": self._events.maxlen,
        }

    def get_threats(self) -> Dict[str, Any]:
        """Return the distinct attack phrases seen, from a maintained set."""
        self._ensure_bootstrapped()

        with self._events_lock:
            threats = sorted(self._threat_phrases)
            total = self._total_events

        return {
            "threats": threats,
            "count": len(threats),
            "total_events": total,
        }

    def get_status(self) -> Dict[str, Any]:
        """Return log status without scanning the buffer or the file."""
        self._ensure_bootstrapped()

        with self._events_lock:
            buffered = len(self._events)

        file_size = None
        try:
            file_size = self.events_file.stat().st_size
        except OSError:
            pass

        return {
            "total_events": self._total_events,
            "buffered_events": buffered,
            "buffer_limit": self._events.maxlen,
            "log_file": str(self.events_file),
            "logs_directory": str(self.log_dir),
            "log_file_size_bytes": file_size,
            "writer_stats": jsonl_writer_stats(),
            "status": "operational",
        }

    def clear_logs(self) -> None:
        """Clear all log files and reset the in-memory buffers."""
        for log_file in self.log_dir.glob("*.log"):
            log_file.unlink()
        if self.events_file.exists():
            self.events_file.unlink()

        with self._events_lock:
            self._events.clear()
            self._total_events = 0
            self._total_infected = 0
            self._confidence_sum = 0.0
            self._agents_seen.clear()
            self._threat_phrases.clear()

        # The file is gone; a later bootstrap must not resurrect it.
        self._bootstrapped = True

        close_jsonl_writers()
        self.logger.info("Logs cleared")


# Global logger instance
propagation_logger = PropagationLogger()