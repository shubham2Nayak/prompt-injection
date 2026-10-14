"""
Asynchronous JSONL writer.

Teeing telemetry to disk used to be done with ``open(path, "a")`` / ``write`` /
``close`` at every call site. On the inference hot path that is three syscalls
and a flush per event, and where it happened inside a shared mutex it also
serialised otherwise-parallel work behind the disk.

This module replaces that pattern with a single background writer thread per log
file:

* Callers only serialize the record and push it onto a bounded queue.
* One long-lived ``RotatingFileHandler`` keeps the file descriptor open, so the
  cost per event collapses to a ``write`` with no ``open``/``close``.
* The queue is bounded and drops the oldest pending record when full, so a slow
  or wedged disk degrades telemetry instead of stalling inference.
* A single worker thread drains the queue with no lock contention against the
  callers.

Usage::

    from jsonl_writer import enqueue_jsonl, close_jsonl_writers

    enqueue_jsonl("/path/to/events.jsonl", {"agent": "research", "ms": 12.5})

Writers are registered lazily on first use and shut down via
``close_jsonl_writers()`` at process exit. Callers are not expected to manage
the lifecycle; a flush-on-exit is registered with ``atexit``.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from config import config

logger = logging.getLogger(__name__)

# Sentinel pushed on shutdown so the worker knows to drain and exit.
_SHUTDOWN = object()

# A record that failed to serialize still needs to occupy a slot so the worker
# can account for it without special-casing None.
_UNSERIALIZABLE = "\x00<unserializable>"


class _JSONLWriter:
    """Background writer owning one rotating JSONL file."""

    def __init__(
        self,
        path: str,
        max_bytes: int = config.INVOCATION_LOG_MAX_BYTES,
        backup_count: int = config.INVOCATION_LOG_BACKUP_COUNT,
        queue_size: int = config.INVOCATION_LOG_QUEUE_SIZE,
    ) -> None:
        self._path = Path(path)
        self._max_bytes = max(1, int(max_bytes))
        self._backup_count = max(0, int(backup_count))
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._dropped = 0
        self._written = 0
        self._closed = False

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handler = RotatingFileHandler(
                self._path,
                maxBytes=self._max_bytes,
                backupCount=self._backup_count,
                encoding="utf-8",
                delay=False,
            )
            # Records are already serialized JSON lines, so the logging prefix
            # is suppressed; only the message reaches the file.
            self._handler.setFormatter(logging.Formatter("%(message)s"))
        except Exception as exc:  # pragma: no cover - unwritable log directory
            # Telemetry must never be the reason a request fails. Degrade to a
            # no-op writer and keep the application running.
            logger.warning(
                "JSONL writer for %s disabled; cannot open file: %s", self._path, exc
            )
            self._handler = None
            self._queue = queue.Queue(maxsize=1)

        self._thread = threading.Thread(
            target=self._run,
            name=f"jsonl-writer:{self._path.name}",
            daemon=True,
        )
        self._thread.start()

    # -- producer side ----------------------------------------------------

    def write(self, record: Dict[str, Any]) -> None:
        """
        Serialize and enqueue one record.

        Never blocks and never raises: a full queue discards the oldest pending
        record so that logging cannot become the bottleneck it was added to
        measure.
        """
        if self._closed or self._handler is None:
            return

        try:
            payload = json.dumps(record, default=str) + "\n"
        except Exception:
            payload = _UNSERIALIZABLE + "\n"

        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._dropped += 1
                self._queue.put_nowait(payload)
            except Exception:
                self._dropped += 1

    # -- consumer side ----------------------------------------------------

    def _run(self) -> None:
        """Drain the queue until the shutdown sentinel arrives."""
        while True:
            item = self._queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                # RotatingFileHandler is a FileHandler, not a StreamHandler, so
                # it has no .write(). Routing a bare LogRecord through handle()
                # is what performs the rollover check and the actual write.
                self._handler.handle(
                    logging.LogRecord(
                        name="jsonl",
                        level=logging.INFO,
                        pathname=str(self._path),
                        lineno=0,
                        msg=item,
                        args=None,
                        exc_info=None,
                    )
                )
                self._handler.flush()
                self._written += 1
            except Exception as exc:
                logger.warning("Failed to write JSONL record to %s: %s", self._path, exc)

    # -- lifecycle --------------------------------------------------------

    def close(self, timeout: float = 5.0) -> None:
        """Flush pending records and stop the worker thread."""
        if self._closed:
            return
        self._closed = True

        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            # Queue is saturated; drop pending records so shutdown can proceed.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

        self._thread.join(timeout=timeout)

        if self._handler is not None:
            try:
                self._handler.close()
            except Exception:
                pass

    def stats(self) -> Dict[str, int]:
        """Return writer counters for diagnostics."""
        return {
            "path": str(self._path),
            "written": self._written,
            "dropped": self._dropped,
            "pending": self._queue.qsize(),
        }


# Process-wide registry, guarded only during writer creation. Writers are
# long-lived so this lock is taken once per file, not once per record.
_writers: "OrderedDict[str, _JSONLWriter]" = OrderedDict()
_registry_lock = threading.Lock()
_atexit_registered = False


def _register_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(close_jsonl_writers)
        _atexit_registered = True


def get_writer(path: str) -> Optional[_JSONLWriter]:
    """
    Return the writer for ``path``, creating it on first use.

    Returns ``None`` if the file cannot be opened for writing.
    """
    key = str(path)

    writer = _writers.get(key)
    if writer is not None:
        return writer

    with _registry_lock:
        # Re-check inside the lock: another thread may have created it while we
        # waited.
        writer = _writers.get(key)
        if writer is not None:
            return writer

        writer = _JSONLWriter(key)
        if writer._handler is None:
            return None

        _writers[key] = writer
        _register_atexit()
        logger.debug("Started async JSONL writer for %s", key)
        return writer


def enqueue_jsonl(path: str, record: Dict[str, Any]) -> bool:
    """
    Enqueue ``record`` for asynchronous JSONL persistence.

    Returns True when the record was accepted by a writer, False when no writer
    is available. Never raises and never blocks.
    """
    try:
        writer = get_writer(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not obtain JSONL writer for %s: %s", path, exc)
        return False

    if writer is None:
        return False

    writer.write(record)
    return True


def close_jsonl_writers(timeout: float = 5.0) -> None:
    """Flush and stop every registered writer. Safe to call more than once."""
    with _registry_lock:
        writers = list(_writers.values())
        _writers.clear()

    for writer in writers:
        try:
            writer.close(timeout=timeout)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error closing JSONL writer: %s", exc)


def jsonl_writer_stats() -> Dict[str, Dict[str, int]]:
    """Return per-file writer counters."""
    return {path: writer.stats() for path, writer in list(_writers.items())}