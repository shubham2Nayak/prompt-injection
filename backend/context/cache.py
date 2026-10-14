"""Cache annotations using normalized text hashes.

Avoids repeated inference for identical chunks by caching annotation
results. Uses SHA-256 of normalized text as the key.

Caching is intentional: when the same chunk text appears across multiple
queries or documents, we reuse the previously computed annotation instead
of re-running the model, which improves throughput and reduces cost.
"""

import hashlib
import logging
from typing import Dict, Optional, Any

from .taxonomy import AnnotationCategory
from .schema import ContextAnnotation

logger = logging.getLogger(__name__)


class AnnotationCache:
    """Simple in-memory cache for annotation results."""

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initialize the annotation cache.

        Args:
            max_size: Maximum number of entries before eviction
        """
        self._cache: Dict[str, ContextAnnotation] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _hash_text(self, text: str) -> str:
        """Compute a normalized SHA-256 hash for the given text."""
        # Normalize: lowercase, strip, collapse whitespace
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[ContextAnnotation]:
        """
        Retrieve a cached annotation for the given text.

        Args:
            text: The chunk text to look up

        Returns:
            Cached ContextAnnotation if found, None otherwise
        """
        h = self._hash_text(text)
        if h in self._cache:
            self._hits += 1
            return self._cache[h]
        self._misses += 1
        return None

    def set(self, text: str, annotation: ContextAnnotation) -> None:
        """
        Cache an annotation for the given text.

        Args:
            text: The chunk text
            annotation: The ContextAnnotation to cache
        """
        h = self._hash_text(text)
        # Enforce max size with simple LRU-like eviction
        if len(self._cache) >= self._max_size and h not in self._cache:
            # Remove first entry (simple FIFO)
            first_key = next(iter(self._cache))
            del self._cache[first_key]
            logger.debug(f"Cache evicted entry for hash {first_key[:8]}")

        self._cache[h] = annotation
        logger.debug(f"Cached annotation for hash {h[:8]}")

    def stats(self) -> dict:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "cache_size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }