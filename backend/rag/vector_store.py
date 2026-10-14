"""
Vector Store for PromptTrace RAG
FAISS-based vector store for semantic search over chunks.

Prefers a GPU-resident FAISS index. When one cannot be created the previous
behaviour was to log a warning and continue on CPU, which hid the fact that every
query was paying a blocking device-to-host copy on a host believed to be doing
GPU retrieval. Set ``FAISS_REQUIRE_GPU=false`` to restore that fallback
deliberately; by default a requested GPU index that is unavailable is an error.

Embeddings are cast to float32 on the device and transferred through page-locked
(pinned) host staging memory, so the copy overlaps with the pending GPU work
instead of forcing a synchronization point on every call.
"""

import logging
import json
import pickle
import threading
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import faiss
import torch

from config import config

logger = logging.getLogger(__name__)


class FAISSGPUNotAvailableError(RuntimeError):
    """Raised when a GPU index is required but FAISS cannot provide one."""


class FAISSVectorStore:
    """FAISS-based vector store for semantic search over chunks.

    Supports GPU acceleration via faiss-gpu with direct CUDA tensor input.
    Host transfers use pinned staging buffers so embedding on the device does not
    stall behind a synchronous ``cudaMemcpy``.
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        metric: str = "L2",
        use_gpu: bool = False,
        gpu_device: int = 0,
    ) -> None:
        """
        Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('Flat', 'IVF')
            metric: Distance metric ('L2' or 'IP')
            use_gpu: Whether to place the index on GPU
            gpu_device: GPU device ID to use
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[dict] = []
        self._gpu_resources = None

        # Statistics maintained incrementally. Recomputing them with set()
        # comprehensions over every record made get_stats() O(total_chunks) on
        # each call, and it is called on every ingest and on the status endpoint.
        self._document_names: set = set()
        self._page_numbers: set = set()

        # FAISS and the vector store are shared across request threads. Index
        # mutation and metadata append must not interleave, or metadata stops
        # lining up with index rows.
        self._lock = threading.RLock()

        # Reusable pinned host buffers for the query path. Allocating page-locked
        # memory is expensive, so one buffer per vector-store instance is kept and
        # reused; a GPU vector store does one query copy per search, and this makes
        # that copy a genuine async DMA rather than a stall.
        self._query_staging: Optional[torch.Tensor] = None

        self._initialize_index()

    def _initialize_index(self) -> None:
        """Initialize FAISS index, optionally on GPU."""
        # Create CPU index first
        if self.metric == "L2":
            cpu_index = faiss.IndexFlatL2(self.embedding_dim)
        elif self.metric == "IP":  # Inner Product
            cpu_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        if not self.use_gpu:
            self.index = cpu_index
            logger.info(
                f"Initialized FAISS {self.index_type} index on CPU "
                f"(dim={self.embedding_dim}, metric={self.metric})"
            )
            return

        # A GPU index was requested. Resolve it, or fail loudly.
        if not self._is_gpu_available():
            self._handle_gpu_unavailable(
                "faiss was built without GPU support "
                "(faiss.StandardGpuResources is unavailable); install "
                "faiss-gpu-cu12"
            )
            return

        try:
            self._gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(
                self._gpu_resources, self.gpu_device, cpu_index
            )
            logger.info(
                f"Initialized FAISS {self.index_type} index on GPU:{self.gpu_device} "
                f"(dim={self.embedding_dim}, metric={self.metric})"
            )
        except Exception as e:
            self._handle_gpu_unavailable(
                f"faiss.index_cpu_to_gpu failed on device {self.gpu_device}: {e}"
            )

    def _handle_gpu_unavailable(self, reason: str) -> None:
        """
        Decide what to do when a requested GPU index cannot be created.

        Silently degrading to CPU here is what made the previous GPU path
        unverifiable: the store kept answering queries, just slowly, and the only
        signal was a warning buried in the startup log. The fallback is now
        opt-in via ``FAISS_REQUIRE_GPU``.
        """
        message = (
            f"FAISS GPU index requested but unavailable: {reason}. "
            "Set FAISS_REQUIRE_GPU=false to fall back to a CPU index."
        )

        if config.FAISS_REQUIRE_GPU:
            logger.error(message)
            raise FAISSGPUNotAvailableError(message)

        logger.warning(message + " Continuing on CPU.")
        self.index = faiss.IndexFlatL2(self.embedding_dim) if self.metric == "L2" \
            else faiss.IndexFlatIP(self.embedding_dim)
        self.use_gpu = False

    def _is_gpu_available(self) -> bool:
        """Check if FAISS GPU support is available in this build."""
        try:
            return hasattr(faiss, "StandardGpuResources") and hasattr(
                faiss, "index_cpu_to_gpu"
            )
        except Exception:
            return False

    def _pinned_buffer(self, needed: int) -> torch.Tensor:
        """
        Return a reusable page-locked float32 host tensor of at least ``needed``.

        PyTorch only honours ``non_blocking=True`` on a device-to-host copy when
        the destination is page-locked. With ordinary pageable memory the copy
        degenerates into a synchronous ``cudaMemcpy`` that waits for all pending
        GPU work to drain, which is a hard synchronization point on every search.
        Allocating pinned memory is itself slow, so one buffer per store is kept
        and grown on demand rather than reallocated per query.
        """
        buffer = self._query_staging
        if buffer is not None and buffer.numel() >= needed:
            return buffer

        try:
            buffer = torch.empty(needed, dtype=torch.float32).pin_memory()
        except (RuntimeError, AttributeError) as exc:
            # Pinning can fail on locked-down hosts. Fall back to pageable
            # memory rather than failing the request; the transfer is then
            # synchronous, which is the behaviour we had before.
            logger.warning(
                "Could not allocate pinned staging buffer (%s); "
                "falling back to pageable host memory",
                exc,
            )
            buffer = torch.empty(needed, dtype=torch.float32)

        self._query_staging = buffer
        return buffer

    @staticmethod
    def _copy_to_faiss_array(
        embeddings: Union[torch.Tensor, np.ndarray, List],
    ) -> np.ndarray:
        """
        Convert embeddings into a contiguous float32 numpy array for FAISS.

        Two copies used to happen here. Embeddings arrive as float16 CUDA
        tensors, so ``.to("cpu", non_blocking=True)`` produced a float16 host
        array and the following ``.astype("float32")`` allocated and filled a
        second full-size array. The dtype conversion is now done on the device,
        where it is essentially free, so exactly one host array is allocated.
        """
        if isinstance(embeddings, list):
            # Stack list of tensors/arrays
            if embeddings and isinstance(embeddings[0], torch.Tensor):
                embeddings = torch.stack(embeddings)
            else:
                embeddings = np.stack(embeddings)

        if isinstance(embeddings, torch.Tensor):
            if embeddings.dtype != torch.float32:
                # Cheap on-device widening; avoids a full-size host-side astype.
                embeddings = embeddings.to(torch.float32)

            tensor = embeddings.detach()
            if tensor.device.type != "cpu":
                tensor = tensor.to("cpu", non_blocking=True)
            tensor = tensor.contiguous()

            # numpy() shares the tensor's memory, so clone before handing the
            # array to FAISS: the staging tensor is reused by the next call.
            return tensor.numpy().copy()

        # Already numpy
        array = np.asarray(embeddings)
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)
        return np.ascontiguousarray(array)

    def _query_to_faiss_array(
        self,
        query_embedding: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        """
        Stage a single query vector into the reusable pinned buffer.

        The pinned destination lets the device-to-host copy overlap with
        outstanding GPU work; the explicit synchronize afterwards is a cheap
        no-op when the copy already completed and guarantees the buffer holds
        valid data before FAISS reads it.
        """
        if isinstance(query_embedding, np.ndarray):
            array = np.asarray(query_embedding, dtype=np.float32)
            return np.ascontiguousarray(array.reshape(1, -1))

        tensor = query_embedding
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)

        tensor = tensor.detach().reshape(-1)
        expected = tensor.numel()

        staging = self._pinned_buffer(expected)
        target = staging[:expected]

        if tensor.device.type != "cpu":
            # Copy into the pinned destination as a view, so the DMA lands
            # directly in page-locked memory.
            target = staging[:expected].view(expected)
            target.copy_(tensor, non_blocking=True)
            # Wait for the DMA to complete; the buffer is reused immediately.
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()
            return target.numpy().reshape(1, -1)

        staging[:expected].copy_(tensor)
        return staging[:expected].numpy().reshape(1, -1)

    def add_chunks(self, chunks: List[dict]) -> None:
        """
        Add chunks with embeddings to vector store.
        Accepts GPU tensors directly from embedding generator.

        Args:
            chunks: List of chunk dictionaries with 'embedding' key (tensor or numpy)
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        with self._lock:
            embeddings_list = []
            new_metadata = []
            for chunk in chunks:
                if "embedding" not in chunk:
                    raise ValueError("Chunk missing 'embedding' key")
                embeddings_list.append(chunk["embedding"])

                # Store metadata (without embedding)
                metadata = {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "page_number": chunk.get("page_number", 0),
                    "source_document": chunk.get("source_document", ""),
                    "paragraph_offset": chunk.get("paragraph_offset", 0),
                    "character_offset": chunk.get("character_offset", 0),
                    "retrieval_score": chunk.get("retrieval_score", 0.0),
                    "text": chunk.get("text", ""),
                }
                new_metadata.append(metadata)

            # Convert to FAISS array. The dtype widening happens on the device and
            # the pinned destination lets the transfer overlap, so this is one
            # host allocation instead of two full copies.
            embeddings_array = self._copy_to_faiss_array(embeddings_list)

            # Normalize if using inner product
            if self.metric == "IP":
                faiss.normalize_L2(embeddings_array)

            self.index.add(embeddings_array)

            # Metadata must be appended in the same order the rows were added, or
            # every search result maps to the wrong chunk.
            self.chunks_metadata.extend(new_metadata)
            for metadata in new_metadata:
                self._document_names.add(metadata["source_document"])
                self._page_numbers.add(metadata["page_number"])

            total = self.index.ntotal
            logger.info(
                f"Added {len(chunks)} chunks to vector store. Total: {total}"
            )

    def add_embeddings_batch(
        self, embeddings: Union[torch.Tensor, np.ndarray], metadata_list: List[dict]
    ) -> None:
        """
        Add pre-computed embeddings with metadata in a single batch.
        Optimized for GPU tensor input from embedding generator.

        Args:
            embeddings: GPU tensor (n, dim) or numpy array
            metadata_list: List of metadata dicts
        """
        if len(embeddings) == 0:
            return

        with self._lock:
            # Convert efficiently
            embeddings_array = self._copy_to_faiss_array(embeddings)

            if self.metric == "IP":
                faiss.normalize_L2(embeddings_array)

            self.index.add(embeddings_array)
            self.chunks_metadata.extend(metadata_list)
            for metadata in metadata_list:
                self._document_names.add(metadata.get("source_document", ""))
                self._page_numbers.add(metadata.get("page_number", 0))

            total = self.index.ntotal
            logger.info(f"Batch added {len(embeddings)} embeddings. Total: {total}")

    def search(
        self, query_embedding: Union[torch.Tensor, np.ndarray], top_k: int = 5
    ) -> List[Tuple[dict, float]]:
        """
        Search for similar chunks using GPU tensor query.

        Args:
            query_embedding: Query embedding vector (tensor on GPU or numpy)
            top_k: Number of results to return

        Returns:
            List of (chunk_metadata, similarity) tuples
        """
        with self._lock:
            ntotal = self.index.ntotal

        if ntotal == 0:
            logger.warning("Vector store is empty")
            return []

        # Stage the query through the pinned buffer instead of letting the
        # transfer itself become a synchronization point.
        query_array = self._query_to_faiss_array(query_embedding)

        if self.metric == "IP":
            faiss.normalize_L2(query_array)

        # Search on GPU (if GPU index) or CPU
        with self._lock:
            distances, indices = self.index.search(query_array, min(top_k, ntotal))

        # Build results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue

            with self._lock:
                chunk_metadata = self.chunks_metadata[int(idx)].copy()
            if self.metric == "L2":
                similarity = 1 / (1 + float(distance))
            else:  # IP
                similarity = float(distance)

            results.append((chunk_metadata, similarity))

        logger.debug(f"Retrieved {len(results)} results for query")
        return results

    def search_batch(
        self,
        query_embeddings: Union[torch.Tensor, np.ndarray],
        top_k: int = 5,
    ) -> List[List[Tuple[dict, float]]]:
        """
        Batched search for multiple queries using GPU tensor input.

        Args:
            query_embeddings: Batch of query embeddings (n_queries, dim) as GPU tensor or numpy
            top_k: Number of results per query

        Returns:
            List of result lists per query
        """
        with self._lock:
            ntotal = self.index.ntotal

        if ntotal == 0:
            logger.warning("Vector store is empty")
            return [[] for _ in range(len(query_embeddings))]

        # Convert batch to FAISS array
        query_array = self._copy_to_faiss_array(query_embeddings)
        # Ensure 2D: (n_queries, dim)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)

        if self.metric == "IP":
            faiss.normalize_L2(query_array)

        # Batched search on GPU/CPU
        with self._lock:
            distances, indices = self.index.search(query_array, min(top_k, ntotal))

        # Build results for each query
        all_results = []
        for query_idx in range(query_array.shape[0]):
            results = []
            for idx, distance in zip(indices[query_idx], distances[query_idx]):
                if idx == -1:
                    continue

                with self._lock:
                    chunk_metadata = self.chunks_metadata[int(idx)].copy()
                if self.metric == "L2":
                    similarity = 1 / (1 + float(distance))
                else:
                    similarity = float(distance)

                results.append((chunk_metadata, similarity))

            all_results.append(results)

        return all_results

    # Compatibility alias
    batch_search = search_batch

    def save(self, path: str) -> None:
        """
        Save vector store to disk.

        Args:
            path: Directory path to save to
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save FAISS index (move to CPU first if on GPU)
        index_to_save = self.index
        if self.use_gpu and self._gpu_resources is not None:
            index_to_save = faiss.index_gpu_to_cpu(self.index)

        index_path = path_obj / "faiss_index.bin"
        faiss.write_index(index_to_save, str(index_path))

        # Save metadata
        metadata_path = path_obj / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.chunks_metadata, f, indent=2, default=str)

        # Save config
        config_path = path_obj / "config.pkl"
        save_config = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric": self.metric,
            "use_gpu": self.use_gpu,
            "gpu_device": self.gpu_device,
        }
        with open(config_path, "wb") as f:
            pickle.dump(save_config, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: str) -> None:
        """
        Load vector store from disk.

        Args:
            path: Directory path to load from
        """
        path_obj = Path(path)

        # Load config
        config_path = path_obj / "config.pkl"
        with open(config_path, "rb") as f:
            save_config = pickle.load(f)

        self.embedding_dim = save_config["embedding_dim"]
        self.index_type = save_config["index_type"]
        self.metric = save_config["metric"]
        self.use_gpu = save_config.get("use_gpu", False)
        self.gpu_device = save_config.get("gpu_device", 0)

        # Load FAISS index (CPU)
        index_path = path_obj / "faiss_index.bin"
        cpu_index = faiss.read_index(str(index_path))

        # Move to GPU if configured
        if self.use_gpu and self._is_gpu_available():
            try:
                self._gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(
                    self._gpu_resources, self.gpu_device, cpu_index
                )
                logger.info(f"Loaded vector store on GPU:{self.gpu_device} ({cpu_index.ntotal} chunks)")
            except Exception as e:
                self._handle_gpu_unavailable(
                    f"faiss.index_cpu_to_gpu failed while loading {path}: {e}"
                )
                # The CPU index that was just read is a valid substitute; reuse it
                # rather than building a differently-parameterized one.
                self.index = cpu_index
        else:
            self.index = cpu_index
            logger.info(f"Loaded vector store from {path} on CPU ({cpu_index.ntotal} chunks)")

        # Load metadata
        metadata_path = path_obj / "metadata.json"
        with open(metadata_path, "r") as f:
            self.chunks_metadata = json.load(f)

        # Rebuild the derived statistics incrementally.
        self._document_names = {
            m.get("source_document", "") for m in self.chunks_metadata
        }
        self._page_numbers = {
            m.get("page_number", 0) for m in self.chunks_metadata
        }

    def get_stats(self) -> dict:
        """
        Get vector store statistics.

        Returns:
            Dictionary with stats
        """
        with self._lock:
            total_chunks = self.index.ntotal
            documents = len(self._document_names)
            pages = len(self._page_numbers)

        return {
            "total_chunks": total_chunks,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "metric": self.metric,
            "use_gpu": self.use_gpu,
            "gpu_device": self.gpu_device if self.use_gpu else None,
            # Served from counters maintained during add_chunks rather than
            # rebuilt with set() comprehensions over every record on each call.
            "documents": documents,
            "pages": pages,
        }
