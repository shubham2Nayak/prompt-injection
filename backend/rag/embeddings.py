"""
Embedding Generation for PromptTrace RAG
Creates embeddings using SentenceTransformers on GPU with native PyTorch CUDA tensors.
Eliminates CPU NumPy copies and host-device memcpy overhead.
"""

import logging
from typing import List, Optional, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks on GPU with zero-copy tensor flow."""

    def __init__(
        self, model_name: str, device: str = "cuda", batch_size: int = 32
    ) -> None:
        """
        Initialize embedding model on GPU.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Default batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model directly on target device."""
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            # SentenceTransformer defaults to fp32, which doubles the VRAM the
            # model holds for no retrieval-quality gain on this stack. On CUDA
            # load fp16 to keep the resident set small enough to leave room for
            # the two LLMs.
            model_kwargs = {}
            if self.device.startswith("cuda"):
                model_kwargs["torch_dtype"] = torch.float16
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs,
            )
            # Disable gradient computation for inference
            self.model.eval()
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded successfully on {self.device} "
                f"({next(self.model.parameters()).dtype}). "
                f"Dimension: {self._embedding_dim}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def embed_text(
        self, text: str, return_tensor: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            return_tensor: If True, return PyTorch CUDA tensor; if False, return NumPy array

        Returns:
            Embedding vector on GPU (tensor) or CPU (numpy)
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        # encode with convert_to_tensor=True keeps data on GPU
        embedding = self.model.encode(
            [text],
            batch_size=1,
            convert_to_tensor=True,  # GPU-native tensor, no CPU copy
            device=self.device,
            show_progress_bar=False,
        )
        # embedding shape: (1, dim) - squeeze to (dim,)
        embedding = embedding.squeeze(0)
        if return_tensor:
            return embedding  # CUDA tensor on GPU
        return embedding.detach().cpu().numpy()  # Only move to CPU if explicitly requested

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        return_tensor: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate embeddings for multiple texts with GPU-native batch processing.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (uses default if None)
            show_progress_bar: Whether to show progress bar
            return_tensor: If True, return PyTorch CUDA tensor; if False, return NumPy array

        Returns:
            Embeddings tensor on GPU (n_texts, embedding_dim) or NumPy array on CPU
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        bs = batch_size or self.batch_size
        logger.debug(f"Embedding {len(texts)} texts with batch size {bs} on {self.device}")

        # Single batched call - keeps all computation on GPU
        embeddings = self.model.encode(
            texts,
            batch_size=bs,
            convert_to_tensor=True,  # GPU-native tensor, no CPU copy
            device=self.device,
            show_progress_bar=show_progress_bar,
        )
        # embeddings shape: (n_texts, dim) - already on GPU
        logger.debug(f"Successfully embedded {len(texts)} texts on GPU")

        if return_tensor:
            return embeddings  # CUDA tensor on GPU, shape (n, dim)
        return embeddings.detach().cpu().numpy()  # CPU copy only if explicitly requested

    def embed_chunks(
        self, chunks: List[dict], batch_size: Optional[int] = None
    ) -> List[dict]:
        """
        Embed all chunks and add embeddings to chunk dictionaries as GPU tensors.

        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for embedding (uses default if None)

        Returns:
            Updated chunks with 'embedding' key containing CUDA tensor
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return chunks

        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings as GPU tensor - NO CPU COPY
        embeddings_tensor = self.embed_batch(texts, batch_size=batch_size, return_tensor=True)
        # embeddings_tensor is (n_chunks, dim) on GPU

        # Add embeddings to chunks as GPU tensors (zero-copy)
        # Store as tensor directly - downstream consumers can use .to("cuda", non_blocking=True)
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings_tensor[i]  # GPU tensor slice, zero-copy view

        logger.info(f"Added GPU embeddings to {len(chunks)} chunks")
        return chunks

    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embeddings.

        Returns:
            Embedding dimension
        """
        if self._embedding_dim is None:
            if not self.model:
                raise RuntimeError("Model not loaded")
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        return self._embedding_dim

    def similarity(
        self,
        embedding1: Union[torch.Tensor, np.ndarray],
        embedding2: Union[torch.Tensor, np.ndarray],
    ) -> float:
        """
        Compute cosine similarity between two embeddings on GPU.

        Args:
            embedding1: First embedding (tensor or numpy)
            embedding2: Second embedding (tensor or numpy)

        Returns:
            Similarity score between -1 and 1
        """
        # Ensure tensors on GPU
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1).to(self.device, non_blocking=True)
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2).to(self.device, non_blocking=True)

        # Normalize vectors on GPU
        emb1_norm = embedding1 / (embedding1.norm() + 1e-8)
        emb2_norm = embedding2 / (embedding2.norm() + 1e-8)
        return float(torch.dot(emb1_norm, emb2_norm).item())

    def most_similar(
        self,
        query_embedding: Union[torch.Tensor, np.ndarray],
        embeddings: Union[torch.Tensor, np.ndarray],
        top_k: int = 5,
    ) -> List[int]:
        """
        Find indices of most similar embeddings using GPU batched operations.

        Args:
            query_embedding: Query embedding (tensor or numpy)
            embeddings: Array of embeddings to search (tensor or numpy)
            top_k: Number of top results

        Returns:
            List of indices of top-k similar embeddings
        """
        # Ensure tensors on GPU
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding).to(
                self.device, non_blocking=True
            )
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).to(self.device, non_blocking=True)

        # Normalize vectors on GPU - batched
        query_norm = query_embedding / (query_embedding.norm() + 1e-8)
        embeddings_norm = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)

        # Compute similarities on GPU - single matmul
        similarities = torch.matmul(embeddings_norm, query_norm)

        # Get top-k indices on GPU
        top_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices

        return top_indices.tolist()


def get_embedding_generator(
    model_name: Optional[str] = None, device: Optional[str] = None
) -> EmbeddingGenerator:
    """Factory function to create embedding generator with config defaults."""
    from config import config

    return EmbeddingGenerator(
        model_name=model_name or config.EMBEDDING_MODEL,
        device=device or config.EMBEDDING_DEVICE,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )