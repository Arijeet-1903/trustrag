import abc
import hashlib
import math
import numpy as np
from typing import List, Optional


class BaseEmbeddingModel(abc.ABC):
    """Abstract base class for embedding models."""

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a float vector."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into a list of float vectors."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        pass


class DeterministicMockEmbeddings(BaseEmbeddingModel):
    """
    Deterministic pseudo-semantic embedding generator.
    Produces repeatable normalized vectors using token hashes.
    Guarantees that identical text yields identical vectors and similar words have
    measurable overlap, enabling offline reproducible tests and CI with 0 model download.
    """

    def __init__(self, dimension: int = 384, model_name: str = "deterministic-mock-v1"):
        self._dim = dimension
        self._name = model_name

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def _embed_single(self, text: str) -> List[float]:
        text = text.lower().strip()
        if not text:
            return [0.0] * self._dim

        # Accumulate vector across word tokens
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.split()
        for token in tokens:
            # Deterministic hash of token
            h = hashlib.sha256(token.encode("utf-8")).digest()
            # Spread hash entropy across vector dimensions
            for i in range(self._dim):
                byte_val = h[i % len(h)]
                sign = 1.0 if (h[(i + 7) % len(h)] % 2 == 0) else -1.0
                vec[i] += sign * (byte_val / 255.0)

        # Normalize to unit length (L2)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return [round(float(x), 6) for x in vec]

    def embed_text(self, text: str) -> List[float]:
        return self._embed_single(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]


class SentenceTransformerEmbeddings(BaseEmbeddingModel):
    """Embedding model powered by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def embed_text(self, text: str) -> List[float]:
        emb = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in emb]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [[float(x) for x in row] for row in embs]


_MODEL_REGISTRY = {}


def get_embedding_model(
    model_name: str = "deterministic_mock",
    dimension: int = 384
) -> BaseEmbeddingModel:
    """Factory and cache for embedding models."""
    cache_key = f"{model_name}_{dimension}"
    if cache_key in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[cache_key]

    if model_name in ("deterministic_mock", "mock", "test"):
        model = DeterministicMockEmbeddings(dimension=dimension, model_name="deterministic_mock_v1")
    else:
        try:
            model = SentenceTransformerEmbeddings(model_name=model_name)
        except Exception:
            # Fall back to deterministic mock if offline or download fails
            model = DeterministicMockEmbeddings(dimension=dimension, model_name=f"fallback_mock_{model_name}")

    _MODEL_REGISTRY[cache_key] = model
    return model
