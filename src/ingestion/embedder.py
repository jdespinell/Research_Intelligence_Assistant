"""
Embedding generation using sentence-transformers.
Batched for efficiency; cached model instance.
"""

from functools import lru_cache

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    model = SentenceTransformer(settings.embedding_model)
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    logger.success(f"Embedding model loaded ✅ (dim={dim})")
    return model


def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,   # L2-normalize for cosine similarity
        convert_to_numpy=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.
    sentence-transformers supports query prompts for asymmetric retrieval.
    """
    model = _get_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding


def get_embedding_dim() -> int:
    return _get_model().get_sentence_embedding_dimension()
