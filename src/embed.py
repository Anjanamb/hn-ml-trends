"""Sentence-embedding helpers.

Uses ``sentence-transformers`` with a small, CPU-friendly model
(``all-MiniLM-L6-v2``, 384 dimensions) so notebooks stay fast on a
laptop. Swap in a larger model in one line if quality matters more
than speed.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_model(name: str = DEFAULT_MODEL):
    """Load and cache a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def embed_texts(
    texts: Iterable[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    normalise: bool = True,
) -> np.ndarray:
    """Embed a batch of texts. Returns an ``(N, dim)`` array."""
    model = get_model(model_name)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalise,
        show_progress_bar=False,
    )
