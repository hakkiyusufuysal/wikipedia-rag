"""
Local embedding model using sentence-transformers.

Choice: `all-MiniLM-L6-v2`
  - 384-dim dense embeddings
  - 80 MB model size (downloads once, cached locally)
  - ~14k sentences/sec on CPU
  - Pretrained on 1B sentence pairs — strong semantic similarity
  - Good fit for short Wikipedia chunks (~120 words / ~150 tokens)

Why not nomic-embed-text via Ollama?
  - sentence-transformers is more battle-tested for retrieval
  - No need to keep Ollama up just for embeddings
  - MiniLM is significantly faster (smaller model)

The first call downloads the model (~80MB). Subsequent calls are instant.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _get_model():
    """Load the embedding model once per process (cached)."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Return embeddings for a list of texts. Each is a list of 384 floats."""
    model = _get_model()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs.tolist()


def embed_one(text: str) -> list[float]:
    """Convenience wrapper for single-text embedding."""
    return embed([text])[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    e = embed(["Albert Einstein was a physicist.", "The Eiffel Tower is in Paris."])
    print(f"Got {len(e)} embeddings of dim {len(e[0])}")
    print(f"First 5 dims of first vector: {e[0][:5]}")
