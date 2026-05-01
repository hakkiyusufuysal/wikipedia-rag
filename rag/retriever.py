"""
Retriever — orchestrates classify → embed → vector-search.

Returns top-k most relevant chunks for a query, with type filtering applied
based on the classifier's decision.
"""

from __future__ import annotations

import logging
import time

from .classifier import classify
from .embedder import embed_one
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


def retrieve(query: str, store: VectorStore, top_k: int = 5) -> dict:
    """
    Retrieve relevant chunks for a query.

    Returns:
        {
          "query": str,
          "classification": {type, matched_people, matched_places, reason},
          "chunks": [{chunk_id, title, type, text, score, chunk_index}],
          "latency_ms": {classify, embed, search, total},
        }
    """
    timings = {}
    t0 = time.perf_counter()

    # 1. Classify
    t1 = time.perf_counter()
    classification = classify(query)
    timings["classify"] = round((time.perf_counter() - t1) * 1000, 2)

    # 2. Embed
    t2 = time.perf_counter()
    query_emb = embed_one(query)
    timings["embed"] = round((time.perf_counter() - t2) * 1000, 2)

    # 3. Decide filter
    type_filter = None
    if classification["type"] in ("person", "place"):
        type_filter = classification["type"]
    # type='both' → no filter, search all chunks

    # 4. Vector search
    t3 = time.perf_counter()
    chunks = store.query(query_emb, top_k=top_k, type_filter=type_filter)
    timings["search"] = round((time.perf_counter() - t3) * 1000, 2)

    timings["total"] = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "query": query,
        "classification": classification,
        "type_filter_applied": type_filter,
        "chunks": chunks,
        "latency_ms": timings,
    }
