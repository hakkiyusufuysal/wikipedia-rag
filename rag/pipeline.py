"""
End-to-end RAG pipeline: query → classify → retrieve → generate.

Wraps the 5 runtime stages into one function for the chat API.
Each call returns full timing breakdown for observability (shown in UI).
"""

from __future__ import annotations

import time
from pathlib import Path

from .generator import DEFAULT_MODEL, generate, generate_stream
from .retriever import retrieve
from .vectorstore import VectorStore


def answer(
    query: str,
    store: VectorStore,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Run the full RAG pipeline. Returns:
      {
        query, classification, type_filter_applied,
        chunks: [...],
        answer: str,
        latency_ms: {classify, embed, search, generate, total},
        model: str,
      }
    """
    t0 = time.perf_counter()
    retrieval = retrieve(query, store, top_k=top_k)

    t_gen = time.perf_counter()
    answer_text = generate(query, retrieval["chunks"], model=model)
    gen_ms = round((time.perf_counter() - t_gen) * 1000, 2)

    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    latency = retrieval["latency_ms"].copy()
    latency["generate"] = gen_ms
    latency["total"] = total_ms

    return {
        "query": query,
        "classification": retrieval["classification"],
        "type_filter_applied": retrieval["type_filter_applied"],
        "chunks": retrieval["chunks"],
        "answer": answer_text,
        "latency_ms": latency,
        "model": model,
    }


def answer_stream(
    query: str,
    store: VectorStore,
    top_k: int = 5,
    model: str = DEFAULT_MODEL,
):
    """
    Streaming variant. Yields events:
      {"type": "retrieval", "data": {...}}     — once, before generation
      {"type": "token", "data": "..."}          — many, during generation
      {"type": "done", "data": {latency_ms}}    — once, at end
    """
    t0 = time.perf_counter()
    retrieval = retrieve(query, store, top_k=top_k)
    yield {"type": "retrieval", "data": retrieval}

    t_gen = time.perf_counter()
    full_answer = []
    for token in generate_stream(query, retrieval["chunks"], model=model):
        full_answer.append(token)
        yield {"type": "token", "data": token}

    gen_ms = round((time.perf_counter() - t_gen) * 1000, 2)
    total_ms = round((time.perf_counter() - t0) * 1000, 2)
    latency = retrieval["latency_ms"].copy()
    latency["generate"] = gen_ms
    latency["total"] = total_ms

    yield {
        "type": "done",
        "data": {
            "answer": "".join(full_answer).strip(),
            "latency_ms": latency,
            "model": model,
        },
    }
