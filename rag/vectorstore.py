"""
Vector store layer using ChromaDB (persistent local, no server).

Design choice: Option B — single collection with metadata filtering.

Why one collection instead of two (Option A: people_collection + places_collection)?
  - Mixed queries ("Compare Einstein and the Eiffel Tower") need to search
    both at once. Two separate collections force two queries + result merging.
  - Metadata filtering is a first-class Chroma feature; the cost is identical.
  - Simpler operational model: one persist directory, one collection to back up.
  - Adding a new entity type later (e.g., 'event', 'organization') means
    adding a metadata value, not creating a new collection.

The collection schema:
  - ids: chunk_id (e.g., "Albert_Einstein__003")
  - embeddings: 384-dim vectors from MiniLM
  - documents: chunk text (returned with results so we can build prompts)
  - metadatas: {title, type, chunk_index, char_count}
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "wiki_chunks"


class VectorStore:
    """Wraps a Chroma persistent collection."""

    def __init__(self, persist_dir: Path):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        # get_or_create — idempotent across runs
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine works best for MiniLM
        )

    def add_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        """
        Add chunks to the collection.
        Each chunk dict needs: chunk_id, title, type, chunk_index, text, char_count.
        """
        if not chunks:
            return
        self._collection.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[
                {
                    "title": c["title"],
                    "type": c["type"],
                    "chunk_index": c["chunk_index"],
                    "char_count": c["char_count"],
                }
                for c in chunks
            ],
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        type_filter: str | None = None,
    ) -> list[dict]:
        """
        Search for top-k similar chunks.

        type_filter ∈ {None, 'person', 'place'}.
        Returns list of {chunk_id, title, type, text, score, chunk_index}.
        """
        where = None
        if type_filter:
            where = {"type": type_filter}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        out = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for i in range(len(ids)):
            # cosine distance → similarity; Chroma returns distance in [0, 2]
            score = 1.0 - (dists[i] / 2.0)
            out.append({
                "chunk_id": ids[i],
                "title": metas[i]["title"],
                "type": metas[i]["type"],
                "chunk_index": metas[i]["chunk_index"],
                "text": docs[i],
                "score": round(score, 4),
            })
        return out

    def stats(self) -> dict:
        count = self._collection.count()
        # Approx breakdown by type — Chroma doesn't aggregate, so we sample
        # via a get() with where clauses. Cheap because we only count.
        people = self._collection.get(where={"type": "person"}, include=[])
        places = self._collection.get(where={"type": "place"}, include=[])
        return {
            "total_chunks": count,
            "person_chunks": len(people["ids"]),
            "place_chunks": len(places["ids"]),
        }

    def reset(self):
        """Delete the entire collection. Used for re-ingestion."""
        self._client.delete_collection(name=COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
