"""
One-shot orchestrator: ingest → chunk → embed → store.

Usage:
    python build_index.py            # full build (40 entities)
    python build_index.py --reset    # delete vector store first
    python build_index.py --skip-ingest  # use existing wiki.db, just re-embed

Run this once after cloning. Takes ~3-5 minutes (mostly Wikipedia + embed time).
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

from rag.chunker import chunk_document
from rag.embedder import embed
from rag.entities import all_entities
from rag.ingest import ingest
from rag.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_index")


DATA_DIR = Path(__file__).parent / "data"
WIKI_DB = DATA_DIR / "wiki.db"
CHROMA_DIR = DATA_DIR / "chroma"


def build(reset: bool = False, skip_ingest: bool = False) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    stats = {"ingested": 0, "documents": 0, "chunks": 0, "embeddings": 0}

    # ── Step 1: Ingest from Wikipedia ──
    if not skip_ingest:
        logger.info("Step 1/4: Fetching Wikipedia articles...")
        result = ingest(all_entities(), WIKI_DB)
        stats["ingested"] = result["ingested"]
        if result["failed"]:
            for title, err in result["failed"]:
                logger.warning(f"  failed: {title} ({err})")
        logger.info(f"  Ingested {result['ingested']}/{result['total']} entities")
    else:
        logger.info("Step 1/4: SKIPPED (--skip-ingest)")

    # ── Step 2: Reset vector store if requested ──
    store = VectorStore(CHROMA_DIR)
    if reset:
        logger.info("Step 2/4: Resetting vector store...")
        store.reset()
    else:
        logger.info("Step 2/4: Using existing vector store (use --reset to wipe)")

    # ── Step 3: Chunk all documents ──
    logger.info("Step 3/4: Chunking documents...")
    conn = sqlite3.connect(str(WIKI_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT title, type, content FROM documents").fetchall()
    conn.close()
    stats["documents"] = len(rows)

    all_chunks: list[dict] = []
    for row in rows:
        chunks = chunk_document(row["title"], row["type"], row["content"])
        all_chunks.extend(chunks)
        logger.info(f"  {row['title']:40s} → {len(chunks)} chunks")
    stats["chunks"] = len(all_chunks)

    # ── Step 4: Embed and store (batched) ──
    logger.info(f"Step 4/4: Embedding {len(all_chunks)} chunks...")
    BATCH_SIZE = 64
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        t0 = time.perf_counter()
        embs = embed([c["text"] for c in batch])
        store.add_chunks(batch, embs)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"  batch {i // BATCH_SIZE + 1}/{(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}: "
            f"{len(batch)} chunks in {elapsed:.1f}s "
            f"({len(batch) / elapsed:.0f} chunks/s)"
        )
        stats["embeddings"] += len(batch)

    final_stats = store.stats()
    logger.info(f"Done. Vector store now has {final_stats['total_chunks']} chunks "
                f"({final_stats['person_chunks']} person + {final_stats['place_chunks']} place)")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Wipe vector store first")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Use existing data/wiki.db, only re-embed")
    args = parser.parse_args()

    if args.skip_ingest and not WIKI_DB.exists():
        logger.error(f"--skip-ingest requires {WIKI_DB} to exist")
        sys.exit(1)

    stats = build(reset=args.reset, skip_ingest=args.skip_ingest)
    print(f"\n=== Build complete ===\n{stats}")


if __name__ == "__main__":
    main()
