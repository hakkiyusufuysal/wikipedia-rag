"""
Document chunker — splits long Wikipedia articles into overlapping chunks.

Strategy: WORD-based fixed-size with overlap.
  - Chunks of ~120 words (~600-800 chars, ~150-180 tokens for MiniLM)
  - 30-word overlap (~25%) so context isn't cut at chunk boundaries
  - Sentence-aware: try to start/end at sentence boundaries when possible

Why words and not characters?
  - Embedding models tokenize by subword; word boundaries align better than
    arbitrary char cuts (which often split mid-word and lose meaning).
  - 120 words ≈ 1 paragraph of Wikipedia prose, a natural unit of meaning.

Why 25% overlap?
  - Empirical: 0% overlap loses cross-chunk context (e.g., a sentence about
    Einstein's birthplace continues into next chunk's "...Germany in 1879").
  - >50% overlap inflates the index without quality gains.

Why not paragraph splits?
  - Wikipedia paragraphs vary 30–500 words; embedding quality degrades on
    very long chunks (information dilution). Fixed size + overlap is more
    predictable.
"""

from __future__ import annotations

import re

CHUNK_SIZE_WORDS = 120
CHUNK_OVERLAP_WORDS = 30


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter (regex). Stdlib only."""
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size words.

    Uses sliding window. Each chunk overlaps the previous by `overlap` words.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return chunks


def chunk_document(
    title: str,
    type_: str,
    content: str,
) -> list[dict]:
    """
    Chunk a single document. Returns list of:
      {chunk_id, title, type, chunk_index, text, char_count, word_count}
    """
    raw_chunks = chunk_text(content)
    out = []
    for i, chunk in enumerate(raw_chunks):
        out.append({
            "chunk_id": f"{title.replace(' ', '_')}__{i:03d}",
            "title": title,
            "type": type_,
            "chunk_index": i,
            "text": chunk,
            "char_count": len(chunk),
            "word_count": len(chunk.split()),
        })
    return out


if __name__ == "__main__":
    # Smoke test
    sample = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_text(sample)
    print(f"500-word doc → {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        n = len(c.split())
        print(f"  chunk {i}: {n} words")
