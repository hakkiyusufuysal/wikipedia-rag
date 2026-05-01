# Product PRD — Wikipedia RAG Assistant

## Goal
A local-only, ChatGPT-style assistant that answers questions about famous people and famous places, grounded in Wikipedia content. Must run entirely on a developer laptop with no external APIs.

## Users
- A student or developer evaluating retrieval-augmented generation (RAG)
- Anyone who wants a privacy-preserving "ask Wikipedia" tool

## Functional requirements

| # | Requirement | How met |
|---|---|---|
| F1 | Ingest Wikipedia content for ≥20 people + ≥20 places | `build_index.py` ingests 40 entities (assignment minimum + 20 extras) via Wikipedia REST API |
| F2 | Chunk documents with overlap | `rag/chunker.py` — 120-word chunks with 30-word overlap |
| F3 | Generate embeddings locally | `rag/embedder.py` — `all-MiniLM-L6-v2` runs on CPU, no API |
| F4 | Store embeddings in a vector DB | ChromaDB persistent client at `data/chroma/` |
| F5 | Classify query → person / place / both | `rag/classifier.py` — rule-based (entity match + keyword cues) |
| F6 | Retrieve top-k chunks with type filter | `rag/vectorstore.py` Chroma `where={"type": "person"}` |
| F7 | Generate answer with local LLM | `rag/generator.py` — calls Ollama HTTP API for `llama3.2:3b` |
| F8 | Answer "I don't know" when context is insufficient | System prompt explicitly instructs this; verified on failure cases |
| F9 | Show source chunks (optional) | UI has expandable "Show retrieved sources" section per answer |
| F10 | Chat UI with reset | Single-page Flask + vanilla JS at http://localhost:8091 |

## Non-functional requirements

| # | Requirement | Target | Actual |
|---|---|---|---|
| N1 | Run fully on localhost | no external API call | ✅ Wikipedia is the only outbound call (one-time, during ingest) |
| N2 | First-token latency | < 5s on M1 / Intel Mac | typically 2-4s for `llama3.2:3b` on Apple Silicon |
| N3 | Total response latency | < 30s | typically 5-12s for retrieve + generate |
| N4 | Memory footprint | < 4 GB | MiniLM ~200MB + llama3.2:3b ~2GB resident |
| N5 | No external LLM API | strict | enforced — only `localhost:11434` (Ollama) |

## Out of scope
- Multi-turn conversational memory (chat history) — explicitly listed as optional in assignment
- Streaming responses — implementation present (`/chat/stream`) but UI uses non-streaming for simplicity
- Multi-language support
- Citations beyond chunk source title

## Acceptance criteria
- [x] All 10 required people from the assignment ingested and queryable
- [x] All 10 required places from the assignment ingested and queryable
- [x] At least one example question per category (people / places / mixed / failure) returns a sensible answer
- [x] System runs end-to-end starting from a fresh clone using only the README
- [x] Answers explicitly say "I don't know based on the provided context" when retrieval misses
- [x] No outbound LLM API calls (verified by code review and Ollama-only HTTP target)

## Future work
See `recommendation.md` — covers production deployment, scaling beyond 40 entities, learned classifiers, multi-language support, and observability.
