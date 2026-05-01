# Wikipedia RAG Assistant

A local-only ChatGPT-style assistant that answers questions about famous people and places using **only** retrieved Wikipedia context. No external LLM API.

**Stack:** Python 3.12 · Ollama (`llama3.2:3b`) · sentence-transformers (`all-MiniLM-L6-v2`) · ChromaDB · Flask + vanilla JS

> Built for **Project 3** of BLG483E. Combines Project 1 (retrieval) and Project 2 (AI workflows) into a complete RAG application.

---

## How it works

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   User question                                                     │
│        │                                                            │
│        ▼                                                            │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐  │
│   │ Classify │───▶│  Embed   │───▶│ Retrieve │───▶│   Generate   │  │
│   │ (rule-   │    │ (MiniLM) │    │ (Chroma  │    │   (Ollama    │  │
│   │  based)  │    │  384-dim │    │  cosine) │    │ llama3.2:3b) │  │
│   └──────────┘    └──────────┘    └──────────┘    └──────────────┘  │
│        │                                │                  │        │
│        ▼                                ▼                  ▼        │
│   person/place/both          top-5 chunks +          grounded       │
│   filter decision            metadata                answer         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

Each query goes through 5 runtime stages, each timed and shown in the UI:

| Stage | Module | What it does |
|---|---|---|
| **1. Classify** | `rag/classifier.py` | Rule-based: detects whether the question is about a person, place, or both |
| **2. Embed** | `rag/embedder.py` | Encodes the query into a 384-dim vector using MiniLM (local, on CPU) |
| **3. Retrieve** | `rag/vectorstore.py` | Chroma cosine-similarity search, filtered by classified type |
| **4. Generate** | `rag/generator.py` | Calls Ollama HTTP API with retrieved context as grounding |
| **5. Respond** | `app.py` | Returns answer + sources + per-stage timings to the UI |

---

## Quick start

### 1. Install dependencies

```bash
# Python 3.12 (required — sentence-transformers doesn't yet build cleanly on 3.13)
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# macOS
brew install ollama
ollama serve &           # starts the local LLM server on :11434
ollama pull llama3.2:3b  # ~2 GB download, one-time
```

For other platforms see [ollama.com/download](https://ollama.com/download).

### 3. Build the index

```bash
python build_index.py
```

This script:
- Downloads Wikipedia articles for **20 people + 20 places** (~3-5 minutes, polite 0.5s/req)
- Stores raw text in `data/wiki.db`
- Chunks each article into ~120-word overlapping chunks
- Embeds chunks with MiniLM
- Stores in `data/chroma/`

Re-runnable: `python build_index.py --reset` wipes the vector store first.

### 4. Start the chat app

```bash
python app.py
```

Open http://localhost:8091 in your browser.

---

## Example queries

Click any example in the sidebar, or try:

**People**
- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Compare Lionel Messi and Cristiano Ronaldo*

**Places**
- *Where is the Eiffel Tower located?*
- *What was the Colosseum used for?*
- *Why is the Great Wall of China important?*

**Mixed**
- *Compare Einstein and Tesla*
- *Which famous place is in Turkey?*

**Failure cases (model says "I don't know")**
- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe*

---

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| LLM | `llama3.2:3b` via Ollama | 3B fits in 4GB RAM, fast on CPU, good instruction following |
| Embedding | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dim, 80MB, ~14k sentences/sec on CPU |
| Vector DB | Chroma persistent (single collection + metadata) | One collection serves person, place, and mixed queries via `where` filtering — no duplication |
| Chunking | 120 words, 30-word overlap (25%) | Word-aligned (better than char), one paragraph of meaning, overlap preserves cross-chunk context |
| Classifier | Rule-based (entity name match + keyword cues) | Deterministic, explainable, sub-millisecond — assignment explicitly allows |
| Web framework | Flask + vanilla HTML/CSS/JS | No build step, single-file UI, easy for instructor to run |

See `product_prd.md` for the formal spec and `recommendation.md` for production-deployment guidance.

---

## Repo layout

```
wikipedia-rag/
├── app.py                 # Flask API + chat dashboard server
├── build_index.py         # One-shot: ingest → chunk → embed → store
├── readme.md              # this file
├── product_prd.md         # PRD
├── recommendation.md      # production notes
├── requirements.txt
├── rag/
│   ├── entities.py        # 40 entity titles (20 people + 20 places)
│   ├── ingest.py          # Wikipedia API client (stdlib urllib + json)
│   ├── chunker.py         # word-based fixed-size with overlap
│   ├── embedder.py        # MiniLM wrapper (cached singleton)
│   ├── vectorstore.py     # Chroma persistent client
│   ├── classifier.py      # rule-based person/place/both detection
│   ├── retriever.py       # classify → embed → search orchestration
│   ├── generator.py       # Ollama HTTP API (stdlib urllib)
│   └── pipeline.py        # full pipeline end-to-end
├── static/
│   └── index.html         # chat UI (single file, no framework)
└── data/                  # gitignored — created by build_index.py
    ├── wiki.db            # SQLite of raw articles
    └── chroma/            # Chroma persistent store
```

---

## Limitations

- **English only** — Wikipedia API + MiniLM both English-tuned
- **Static corpus** — re-run `build_index.py` to refresh
- **Single-process** — no horizontal scaling (out of scope for educational project)
- **Rule-based classifier** — won't generalize to entities outside the 40-item list. See `recommendation.md` for production fix.

---

## License

Educational project. Wikipedia content used under CC-BY-SA.
