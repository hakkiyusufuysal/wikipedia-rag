"""
Wikipedia RAG Assistant — Flask API + Chat Dashboard.

Endpoints:
  GET  /              — Chat dashboard (single-page HTML/CSS/JS)
  POST /chat          — Ask a question, get back answer + sources + timings
  POST /chat/stream   — SSE streaming variant (token-by-token)
  GET  /stats         — Index stats (total chunks, by type)
  GET  /health        — Ollama + model availability check
  POST /reset         — Clear chat history (server is stateless; this is a UI hook)

Concurrency: Flask threaded mode handles multiple chat requests in parallel.
The vector store is read-only at query time — Chroma is thread-safe for reads.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from rag.generator import DEFAULT_MODEL, health_check
from rag.pipeline import answer, answer_stream
from rag.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"

app = Flask(__name__, static_folder="static")
store = VectorStore(CHROMA_DIR)


@app.get("/")
def dashboard():
    return send_from_directory("static", "index.html")


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k", 5))
    model = data.get("model", DEFAULT_MODEL)

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        result = answer(query, store, top_k=top_k, model=model)
        return jsonify(result)
    except RuntimeError as e:
        # Ollama unreachable
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Error in /chat")
        return jsonify({"error": f"Internal error: {e}"}), 500


@app.post("/chat/stream")
def chat_stream():
    """Server-Sent Events stream of tokens."""
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k", 5))
    model = data.get("model", DEFAULT_MODEL)

    if not query:
        return jsonify({"error": "query is required"}), 400

    def event_stream():
        try:
            for evt in answer_stream(query, store, top_k=top_k, model=model):
                yield f"data: {json.dumps(evt)}\n\n"
        except RuntimeError as e:
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
        except Exception as e:
            logger.exception("Error in /chat/stream")
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/stats")
def stats():
    try:
        s = store.stats()
        s["model"] = DEFAULT_MODEL
        return jsonify(s)
    except Exception as e:
        logger.exception("stats failed")
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({
        "ollama": health_check(),
        "vector_store": store.stats(),
    })


@app.post("/reset")
def reset():
    """Stateless server — just acknowledge so the UI can clear its history."""
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8091))
    print(f"\n  Wikipedia RAG Assistant")
    print(f"  Dashboard: http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
