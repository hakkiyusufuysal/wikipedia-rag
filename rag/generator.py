"""
Answer generator — calls local Ollama HTTP API (no SDK, just stdlib urllib).

Why direct HTTP and not the `ollama` Python package?
  - Assignment says "language native functionality... rather than fully featured
    libraries that do the core work." HTTP + JSON is the language-native way.
  - The Python `ollama` package is a thin wrapper anyway.
  - We want streaming for the UI; managing the SSE-style stream ourselves is
    straightforward with urllib.

Prompting strategy:
  - System prompt: act strictly as a context-grounded answerer.
  - User prompt template: numbered context blocks + question + explicit
    "If not in context, say I don't know" instruction.
  - Temperature 0.2 — low to reduce hallucination.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Iterator

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about famous people and places using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context. Do not invent facts.
2. If the answer is not in the context, reply exactly: "I don't know based on the provided context."
3. Be concise. 2-4 sentences for simple questions. Up to 6 for comparisons.
4. Do not mention "the context" or "based on the documents" in your answer — just answer naturally.
5. If the question is about a person, focus on what they did and why they are famous.
6. If the question is about a place, focus on location, significance, and history."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Format the user prompt with retrieved context blocks."""
    if not chunks:
        return (
            f"No context was retrieved for this question.\n\n"
            f"Question: {query}\n\n"
            f"Reply: \"I don't know based on the provided context.\""
        )

    context_blocks = []
    for i, ch in enumerate(chunks, 1):
        # Truncate very long chunks to keep prompt size reasonable
        text = ch["text"]
        if len(text) > 800:
            text = text[:800] + "..."
        context_blocks.append(f"[Source {i} — {ch['title']} ({ch['type']})]\n{text}")

    context_str = "\n\n".join(context_blocks)
    return f"""Context:
{context_str}

Question: {query}

Answer:"""


def generate(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> str:
    """Generate an answer (non-streaming). Returns the full text."""
    prompt = build_prompt(query, chunks)
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 400,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError as e:
        logger.error(f"Ollama request failed: {e}")
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Is `ollama serve` running and `{model}` pulled?"
        ) from e


def generate_stream(
    query: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
) -> Iterator[str]:
    """
    Generate answer as a stream. Yields tokens as they arrive.
    Used by the dashboard for ChatGPT-style streaming UX.
    """
    prompt = build_prompt(query, chunks)
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": 400,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                evt = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            token = evt.get("response", "")
            if token:
                yield token
            if evt.get("done"):
                break


def health_check(model: str = DEFAULT_MODEL) -> dict:
    """Check whether Ollama is reachable and the model is available."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["name"] for m in data.get("models", [])]
            return {
                "ollama_reachable": True,
                "available_models": models,
                "target_model": model,
                "model_available": any(model in m for m in models),
            }
    except Exception as e:
        return {
            "ollama_reachable": False,
            "error": str(e),
            "target_model": model,
        }
