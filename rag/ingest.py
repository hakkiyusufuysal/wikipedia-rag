"""
Wikipedia ingester — uses ONLY stdlib (urllib + json).

Strategy:
- Use Wikipedia's REST summary endpoint (`/api/rest_v1/page/summary/`) for the
  intro, then the action API (`?action=query&prop=extracts&explaintext=1`) for
  the full plain-text article body.
- Both endpoints return clean text — no HTML cleanup needed.
- Save raw text + metadata to SQLite.

This deliberately avoids the `wikipedia` Python package because the assignment
asks for "language native functionality... rather than fully featured libraries
that do the core work."
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

USER_AGENT = "WikipediaRAG/1.0 (educational; contact: hakki@example.edu)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_REST_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def _http_get_json(url: str) -> dict:
    """Fetch a URL and return parsed JSON. Single retry on transient errors."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise


def fetch_summary(title: str) -> dict:
    """Return Wikipedia summary {extract, description, url} for a title."""
    encoded = urllib.parse.quote(title.replace(" ", "_"))
    return _http_get_json(WIKI_REST_SUMMARY + encoded)


def fetch_full_extract(title: str) -> str:
    """Return the full plain-text article body for a title."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "exsectionformat": "plain",
        "redirects": "1",
        "titles": title,
    }
    url = WIKI_API + "?" + urllib.parse.urlencode(params)
    data = _http_get_json(url)
    pages = data.get("query", {}).get("pages", {})
    for _page_id, page in pages.items():
        return page.get("extract", "") or ""
    return ""


# ── Storage ────────────────────────────────────────────────────────────────


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            title TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            description TEXT,
            url TEXT,
            content TEXT NOT NULL,
            char_count INTEGER,
            ingested_at REAL
        )
    """)
    conn.commit()
    return conn


def save_document(conn: sqlite3.Connection, title: str, type_: str,
                  description: str, url: str, content: str):
    conn.execute(
        """INSERT OR REPLACE INTO documents
           (title, type, description, url, content, char_count, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (title, type_, description, url, content, len(content), time.time()),
    )
    conn.commit()


# ── Public API ─────────────────────────────────────────────────────────────


def ingest(entities: list[tuple[str, str]], db_path: Path) -> dict:
    """
    Ingest entities into SQLite.
    entities: list of (title, type) where type ∈ {'person', 'place'}.
    Returns: stats dict {ingested, failed, total}.
    """
    conn = init_db(db_path)
    ingested = 0
    failed: list[tuple[str, str]] = []

    for title, type_ in entities:
        try:
            summary = fetch_summary(title)
            full_text = fetch_full_extract(title)
            if not full_text:
                # fall back to summary extract if full text empty
                full_text = summary.get("extract", "")

            description = summary.get("description", "") or ""
            url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")

            if not full_text.strip():
                failed.append((title, "empty content"))
                logger.warning(f"✗ {title}: empty content")
                continue

            save_document(conn, title, type_, description, url, full_text)
            ingested += 1
            logger.info(f"✓ {title} ({type_}) — {len(full_text):,} chars")
            time.sleep(0.5)  # polite to Wikipedia
        except Exception as e:
            failed.append((title, str(e)))
            logger.error(f"✗ {title}: {e}")

    conn.close()
    return {"ingested": ingested, "failed": failed, "total": len(entities)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from .entities import all_entities
    db_path = Path(__file__).parent.parent / "data" / "wiki.db"
    db_path.parent.mkdir(exist_ok=True)
    stats = ingest(all_entities(), db_path)
    print(f"\n=== Ingestion complete ===")
    print(f"Ingested: {stats['ingested']}/{stats['total']}")
    if stats["failed"]:
        print(f"Failed: {len(stats['failed'])}")
        for title, err in stats["failed"]:
            print(f"  - {title}: {err}")
