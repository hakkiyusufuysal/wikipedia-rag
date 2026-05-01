"""
Query classifier — decides whether a query is about a PERSON, PLACE, or BOTH.

Strategy: rule-based with two layers (fast, deterministic, explainable).
The assignment explicitly says "Keyword based or rule based approaches are acceptable."

Layer 1 — Entity name detection
  Match the exact (case-insensitive) names from our 40 known entities. If we
  see "Einstein" in the query, we know it's about a person. If we see two
  names from different types ("Compare Einstein and the Eiffel Tower"), we
  classify as BOTH.

Layer 2 — Semantic keyword fallback
  When no known entity is mentioned (e.g., "Which famous place is in Turkey"),
  we fall back to keyword cues:
    - "place", "city", "country", "located", "where" → PLACE
    - "person", "who", "born", "scientist", "inventor", "famous for" → PERSON
    - Neither matches → BOTH (let semantic search decide via no filter)

This is intentionally simple. A learned classifier would be overkill for
the question scope. The rule-based version produces explainable decisions
the dashboard can show ("classified as person because: matched 'Einstein'").
"""

from __future__ import annotations

from .entities import PEOPLE, PLACES

# Lowercase last names + full names for matching
_PERSON_NAMES = set()
for full in PEOPLE:
    _PERSON_NAMES.add(full.lower())
    parts = full.lower().split()
    if len(parts) >= 2:
        _PERSON_NAMES.add(parts[-1])  # last name
        _PERSON_NAMES.add(parts[0])  # first name

_PLACE_NAMES = set()
for full in PLACES:
    _PLACE_NAMES.add(full.lower())
    # Also index without parentheses (e.g., "Christ the Redeemer")
    if "(" in full:
        _PLACE_NAMES.add(full.split("(")[0].strip().lower())

_PERSON_KEYWORDS = {
    "who", "person", "scientist", "physicist", "artist", "actor", "actress",
    "writer", "poet", "philosopher", "inventor", "musician", "composer",
    "born", "died", "lived", "famous for", "known for", "wrote", "invented",
    "discovered", "painted", "composed", "singer", "footballer", "career",
}

_PLACE_KEYWORDS = {
    "where", "place", "city", "country", "located", "monument", "building",
    "tower", "wall", "mountain", "temple", "stadium", "river", "lake",
    "island", "park", "site", "landmark", "structure", "ruins",
}


def classify(query: str) -> dict:
    """
    Classify a query.

    Returns:
        {
          "type": "person" | "place" | "both",
          "matched_people": [...],    # entity names found in query
          "matched_places": [...],
          "reason": "..."             # human-readable explanation
        }
    """
    q = query.lower()

    # Layer 1: known entity name match
    matched_people = sorted({p for p in _PERSON_NAMES if p in q and len(p) >= 4})
    matched_places = sorted({pl for pl in _PLACE_NAMES if pl in q and len(pl) >= 4})

    if matched_people and matched_places:
        return {
            "type": "both",
            "matched_people": matched_people,
            "matched_places": matched_places,
            "reason": f"matched person names {matched_people} AND place names {matched_places}",
        }
    if matched_people:
        return {
            "type": "person",
            "matched_people": matched_people,
            "matched_places": [],
            "reason": f"matched person name(s): {matched_people}",
        }
    if matched_places:
        return {
            "type": "place",
            "matched_people": [],
            "matched_places": matched_places,
            "reason": f"matched place name(s): {matched_places}",
        }

    # Layer 2: keyword fallback
    person_hits = sum(1 for kw in _PERSON_KEYWORDS if kw in q)
    place_hits = sum(1 for kw in _PLACE_KEYWORDS if kw in q)

    if person_hits > place_hits:
        return {
            "type": "person",
            "matched_people": [],
            "matched_places": [],
            "reason": f"keyword score: person={person_hits} > place={place_hits}",
        }
    if place_hits > person_hits:
        return {
            "type": "place",
            "matched_people": [],
            "matched_places": [],
            "reason": f"keyword score: place={place_hits} > person={person_hits}",
        }
    return {
        "type": "both",
        "matched_people": [],
        "matched_places": [],
        "reason": f"no clear signal (person={person_hits}, place={place_hits}); searching all",
    }


if __name__ == "__main__":
    tests = [
        "Who was Albert Einstein and what is he known for",
        "Where is the Eiffel Tower located",
        "Compare Einstein and the Eiffel Tower",
        "Which famous place is located in Turkey",
        "Tell me about a random unknown person John Doe",
        "Compare Lionel Messi and Cristiano Ronaldo",
    ]
    for t in tests:
        r = classify(t)
        print(f'"{t}"\n  → {r["type"]}: {r["reason"]}\n')
