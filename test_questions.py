"""
Smoke test — runs all assignment example questions end-to-end and prints results.

Usage:
    python test_questions.py            # full set
    python test_questions.py --quick    # just 3 questions
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rag.pipeline import answer
from rag.vectorstore import VectorStore

QUESTIONS = [
    # People
    ("people", "Who was Albert Einstein and what is he known for"),
    ("people", "What did Marie Curie discover"),
    ("people", "Why is Nikola Tesla famous"),
    ("people", "Compare Lionel Messi and Cristiano Ronaldo"),
    ("people", "What is Frida Kahlo known for"),
    # Places
    ("places", "Where is the Eiffel Tower located"),
    ("places", "Why is the Great Wall of China important"),
    ("places", "What is Machu Picchu"),
    ("places", "What was the Colosseum used for"),
    ("places", "Where is Mount Everest"),
    # Mixed
    ("mixed", "Which famous place is located in Turkey"),
    ("mixed", "Which person is associated with electricity"),
    ("mixed", "Compare Albert Einstein and Nikola Tesla"),
    ("mixed", "Compare the Eiffel Tower and the Statue of Liberty"),
    # Failures
    ("failure", "Who is the president of Mars"),
    ("failure", "Tell me about a random unknown person John Doe"),
]

QUICK = [
    ("people", "Who was Albert Einstein and what is he known for"),
    ("places", "Where is the Eiffel Tower located"),
    ("failure", "Who is the president of Mars"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    chroma_dir = Path(__file__).parent / "data" / "chroma"
    store = VectorStore(chroma_dir)
    stats = store.stats()
    print(f"Vector store: {stats['total_chunks']} chunks "
          f"({stats['person_chunks']} person + {stats['place_chunks']} place)\n")

    if stats["total_chunks"] == 0:
        print("ERROR: index is empty. Run `python build_index.py` first.")
        sys.exit(1)

    questions = QUICK if args.quick else QUESTIONS
    print(f"Running {len(questions)} questions...\n")
    print("=" * 80)

    for i, (category, q) in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] [{category.upper()}] {q}")
        t0 = time.perf_counter()
        try:
            result = answer(q, store)
            elapsed = time.perf_counter() - t0

            print(f"  Classification: {result['classification']['type']} "
                  f"({result['classification']['reason']})")
            print(f"  Top sources: {[c['title'] for c in result['chunks'][:3]]}")
            print(f"  Latency: total {elapsed:.1f}s "
                  f"(retrieve {result['latency_ms'].get('embed', 0) + result['latency_ms'].get('search', 0):.0f}ms, "
                  f"generate {result['latency_ms'].get('generate', 0):.0f}ms)")
            print(f"  Answer: {result['answer']}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 80)
    print(f"Done. Tested {len(questions)} questions.")


if __name__ == "__main__":
    main()
