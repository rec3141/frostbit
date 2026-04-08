#!/usr/bin/env python3
"""
Filter validated questions and merge with existing 286 hard questions.

Filtering rules:
  - Drop if either model got it wrong
  - Drop if either model flagged a quality issue
  - Keep "not_in_source" flags (artifact of limited context window)

Then merge with the existing hard question bank and output a unified exam bank.

Usage:
    python3 filter_and_merge.py
"""

import json
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
VALIDATED_PATH = PROJECT_DIR / "validation_results" / "all_validated.json"
HARD_QUESTIONS_PATH = PROJECT_DIR / "exam_compact.json"
OUTPUT_PATH = PROJECT_DIR / "exam_bank_full.json"
STATS_PATH = PROJECT_DIR / "filter_stats.json"

# Flags to ignore during filtering (artifacts, not quality issues)
IGNORABLE_FLAGS = {"not_in_source", "missing source material"}


def load_validated():
    with open(VALIDATED_PATH) as f:
        return json.load(f)


def load_hard_questions():
    """Load existing 286 hard questions and normalize format."""
    with open(HARD_QUESTIONS_PATH) as f:
        raw = json.load(f)

    questions = []
    for q in raw:
        questions.append({
            "question": q["q"],
            "A": q["a"].split(") ", 1)[1] if ") " in q["a"] else q["a"],
            "B": q["b"].split(") ", 1)[1] if ") " in q["b"] else q["b"],
            "C": q["c"].split(") ", 1)[1] if ") " in q["c"] else q["c"],
            "D": q["d"].split(") ", 1)[1] if ") " in q["d"] else q["d"],
            "E": q["e"].split(") ", 1)[1] if ") " in q["e"] else q["e"],
            "answer": q["answer"],
            "difficulty": "hard",
            "original_id": q["id"],
        })
    return questions


def is_ignorable_flag(flag: str | None) -> bool:
    if not flag:
        return True
    return flag.lower().strip() in IGNORABLE_FLAGS


def filter_questions(validated: list[dict]) -> tuple[list[dict], dict]:
    """Filter validated questions. Returns (kept, stats)."""
    kept = []
    stats = {
        "total": len(validated),
        "dropped_wrong_haiku": 0,
        "dropped_wrong_sonnet": 0,
        "dropped_flagged_haiku": 0,
        "dropped_flagged_sonnet": 0,
        "kept": 0,
        "kept_easy": 0,
        "kept_medium": 0,
    }

    for q in validated:
        # Check correctness
        if q.get("haiku_answer", "?") != q["answer"]:
            stats["dropped_wrong_haiku"] += 1
            continue
        if q.get("sonnet_answer", "?") != q["answer"]:
            stats["dropped_wrong_sonnet"] += 1
            continue

        # Check flags
        haiku_flag = q.get("haiku_flag")
        sonnet_flag = q.get("sonnet_flag")
        if not is_ignorable_flag(haiku_flag):
            stats["dropped_flagged_haiku"] += 1
            continue
        if not is_ignorable_flag(sonnet_flag):
            stats["dropped_flagged_sonnet"] += 1
            continue

        # Clean up: remove validation fields, keep core question + difficulty ratings
        difficulty = q.get("generation_difficulty") or q.get("difficulty", "unknown")
        clean = {
            "question": q["question"],
            "A": q["A"],
            "B": q["B"],
            "C": q["C"],
            "D": q["D"],
            "E": q["E"],
            "answer": q["answer"],
            "difficulty": difficulty,
            "haiku_difficulty": q.get("haiku_difficulty"),
            "sonnet_difficulty": q.get("sonnet_difficulty"),
        }
        kept.append(clean)

    stats["kept"] = len(kept)
    stats["kept_easy"] = sum(1 for q in kept if q["difficulty"] == "easy")
    stats["kept_medium"] = sum(1 for q in kept if q["difficulty"] == "medium")
    stats["kept_hard"] = sum(1 for q in kept if q["difficulty"] == "hard")
    return kept, stats


def merge_and_renumber(filtered: list[dict], hard: list[dict]) -> list[dict]:
    """Merge filtered new questions with hard questions, shuffle, and renumber."""
    combined = filtered + hard

    # Shuffle within difficulty tiers to mix page sources
    rng = random.Random(3140)
    easy = [q for q in combined if q["difficulty"] == "easy"]
    medium = [q for q in combined if q["difficulty"] == "medium"]
    hard_qs = [q for q in combined if q["difficulty"] == "hard"]
    rng.shuffle(easy)
    rng.shuffle(medium)
    rng.shuffle(hard_qs)

    # Combine: easy, medium, hard
    merged = easy + medium + hard_qs
    for i, q in enumerate(merged, 1):
        q["id"] = i
        q.pop("original_id", None)

    return merged


def main():
    print("Loading validated questions...", flush=True)
    validated = load_validated()
    print(f"  {len(validated)} questions loaded", flush=True)

    print("Filtering...", flush=True)
    filtered, stats = filter_questions(validated)

    print(f"\nFilter results:")
    print(f"  Total input:          {stats['total']}")
    print(f"  Dropped (Haiku wrong): {stats['dropped_wrong_haiku']}")
    print(f"  Dropped (Sonnet wrong):{stats['dropped_wrong_sonnet']}")
    print(f"  Dropped (Haiku flag):  {stats['dropped_flagged_haiku']}")
    print(f"  Dropped (Sonnet flag): {stats['dropped_flagged_sonnet']}")
    print(f"  Kept:                  {stats['kept']}")
    print(f"    Easy:   {stats['kept_easy']}")
    print(f"    Medium: {stats['kept_medium']}")
    print(f"    Hard:   {stats['kept_hard']}")

    print(f"\nLoading existing hard questions...", flush=True)
    hard = load_hard_questions()
    print(f"  {len(hard)} hard questions loaded", flush=True)

    print("Merging and renumbering...", flush=True)
    merged = merge_and_renumber(filtered, hard)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(merged, f, indent=2)

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    difficulty_counts = {}
    for q in merged:
        d = q["difficulty"]
        difficulty_counts[d] = difficulty_counts.get(d, 0) + 1

    print(f"\n{'='*60}")
    print(f"Final exam bank: {len(merged)} questions")
    for d, count in sorted(difficulty_counts.items()):
        print(f"  {d:8s}: {count}")
    print(f"  Saved: {OUTPUT_PATH}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
