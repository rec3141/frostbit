#!/usr/bin/env python3
"""
Filter validated questions and output a clean exam bank.

Filtering rules:
  - Drop if either model got it wrong
  - Drop if either model flagged a quality issue
  - Keep "not_in_source" flags (artifact of limited context window)

Usage:
    python3 filter_merge.py --exam-dir /path/to/exam
"""

import json
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
VALIDATED_PATH = None
OUTPUT_PATH = None
STATS_PATH = None


def load_env(*search_dirs):
    for d in list(search_dirs) + [PROJECT_DIR]:
        env_path = Path(d) / ".env.local"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())

# Flags to ignore during filtering (artifacts, not quality issues)
IGNORABLE_FLAGS = {"not_in_source", "missing source material"}


def load_validated():
    with open(VALIDATED_PATH) as f:
        return json.load(f)



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



def main():
    global VALIDATED_PATH, OUTPUT_PATH, STATS_PATH

    import argparse
    parser = argparse.ArgumentParser(description="Filter validated questions")
    parser.add_argument("--exam-dir", type=Path, required=True,
                        help="Path to exam directory")
    args = parser.parse_args()

    exam_dir = args.exam_dir.resolve()
    VALIDATED_PATH = exam_dir / "validation_results" / "all_validated.json"
    OUTPUT_PATH = exam_dir / "bank.json"
    STATS_PATH = exam_dir / "filter_stats.json"

    load_env(exam_dir)

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

    # Shuffle within difficulty tiers and renumber
    rng = random.Random(3140)
    easy = [q for q in filtered if q["difficulty"] == "easy"]
    medium = [q for q in filtered if q["difficulty"] == "medium"]
    hard_qs = [q for q in filtered if q["difficulty"] == "hard"]
    rng.shuffle(easy)
    rng.shuffle(medium)
    rng.shuffle(hard_qs)
    merged = easy + medium + hard_qs
    for i, q in enumerate(merged, 1):
        q["id"] = i

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
