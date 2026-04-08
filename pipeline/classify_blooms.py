#!/usr/bin/env python3
"""
Classify each question by Bloom's Taxonomy level using Haiku via OpenRouter.

Adds a "blooms" field to each question in exam_bank_calibrated.json (or exam_bank_full.json).

Usage:
    source .venv/bin/activate
    python3 classify_blooms.py
    python3 classify_blooms.py --workers 3
"""

import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

PROJECT_DIR = Path(__file__).parent.parent
EXAM_PATH = None
OUTPUT_PATH = None
CHECKPOINT_PATH = None

HAIKU_MODEL = "anthropic/claude-haiku-4.5"
BATCH_SIZE = 15
MAX_RETRIES = 3
RETRY_DELAY = 5

BLOOMS_PROMPT = """Classify each multiple-choice question by Bloom's Taxonomy level.

Bloom's Taxonomy levels:
1. Remember — recall facts, definitions, terms
2. Understand — explain concepts, summarize, interpret
3. Apply — use information in a new situation, solve problems
4. Analyze — compare, contrast, distinguish, examine relationships
5. Evaluate — judge, justify, critique, assess
6. Create — synthesize, design, construct new ideas

For each question, return its Bloom's level (1-6) and the level name.

Return strict JSON:
```json
{
  "1": {"level": 2, "name": "Understand"},
  "2": {"level": 4, "name": "Analyze"}
}
```

Return ONLY the JSON object. No commentary."""


def load_env(*search_dirs):
    for d in list(search_dirs) + [PROJECT_DIR]:
        env_path = Path(d) / ".env.local"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def load_api_key(*search_dirs):
    load_env(*search_dirs)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY", flush=True)
        sys.exit(1)
    return api_key


def strip_page_ref(text):
    return re.sub(r'\s*\((?:pp?\.\s*[\d,\s\-/]+(?:\s*(?:and|,)\s*[\d\-/]+)*)\)', '', text).strip()


def build_prompt(batch):
    lines = [BLOOMS_PROMPT, ""]
    for i, q in enumerate(batch, 1):
        clean_q = strip_page_ref(q["question"])
        lines.append(f"Q{i}: {clean_q}")
        for ch in "ABCDE":
            lines.append(f"  {ch}) {q[ch]}")
        lines.append("")
    return "\n".join(lines)


def parse_blooms(text, n):
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]

    results = {}
    try:
        data = json.loads(text)
        for k, v in data.items():
            qnum = int(k)
            if 1 <= qnum <= n and isinstance(v, dict):
                level = v.get("level")
                name = v.get("name", "")
                if isinstance(level, int) and 1 <= level <= 6:
                    results[qnum] = {"level": level, "name": name}
    except (json.JSONDecodeError, ValueError):
        pass
    return results


def main():
    global EXAM_PATH, OUTPUT_PATH, CHECKPOINT_PATH

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam-dir", type=Path, required=True,
                        help="Path to exam directory")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    exam_dir = args.exam_dir.resolve()
    EXAM_PATH = exam_dir / "bank.json"
    OUTPUT_PATH = exam_dir / "bank.json"
    CHECKPOINT_PATH = exam_dir / "blooms_checkpoint.json"

    api_key = load_api_key(exam_dir)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    with open(EXAM_PATH) as f:
        exam = json.load(f)
    print(f"Loaded {len(exam)} questions from {EXAM_PATH.name}", flush=True)

    # Load checkpoint
    classified = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            classified = json.load(f)
        print(f"Resuming from checkpoint ({len(classified)} already classified)", flush=True)

    # Build batches of unclassified questions
    batches = []
    batch = []
    for q in exam:
        qid = str(q["id"])
        if qid in classified:
            continue
        batch.append(q)
        if len(batch) == BATCH_SIZE:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)

    print(f"{len(batches)} batches to classify, using {args.workers} workers\n", flush=True)

    def process_batch(batch):
        prompt = build_prompt(batch)
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=HAIKU_MODEL,
                    max_tokens=2048,
                    timeout=60,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = resp.choices[0].message.content if resp.choices else ""
                if content:
                    return parse_blooms(content, len(batch)), batch
            except Exception as e:
                if "429" in str(e):
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    time.sleep(RETRY_DELAY)
        return {}, batch

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_batch, b): b for b in batches}
        for future in as_completed(futures):
            results, batch = future.result()
            for i, q in enumerate(batch, 1):
                qid = str(q["id"])
                if i in results:
                    classified[qid] = results[i]
            done += 1

            # Checkpoint every 10 batches
            if done % 10 == 0 or done == len(batches):
                with open(CHECKPOINT_PATH, "w") as f:
                    json.dump(classified, f)
                print(f"  {done}/{len(batches)} batches, {len(classified)} classified", flush=True)

    # Apply to exam
    for q in exam:
        qid = str(q["id"])
        if qid in classified:
            q["blooms_level"] = classified[qid]["level"]
            q["blooms_name"] = classified[qid]["name"]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(exam, f, indent=2)

    # Summary
    levels = {}
    for q in exam:
        bl = q.get("blooms_level")
        if bl:
            name = q.get("blooms_name", "?")
            key = f"{bl}. {name}"
            levels[key] = levels.get(key, 0) + 1

    print(f"\n{'='*50}")
    print(f"Bloom's Taxonomy Classification")
    print(f"{'='*50}")
    for k in sorted(levels.keys()):
        print(f"  {k:25s} {levels[k]:4d}")
    classified_count = sum(1 for q in exam if q.get("blooms_level"))
    print(f"\n  Classified: {classified_count}/{len(exam)}")
    print(f"  Saved: {OUTPUT_PATH}", flush=True)

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()


if __name__ == "__main__":
    main()
