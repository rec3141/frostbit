#!/usr/bin/env python3
"""
Streaming validator: watches for generated chunk files and validates them as they appear.

Run alongside generate_questions.py. Picks up new chunk_*.json files from
generated_questions/, validates each question against Haiku + Sonnet with
source page images, and saves results to validation_results/.

Polls every few seconds for new chunks. Exits when no new chunks appear
for --timeout seconds (default 120).

Usage:
    source .venv/bin/activate
    python3 validate_streaming.py                  # default settings
    python3 validate_streaming.py --workers 5      # parallel validation workers
    python3 validate_streaming.py --timeout 300    # wait longer for new chunks
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
PDF_PATH = PROJECT_DIR / "mbteee_report.pdf"
CHUNKS_DIR = PROJECT_DIR / "generated_questions"
OUTPUT_DIR = PROJECT_DIR / "validation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

HAIKU_MODEL = "anthropic/claude-haiku-4.5"
SONNET_MODEL = "anthropic/claude-sonnet-4"
MAX_RETRIES = 3
RETRY_DELAY = 5
POLL_INTERVAL = 5  # seconds between polls for new chunks


def load_api_key():
    env_path = PROJECT_DIR / ".env.local"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env.local or environment", flush=True)
        sys.exit(1)
    return api_key


VALIDATION_PROMPT = """You are a careful exam validator. You will receive source text from a textbook,
followed by multiple-choice questions supposedly based on that text.

For EACH question:
1. Answer it using ONLY the information in the source text
2. Flag any quality issues (or null if none)
3. Rate its difficulty from 1 (trivial) to 5 (very hard)

## Flag reasons (use these exact strings or null):
- "wrong_answer" — the stated correct answer is wrong
- "ambiguous" — multiple answers could be correct
- "not_in_source" — answer cannot be determined from the provided text
- "poor_distractors" — distractors are too obvious or nonsensical
- "unclear_wording" — question is confusing or poorly written
- "trivial" — question tests irrelevant details (ISBN, authorship, formatting)

## Difficulty scale:
1 = Trivial recall, answer is stated verbatim in one sentence
2 = Easy recall, requires locating a specific fact
3 = Medium, requires understanding relationships or comparing concepts
4 = Hard, requires synthesis across multiple passages or subtle distinctions
5 = Very hard, requires deep comprehension and careful reasoning

## Output format (strict JSON):
```json
{
  "1": {"answer": "B", "flag": null, "difficulty": 3},
  "2": {"answer": "D", "flag": "ambiguous", "difficulty": 4}
}
```

Return ONLY the JSON object. No commentary."""


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------
def extract_text(first_page: int, last_page: int) -> str:
    """Extract text from PDF pages using pdftotext. Uses PDF page numbers."""
    cmd = [
        "pdftotext", "-layout",
        "-f", str(first_page), "-l", str(last_page),
        str(PDF_PATH), "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_questions(client: OpenAI, model: str, questions: list[dict],
                       pdf_first_page: int, pdf_last_page: int) -> dict | None:
    """Send questions + source text to a model for validation.

    Uses PDF page numbers (not report page numbers) to extract the correct text.
    """
    source_text = extract_text(pdf_first_page, pdf_last_page)
    if not source_text.strip():
        return None

    prompt = f"## Source text (PDF pages {pdf_first_page}-{pdf_last_page}):\n\n"
    prompt += source_text + "\n\n"
    prompt += "## Questions to validate:\n\n"
    for i, q in enumerate(questions, 1):
        # Strip page refs from question text to avoid anchoring bias
        clean_q = re.sub(r'\s*\((?:pp?\.\s*[\d,\s\-/]+(?:\s*(?:and|,)\s*[\d\-/]+)*)\)', '', q["question"]).strip()
        prompt += f"Question {i}: {clean_q}\n"
        for letter in "ABCDE":
            prompt += f"  {letter}) {q[letter]}\n"
        prompt += f"  Stated correct answer: {q['answer']}\n\n"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": VALIDATION_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return parse_validation(response)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                wait = RETRY_DELAY * (attempt + 1)
                time.sleep(wait)
            else:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
    return None


def parse_validation(response) -> dict | None:
    """Parse validation JSON. New format: {"1": {"answer": "B", "flag": null, "difficulty": 3}, ...}"""
    text = response.choices[0].message.content
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]
    try:
        data = json.loads(text)
        # Normalize: ensure each entry has answer, flag, difficulty
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                result[k] = {
                    "answer": v.get("answer", "?"),
                    "flag": v.get("flag"),
                    "difficulty": v.get("difficulty"),
                }
        return result
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Chunk validation
# ---------------------------------------------------------------------------
def validate_chunk(client: OpenAI, chunk_path: Path) -> dict:
    """Validate all questions in a chunk file. Returns validation result dict."""
    with open(chunk_path) as f:
        chunk = json.load(f)

    # Parse PDF page range from chunk metadata (e.g. "10-14")
    page_range = chunk.get("pages", "1-5")
    pdf_first, pdf_last = map(int, page_range.split("-"))

    all_questions = []
    for difficulty in ["easy", "medium", "hard"]:
        for q in chunk.get(difficulty, []):
            q["_difficulty"] = difficulty
            all_questions.append(q)

    haiku_result = validate_questions(client, HAIKU_MODEL, all_questions,
                                      pdf_first, pdf_last)
    sonnet_result = validate_questions(client, SONNET_MODEL, all_questions,
                                       pdf_first, pdf_last)

    validated = []
    for i, q in enumerate(all_questions):
        idx = str(i + 1)
        h = haiku_result.get(idx, {}) if haiku_result else {}
        s = sonnet_result.get(idx, {}) if sonnet_result else {}
        entry = {
            "question": q["question"],
            "A": q["A"], "B": q["B"], "C": q["C"], "D": q["D"], "E": q["E"],
            "answer": q["answer"],
            "generation_difficulty": q["_difficulty"],
            "source_chunk": chunk_path.name,
            "haiku_answer": h.get("answer", "?"),
            "sonnet_answer": s.get("answer", "?"),
            "haiku_flag": h.get("flag"),
            "sonnet_flag": s.get("flag"),
            "haiku_difficulty": h.get("difficulty"),
            "sonnet_difficulty": s.get("difficulty"),
        }
        validated.append(entry)

    h_ok = "ok" if haiku_result else "FAIL"
    s_ok = "ok" if sonnet_result else "FAIL"
    return {"questions": validated, "haiku": h_ok, "sonnet": s_ok}


def result_path_for(chunk_path: Path) -> Path:
    """Validation result filename for a given chunk."""
    return OUTPUT_DIR / f"val_{chunk_path.stem}.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stream-validate generated chunks")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Exit after N seconds with no new chunks (default: 120)")
    args = parser.parse_args()

    api_key = load_api_key()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    validated_chunks = set()
    last_new_chunk_time = time.time()
    total_questions = 0
    total_clean = 0

    print(f"Watching {CHUNKS_DIR} for new chunks (workers={args.workers}, "
          f"timeout={args.timeout}s)\n", flush=True)

    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending_futures = {}

    try:
        while True:
            # Find chunks that need validation
            chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.json"))
            new_chunks = []
            for cf in chunk_files:
                if cf.name not in validated_chunks and not result_path_for(cf).exists():
                    new_chunks.append(cf)

            # Also mark already-validated ones
            for cf in chunk_files:
                if result_path_for(cf).exists():
                    validated_chunks.add(cf.name)

            # Submit new chunks
            for cf in new_chunks:
                validated_chunks.add(cf.name)
                last_new_chunk_time = time.time()
                future = pool.submit(validate_chunk, client, cf)
                pending_futures[future] = cf

            # Collect completed futures
            done = [f for f in pending_futures if f.done()]
            for future in done:
                cf = pending_futures.pop(future)
                try:
                    result = future.result()
                    out_path = result_path_for(cf)
                    with open(out_path, "w") as f:
                        json.dump(result, f, indent=2)

                    n_q = len(result["questions"])
                    n_clean = sum(
                        1 for q in result["questions"]
                        if q["haiku_answer"] == q["answer"]
                        and q["sonnet_answer"] == q["answer"]
                        and not q.get("haiku_flag")
                        and not q.get("sonnet_flag")
                    )
                    total_questions += n_q
                    total_clean += n_clean
                    print(f"  {cf.name}: {n_clean}/{n_q} clean "
                          f"(haiku={result['haiku']} sonnet={result['sonnet']}) "
                          f"[total: {total_clean}/{total_questions}]", flush=True)
                except Exception as e:
                    print(f"  {cf.name}: ERROR {e}", flush=True)

            # Check timeout
            if not pending_futures and (time.time() - last_new_chunk_time) > args.timeout:
                print(f"\nNo new chunks for {args.timeout}s, finishing up.", flush=True)
                break

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nInterrupted. Waiting for in-flight validations...", flush=True)

    # Wait for remaining futures
    for future in as_completed(pending_futures):
        cf = pending_futures[future]
        try:
            result = future.result()
            out_path = result_path_for(cf)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            n_q = len(result["questions"])
            n_clean = sum(
                1 for q in result["questions"]
                if q["haiku_answer"] == q["answer"]
                and q["sonnet_answer"] == q["answer"]
                and not q.get("haiku_flag")
                and not q.get("sonnet_flag")
            )
            total_questions += n_q
            total_clean += n_clean
            print(f"  {cf.name}: {n_clean}/{n_q} clean", flush=True)
        except Exception as e:
            print(f"  {cf.name}: ERROR {e}", flush=True)

    pool.shutdown(wait=False)

    # Combine all validation results
    all_validated = []
    for vf in sorted(OUTPUT_DIR.glob("val_chunk_*.json")):
        with open(vf) as f:
            data = json.load(f)
        all_validated.extend(data["questions"])

    if all_validated:
        combined_path = OUTPUT_DIR / "all_validated.json"
        with open(combined_path, "w") as f:
            json.dump(all_validated, f, indent=2)

        clean = sum(
            1 for q in all_validated
            if q["haiku_answer"] == q["answer"]
            and q["sonnet_answer"] == q["answer"]
            and not q.get("haiku_flag")
            and not q.get("sonnet_flag")
        )
        print(f"\n{'='*60}")
        print(f"Validation summary:")
        print(f"  Total validated: {len(all_validated)}")
        print(f"  Clean (pass):    {clean}")
        print(f"  Saved:           {combined_path}")
        print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
