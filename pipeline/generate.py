#!/usr/bin/env python3
"""
Generate easy + medium + hard multiple-choice questions from PDF page images.

Renders 5-page chunks of the source PDF as images, sends them to Claude Sonnet
via OpenRouter's vision API, and asks for 5 easy, 5 medium, 5 hard per chunk.
Runs chunks concurrently for speed.

Usage:
    source .venv/bin/activate
    python3 generate_questions.py                  # run all chunks
    python3 generate_questions.py --start 31       # resume from page 31
    python3 generate_questions.py --pages 1-50     # only pages 1-50
    python3 generate_questions.py --workers 10     # concurrent workers (default 5)
"""

import base64
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config (overridden by --exam-dir)
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.parent  # frostbit root
PDF_PATH = None  # set in main()
OUTPUT_DIR = None  # set in main()

CHUNK_SIZE = 5          # pages per batch
DPI = 150               # resolution for page rendering (balance quality vs tokens)
MODEL = "anthropic/claude-sonnet-4.6"
MAX_RETRIES = 3
RETRY_DELAY = 5         # seconds

# Load API key from .env.local
def load_env(*search_dirs):
    """Load .env.local from any of the given directories or the project root."""
    for d in list(search_dirs) + [PROJECT_DIR]:
        env_path = Path(d) / ".env.local"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def load_api_key():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in .env.local or environment", flush=True)
        sys.exit(1)
    return api_key

SYSTEM_PROMPT = """You are an expert instructor creating multiple-choice exam questions
from reference material.

You will receive images of pages from the source material. Generate questions at three difficulty levels:

## EASY (5 questions)
- Direct recall of facts, definitions, or identifications from the pages
- Single-concept questions with one clearly correct answer
- Distractors should be plausible but clearly wrong to someone who read the material

## MEDIUM (5 questions)
- Require understanding relationships, comparisons, or applying concepts
- May ask about causes, consequences, or distinguishing features between regions
- Distractors should be tempting — correct-sounding but subtly wrong

## HARD (5 questions)
- Require synthesis across multiple concepts, subtle distinctions, or deep comprehension
- Distractors should be very tempting — plausible statements that are subtly incorrect
- May involve interpreting data from tables/figures, or distinguishing between closely related regions
- The correct answer should require careful reading, not just recognition

## Rules for ALL questions:
- 5 answer choices (A-E) per question
- NO "All of the above" or "None of the above" options
- Include a page reference (p. XX) at the end of the question text
- Every question must be answerable from the provided pages alone
- Make distractors specific and relevant, not obviously absurd
- If pages contain maps or figures, create questions about them
- Questions must be about the SUBJECT MATTER content of the pages, not administrative details
- DO NOT ask about publication metadata, authorship, ISBN numbers, formatting, table of contents, or acknowledgements
- DO NOT ask trivial lookup questions like "What is the publication number?" or "Who wrote the document?"
- Every question should test understanding of the subject matter presented

## Output format (strict JSON):
```json
{
  "easy": [
    {
      "question": "Question text here (p. 42)",
      "A": "First choice",
      "B": "Second choice",
      "C": "Third choice",
      "D": "Fourth choice",
      "E": "Fifth choice",
      "answer": "B"
    }
  ],
  "medium": [
    {
      "question": "Question text here (p. 43)",
      "A": "First choice",
      "B": "Second choice",
      "C": "Third choice",
      "D": "Fourth choice",
      "E": "Fifth choice",
      "answer": "D"
    }
  ],
  "hard": [
    {
      "question": "Question text here (p. 44)",
      "A": "First choice",
      "B": "Second choice",
      "C": "Third choice",
      "D": "Fourth choice",
      "E": "Fifth choice",
      "answer": "A"
    }
  ]
}
```

Return ONLY the JSON object. No commentary before or after."""


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
def render_pages(first_page: int, last_page: int, tmpdir: str) -> list[Path]:
    """Render PDF pages to PNG using pdftoppm. Returns list of image paths."""
    prefix = os.path.join(tmpdir, "page")
    cmd = [
        "pdftoppm", "-png", "-r", str(DPI),
        "-f", str(first_page), "-l", str(last_page),
        str(PDF_PATH), prefix,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    pngs = sorted(Path(tmpdir).glob("page-*.png"))
    return pngs


def encode_image(path: Path) -> str:
    """Base64-encode a PNG file."""
    return base64.standard_b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def generate_for_chunk(client: OpenAI, first_page: int, last_page: int) -> dict | None:
    """Send page images to Sonnet via OpenRouter and get questions back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        images = render_pages(first_page, last_page, tmpdir)
        if not images:
            print(f"  No images rendered for pages {first_page}-{last_page}", flush=True)
            return None

        # Build message content: images + text prompt (OpenAI vision format)
        content = []
        for img_path in images:
            b64 = encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                },
            })
        content.append({
            "type": "text",
            "text": f"These are pages {first_page}-{last_page} of the textbook. "
                    f"Generate 5 EASY, 5 MEDIUM, and 5 HARD multiple-choice questions based on this content.",
        })

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=8192,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content},
                    ],
                )
                return parse_response(response, first_page, last_page)
            except Exception as e:
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str:
                    wait = RETRY_DELAY * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                else:
                    print(f"  API error (attempt {attempt+1}): {e}", flush=True)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
    return None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_response(response, first_page: int, last_page: int) -> dict | None:
    """Extract JSON from Sonnet's response."""
    text = response.choices[0].message.content

    # Try to extract JSON from markdown code block or raw
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find raw JSON object
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error for pages {first_page}-{last_page}: {e}", flush=True)
        # Save raw response for debugging
        debug_path = OUTPUT_DIR / f"debug_p{first_page:03d}_{last_page:03d}.txt"
        debug_path.write_text(response.choices[0].message.content)
        return None

    # Validate structure
    easy = data.get("easy", [])
    medium = data.get("medium", [])
    hard = data.get("hard", [])
    if not easy and not medium and not hard:
        print(f"  No questions found for pages {first_page}-{last_page}", flush=True)
        return None

    return {
        "pages": f"{first_page}-{last_page}",
        "easy": easy,
        "medium": medium,
        "hard": hard,
    }


# ---------------------------------------------------------------------------
# Answer shuffling
# ---------------------------------------------------------------------------
def shuffle_answers(questions: list[dict], seed: int = 42) -> list[dict]:
    """Shuffle answer positions to avoid letter bias. Preserves correctness."""
    rng = random.Random(seed)
    shuffled = []
    for q in questions:
        correct_letter = q["answer"]
        correct_text = q[correct_letter]
        choices = [q[k] for k in "ABCDE"]
        rng.shuffle(choices)
        new_correct = "ABCDE"[choices.index(correct_text)]
        new_q = {
            "question": q["question"],
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "E": choices[4],
            "answer": new_correct,
        }
        shuffled.append(new_q)
    return shuffled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def get_page_count() -> int:
    """Get PDF page count via pdfinfo."""
    result = subprocess.run(
        ["pdfinfo", str(PDF_PATH)], capture_output=True, text=True, check=True
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":")[1].strip())
    raise RuntimeError("Could not determine page count")


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description="Generate easy/medium MC questions from PDF")
    parser.add_argument("--exam-dir", type=str, required=True, help="Path to exam directory")
    parser.add_argument("--start", type=int, default=1, help="Start page (default: 1)")
    parser.add_argument("--end", type=int, default=None, help="End page (default: last page)")
    parser.add_argument("--pages", type=str, default=None, help="Page range, e.g. '1-50'")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Pages per batch")
    parser.add_argument("--dpi", type=int, default=DPI, help="Render resolution")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent API workers (default: 5)")
    return parser.parse_args()


def main():
    global PDF_PATH, OUTPUT_DIR, DPI

    args = parse_args()
    DPI = args.dpi
    chunk_size = args.chunk_size

    exam_dir = Path(args.exam_dir).resolve()
    PDF_PATH = exam_dir / "reference.pdf"
    OUTPUT_DIR = exam_dir / "generated_questions"
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}", flush=True)
        sys.exit(1)

    load_env(exam_dir)
    api_key = load_api_key()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    total_pages = get_page_count()
    print(f"PDF has {total_pages} pages", flush=True)

    # Determine page range
    if args.pages:
        start, end = map(int, args.pages.split("-"))
    else:
        start = args.start
        end = args.end or total_pages

    print(f"Generating questions for pages {start}-{end} (chunks of {chunk_size})", flush=True)

    all_easy = []
    all_medium = []
    all_hard = []
    chunks_done = 0
    chunks_failed = 0

    # Build list of chunks to process
    chunks = []
    for first_page in range(start, end + 1, chunk_size):
        last_page = min(first_page + chunk_size - 1, end)
        chunks.append((first_page, last_page))

    # Load already-completed chunks
    to_run = []
    for first_page, last_page in chunks:
        chunk_file = OUTPUT_DIR / f"chunk_p{first_page:03d}_{last_page:03d}.json"
        if chunk_file.exists():
            print(f"  Skipping pages {first_page}-{last_page} (already done)", flush=True)
            with open(chunk_file) as f:
                data = json.load(f)
            all_easy.extend(data.get("easy", []))
            all_medium.extend(data.get("medium", []))
            all_hard.extend(data.get("hard", []))
            chunks_done += 1
        else:
            to_run.append((first_page, last_page))

    print(f"\n{len(to_run)} chunks to generate, {chunks_done} already done, "
          f"using {args.workers} workers\n", flush=True)

    def process_chunk(pages):
        first_page, last_page = pages
        chunk_file = OUTPUT_DIR / f"chunk_p{first_page:03d}_{last_page:03d}.json"
        result = generate_for_chunk(client, first_page, last_page)
        if result:
            with open(chunk_file, "w") as f:
                json.dump(result, f, indent=2)
        return first_page, last_page, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_chunk, c): c for c in to_run}
        for future in as_completed(futures):
            first_page, last_page, result = future.result()
            if result:
                n_easy = len(result.get("easy", []))
                n_med = len(result.get("medium", []))
                n_hard = len(result.get("hard", []))
                print(f"  Pages {first_page}-{last_page}: "
                      f"{n_easy} easy, {n_med} medium, {n_hard} hard", flush=True)
                all_easy.extend(result.get("easy", []))
                all_medium.extend(result.get("medium", []))
                all_hard.extend(result.get("hard", []))
                chunks_done += 1
            else:
                print(f"  Pages {first_page}-{last_page}: FAILED", flush=True)
                chunks_failed += 1

    # Shuffle answers
    print(f"\nShuffling answers...", flush=True)
    all_easy = shuffle_answers(all_easy, seed=2026)
    all_medium = shuffle_answers(all_medium, seed=3140)
    all_hard = shuffle_answers(all_hard, seed=4200)

    # Assign IDs and difficulty tags
    questions = []
    next_id = 1
    for q in all_easy:
        q["id"] = next_id
        q["difficulty"] = "easy"
        questions.append(q)
        next_id += 1
    for q in all_medium:
        q["id"] = next_id
        q["difficulty"] = "medium"
        questions.append(q)
        next_id += 1
    for q in all_hard:
        q["id"] = next_id
        q["difficulty"] = "hard"
        questions.append(q)
        next_id += 1

    # Save combined output
    out_path = OUTPUT_DIR / "all_generated_questions.json"
    with open(out_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {chunks_done} chunks processed, {chunks_failed} failed")
    print(f"  Easy:   {len(all_easy)} questions")
    print(f"  Medium: {len(all_medium)} questions")
    print(f"  Hard:   {len(all_hard)} questions")
    print(f"  Total:  {len(questions)} questions")
    print(f"  Saved:  {out_path}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
