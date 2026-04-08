#!/usr/bin/env python3
"""
Calibrate question difficulty by running the full exam bank through local LM Studio models.

Produces an ensemble difficulty score for each question based on how many models
get it wrong (0.0 = all correct = easiest, 1.0 = all wrong = hardest).

Reads: exam_bank_full.json  (output of filter_and_merge.py)
Writes: calibration_results/<model>.json  per model
        exam_bank_calibrated.json         final bank with difficulty scores

Usage:
    # LM Studio must be running at localhost:1234
    python3 calibrate_difficulty.py                          # run all models
    python3 calibrate_difficulty.py "google/gemma-3-27b"     # run one model
    python3 calibrate_difficulty.py --score-only             # skip model runs, just compute scores
"""

import json, urllib.request, time, re, os, sys
from pathlib import Path

API_BASE = "http://127.0.0.1:1234/v1"
PROJECT_DIR = Path(__file__).parent
EXAM_PATH = PROJECT_DIR / "exam_bank_full.json"
RESULTS_DIR = PROJECT_DIR / "calibration_results"
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = PROJECT_DIR / "exam_bank_calibrated.json"
BATCH_SIZE = 10

MODELS = [
    # Reasoning models
    "nvidia/nemotron-3-nano",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistralai/ministral-3-14b-reasoning",
    "unsloth/ministral-3-3b-reasoning-2512",
    "mistralai/ministral-3-14b-reasoning-2512",
    # Standard models
    "google/gemma-3-27b",
    "google/gemma-4-26b-a4b",
    "meta/llama-3.3-70b",
    "qwen/qwen3.5-9b",
    "zai-org/glm-4.7-flash",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

REASONING_MODELS = {
    "nvidia/nemotron-3-nano",
    "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistralai/ministral-3-14b-reasoning",
    "unsloth/ministral-3-3b-reasoning-2512",
    "mistralai/ministral-3-14b-reasoning-2512",
}


def strip_page_ref(text):
    """Remove page references from question text."""
    return re.sub(r'\s*\((?:pp?\.\s*[\d,\s\-/]+(?:\s*(?:and|,)\s*[\d\-/]+)*)\)', '', text).strip()


def call_lm(model, messages, max_tokens=None, timeout=600):
    """Call LM Studio API. Returns combined content + reasoning_content."""
    if max_tokens is None:
        max_tokens = 4096 if model in REASONING_MODELS else 16384
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  API error: {e}", flush=True)
        return ""
    choice = data["choices"][0]["message"]
    parts = []
    if choice.get("reasoning_content"):
        parts.append(choice["reasoning_content"])
    if choice.get("content"):
        parts.append(choice["content"])
    return "\n".join(parts)


def build_batch_prompt(batch):
    """Format a batch of questions into a prompt."""
    lines = [
        "Answer each multiple-choice question with ONLY the letter (A-E).",
        "Format: Q1: B  Q2: D  Q3: A  ...\n",
    ]
    for i, q in enumerate(batch, 1):
        clean_q = strip_page_ref(q["question"])
        lines.append(f"Q{i}: {clean_q}")
        for ch in "ABCDE":
            lines.append(f"  {ch}) {q[ch]}")
        lines.append("")
    return "\n".join(lines)


def parse_answers(text, n):
    """Extract answer letters from model output."""
    answers = {}

    for m in re.finditer(r'Q(\d+)\s*[:.=)\-]\s*\**([A-Ea-e])\**', text):
        qnum = int(m.group(1))
        if 1 <= qnum <= n:
            answers[qnum] = m.group(2).upper()
    if len(answers) >= n * 0.5:
        return answers

    for m in re.finditer(r'(?:^|\n)\s*(\d+)\s*[:.)\-]\s*\**([A-Ea-e])\b', text):
        qnum = int(m.group(1))
        if 1 <= qnum <= n and qnum not in answers:
            answers[qnum] = m.group(2).upper()
    if len(answers) >= n * 0.5:
        return answers

    for m in re.finditer(r'\*?\*?Q(\d+)\*?\*?\s*[:.]\s*\*?\*?([A-Ea-e])\*?\*?', text):
        qnum = int(m.group(1))
        if 1 <= qnum <= n and qnum not in answers:
            answers[qnum] = m.group(2).upper()
    if len(answers) >= n * 0.5:
        return answers

    letters = re.findall(r'\b([A-E])\b', text)
    for i, letter in enumerate(letters[:n], 1):
        if i not in answers:
            answers[i] = letter.upper()
    return answers


def run_model(model, exam):
    """Run all exam questions through one model."""
    safe_name = model.replace("/", "_")
    out_path = RESULTS_DIR / f"{safe_name}.json"
    if out_path.exists():
        print(f"Skipping {model} — results already exist", flush=True)
        return

    print(f"\n{'='*60}\nRunning model: {model}\n{'='*60}", flush=True)
    all_answers = {}
    correct = 0
    total = len(exam)

    for batch_start in range(0, total, BATCH_SIZE):
        batch = exam[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} "
              f"(Q{batch_start+1}-{batch_start+len(batch)})...",
              end=" ", flush=True)

        prompt = build_batch_prompt(batch)
        messages = [{"role": "user", "content": prompt}]
        response = call_lm(model, messages)
        parsed = parse_answers(response, len(batch))

        batch_correct = 0
        for j, q in enumerate(batch, 1):
            global_idx = q["id"]
            model_ans = parsed.get(j, "?")
            all_answers[str(global_idx)] = model_ans
            if model_ans == q["answer"]:
                batch_correct += 1
                correct += 1
        print(f"{batch_correct}/{len(batch)} ({len(parsed)} parsed)", flush=True)

    pct = 100 * correct / total if total else 0
    print(f"\n{model}: {correct}/{total} = {pct:.1f}%\n", flush=True)

    result = {
        "model": model,
        "correct": correct,
        "total": total,
        "pct": round(pct, 1),
        "answers": all_answers,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)


def compute_ensemble_scores(exam):
    """Compute difficulty score from all available model results.

    Score = fraction of models that got the question wrong.
    0.0 = all models correct (easiest)
    1.0 = all models wrong (hardest)
    """
    # Load all available results
    model_results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            model_results.append(json.load(f))

    if not model_results:
        print("No model results found! Run models first.", flush=True)
        return exam

    n_models = len(model_results)
    print(f"\nComputing ensemble scores from {n_models} models:", flush=True)
    for mr in model_results:
        print(f"  {mr['model']:45s} {mr['pct']:.1f}%", flush=True)

    for q in exam:
        qid = str(q["id"])
        wrong_count = 0
        tested_count = 0
        for mr in model_results:
            ans = mr["answers"].get(qid)
            if ans and ans != "?":
                tested_count += 1
                if ans != q["answer"]:
                    wrong_count += 1

        if tested_count > 0:
            q["difficulty_score"] = round(wrong_count / tested_count, 3)
            q["models_tested"] = tested_count
            q["models_wrong"] = wrong_count
        else:
            q["difficulty_score"] = None
            q["models_tested"] = 0
            q["models_wrong"] = 0

    return exam


def main():
    with open(EXAM_PATH) as f:
        exam = json.load(f)
    print(f"Loaded {len(exam)} questions", flush=True)

    score_only = "--score-only" in sys.argv

    if not score_only:
        # Determine which models to run
        model_args = [a for a in sys.argv[1:] if not a.startswith("--")]
        targets = model_args if model_args else MODELS

        for model in targets:
            try:
                run_model(model, exam)
            except Exception as e:
                print(f"ERROR running {model}: {e}", flush=True)

    # Compute ensemble difficulty scores
    exam = compute_ensemble_scores(exam)

    # Save calibrated bank
    with open(OUTPUT_PATH, "w") as f:
        json.dump(exam, f, indent=2)

    # Print summary by difficulty tier
    print(f"\n{'='*60}\nCALIBRATION SUMMARY\n{'='*60}", flush=True)

    tiers = {}
    for q in exam:
        d = q.get("difficulty", "unknown")
        if d not in tiers:
            tiers[d] = {"count": 0, "scores": []}
        tiers[d]["count"] += 1
        if q.get("difficulty_score") is not None:
            tiers[d]["scores"].append(q["difficulty_score"])

    for tier in ["easy", "medium", "hard"]:
        if tier in tiers:
            t = tiers[tier]
            scores = t["scores"]
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {tier:8s}: {t['count']:4d} questions, "
                  f"avg difficulty score: {avg:.3f}", flush=True)

    # Overall stats
    all_scores = [q["difficulty_score"] for q in exam if q.get("difficulty_score") is not None]
    if all_scores:
        print(f"\n  Overall: {len(exam)} questions, "
              f"avg score: {sum(all_scores)/len(all_scores):.3f}, "
              f"min: {min(all_scores):.3f}, max: {max(all_scores):.3f}", flush=True)

    print(f"\n  Saved: {OUTPUT_PATH}", flush=True)

    # Model summary
    print(f"\n{'='*60}\nMODEL RESULTS\n{'='*60}", flush=True)
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            r = json.load(f)
        print(f"  {r['model']:45s} {r['correct']:3d}/{r['total']} = {r['pct']:.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
