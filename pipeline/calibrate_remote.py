#!/usr/bin/env python3
"""
Calibrate question difficulty using free OpenRouter models.

Same logic as calibrate_difficulty.py but calls OpenRouter API instead of local LM Studio.
Runs models concurrently since they're remote.

Usage:
    source .venv/bin/activate
    python3 calibrate_openrouter.py                    # run all free models
    python3 calibrate_openrouter.py --workers 5        # concurrent workers
    python3 calibrate_openrouter.py --score-only       # just recompute scores
"""

import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

PROJECT_DIR = Path(__file__).parent.parent
EXAM_PATH = None
RESULTS_DIR = None
OUTPUT_PATH = None
BATCH_SIZE = 10
MAX_RETRIES = 5
RETRY_DELAY = 15

FREE_MODELS = [
    # Tested and working (return actual answers)
    "arcee-ai/trinity-large-preview:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "minimax/minimax-m2.5:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
]


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
        print("ERROR: Set OPENROUTER_API_KEY in .env.local or environment", flush=True)
        sys.exit(1)
    return api_key


def strip_page_ref(text):
    return re.sub(r'\s*\((?:pp?\.\s*[\d,\s\-/]+(?:\s*(?:and|,)\s*[\d\-/]+)*)\)', '', text).strip()


def build_batch_prompt(batch):
    lines = [
        "Answer each multiple-choice question, rate its difficulty, and flag any quality issues.",
        "",
        "For each question, provide:",
        "- answer: your answer (A-E)",
        "- difficulty: rating 1-5:",
        "  1 = Trivial recall, answer stated verbatim",
        "  2 = Easy recall, requires locating a specific fact",
        "  3 = Medium, requires understanding relationships or comparing concepts",
        "  4 = Hard, requires synthesis or subtle distinctions",
        "  5 = Very hard, requires deep comprehension and careful reasoning",
        "- flag: quality issue or null. Use one of:",
        '  "ambiguous" — multiple answers could be correct',
        '  "poor_distractors" — distractors are too obvious or nonsensical',
        '  "unclear_wording" — question is confusing or poorly written',
        '  "trivial" — question tests irrelevant details',
        '  null — no issues',
        "",
        "Return strict JSON:",
        '{"1": {"answer": "B", "difficulty": 3, "flag": null}, "2": {"answer": "D", "difficulty": 1, "flag": "trivial"}, ...}',
        "",
    ]
    for i, q in enumerate(batch, 1):
        clean_q = strip_page_ref(q["question"])
        lines.append(f"Q{i}: {clean_q}")
        for ch in "ABCDE":
            lines.append(f"  {ch}) {q[ch]}")
        lines.append("")
    lines.append("Return ONLY the JSON object.")
    return "\n".join(lines)


def parse_response(text, n):
    """Parse JSON response with answers, difficulty ratings, and flags."""
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start != -1 and brace_end != -1:
            text = text[brace_start:brace_end + 1]

    answers = {}
    difficulties = {}
    flags = {}
    try:
        data = json.loads(text)
        for k, v in data.items():
            qnum = int(k)
            if 1 <= qnum <= n and isinstance(v, dict):
                ans = v.get("answer", "")
                if isinstance(ans, str) and len(ans) == 1 and ans.upper() in "ABCDE":
                    answers[qnum] = ans.upper()
                diff = v.get("difficulty")
                if isinstance(diff, (int, float)) and 1 <= diff <= 5:
                    difficulties[qnum] = int(diff)
                flag = v.get("flag")
                if flag and isinstance(flag, str):
                    flags[qnum] = flag
        if answers:
            return answers, difficulties, flags
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: parse Q1: B style (no difficulty/flags from fallback)
    answers = {}
    for m in re.finditer(r'Q(\d+)\s*[:.=)\-]\s*\**([A-Ea-e])\**', text):
        qnum = int(m.group(1))
        if 1 <= qnum <= n:
            answers[qnum] = m.group(2).upper()
    if len(answers) >= n * 0.5:
        return answers, difficulties, flags

    for m in re.finditer(r'(?:^|\n)\s*(\d+)\s*[:.)\-]\s*\**([A-Ea-e])\b', text):
        qnum = int(m.group(1))
        if 1 <= qnum <= n and qnum not in answers:
            answers[qnum] = m.group(2).upper()
    if len(answers) >= n * 0.5:
        return answers, difficulties, flags

    letters = re.findall(r'\b([A-E])\b', text)
    for i, letter in enumerate(letters[:n], 1):
        if i not in answers:
            answers[i] = letter.upper()
    return answers, difficulties, flags


def call_api(client, model, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                timeout=60,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else None
            if content:
                return content
            # Empty response — retry
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            err = str(e).lower()
            wait = RETRY_DELAY * (attempt + 1)
            if "rate" in err or "429" in err:
                print(f"    {model}: 429, waiting {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"    {model}: error ({str(e)[:60]})", flush=True)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
    return ""


def run_model(client, model, exam):
    safe_name = model.replace("/", "_").replace(":", "_")
    out_path = RESULTS_DIR / f"{safe_name}.json"
    checkpoint_path = RESULTS_DIR / f"{safe_name}.partial.json"

    if out_path.exists():
        print(f"  Skipping {model} (already done)", flush=True)
        return

    # Resume from checkpoint if available
    all_answers = {}
    start_batch = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        all_answers = checkpoint.get("answers", {})
        start_batch = checkpoint.get("next_batch", 0)
        print(f"  Resuming {model} from batch {start_batch + 1}...", flush=True)
    else:
        print(f"  Starting {model}...", flush=True)

    all_difficulties = {}
    all_flags = {}
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        all_difficulties = checkpoint.get("difficulties", {})
        all_flags = checkpoint.get("flags", {})

    total = len(exam)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in range(start_batch * BATCH_SIZE, total, BATCH_SIZE):
        batch = exam[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        prompt = build_batch_prompt(batch)
        response = call_api(client, model, prompt)
        parsed, diffs, batch_flags = parse_response(response, len(batch))
        time.sleep(2)  # pace requests to avoid rate limits

        for j, q in enumerate(batch, 1):
            global_idx = q["id"]
            model_ans = parsed.get(j, "?")
            all_answers[str(global_idx)] = model_ans
            if j in diffs:
                all_difficulties[str(global_idx)] = diffs[j]
            if j in batch_flags:
                all_flags[str(global_idx)] = batch_flags[j]

        # Save checkpoint every 5 batches
        if batch_num % 5 == 0 or batch_num == total_batches:
            correct = sum(1 for q in exam if all_answers.get(str(q["id"])) == q["answer"])
            checkpoint = {
                "model": model,
                "answers": all_answers,
                "difficulties": all_difficulties,
                "flags": all_flags,
                "next_batch": batch_num,
                "correct": correct,
                "total": total,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f)

        if batch_num % 10 == 0 or batch_num == total_batches:
            correct = sum(1 for q in exam if all_answers.get(str(q["id"])) == q["answer"])
            print(f"    {model}: batch {batch_num}/{total_batches}, "
                  f"running {correct}/{batch_start + len(batch)}", flush=True)

    correct = sum(1 for q in exam if all_answers.get(str(q["id"])) == q["answer"])
    pct = 100 * correct / total if total else 0
    print(f"  {model}: {correct}/{total} = {pct:.1f}%", flush=True)

    result = {
        "model": model,
        "correct": correct,
        "total": total,
        "pct": round(pct, 1),
        "answers": all_answers,
        "difficulties": all_difficulties,
        "flags": all_flags,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def compute_ensemble_scores(exam):
    model_results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if ".partial." in path.name:
            continue
        with open(path) as f:
            model_results.append(json.load(f))

    if not model_results:
        print("No model results found!", flush=True)
        return exam

    n_models = len(model_results)
    print(f"\nComputing ensemble scores from {n_models} models:", flush=True)
    for mr in model_results:
        print(f"  {mr['model']:55s} {mr['pct']:.1f}%", flush=True)

    for q in exam:
        qid = str(q["id"])
        wrong = 0
        tested = 0
        diff_ratings = []
        for mr in model_results:
            ans = mr["answers"].get(qid)
            if ans and ans != "?":
                tested += 1
                if ans != q["answer"]:
                    wrong += 1
            # Collect subjective difficulty ratings
            diffs = mr.get("difficulties", {})
            d = diffs.get(qid)
            if d and isinstance(d, (int, float)):
                diff_ratings.append(d)

        if tested > 0:
            q["difficulty_score"] = round(wrong / tested, 3)
            q["models_tested"] = tested
            q["models_wrong"] = wrong
        else:
            q["difficulty_score"] = None
            q["models_tested"] = 0
            q["models_wrong"] = 0

        if diff_ratings:
            q["ensemble_difficulty_rating"] = round(sum(diff_ratings) / len(diff_ratings), 2)
            q["ensemble_difficulty_n"] = len(diff_ratings)
        else:
            q["ensemble_difficulty_rating"] = None
            q["ensemble_difficulty_n"] = 0

        # Aggregate flags across models
        flag_counts = {}
        for mr in model_results:
            model_flags = mr.get("flags", {})
            f = model_flags.get(qid)
            if f:
                flag_counts[f] = flag_counts.get(f, 0) + 1
        if flag_counts:
            q["ensemble_flags"] = flag_counts
            q["ensemble_flag_count"] = sum(flag_counts.values())
        else:
            q["ensemble_flags"] = None
            q["ensemble_flag_count"] = 0

    return exam


def main():
    global EXAM_PATH, RESULTS_DIR, OUTPUT_PATH

    import argparse
    parser = argparse.ArgumentParser(description="Calibrate question difficulty via OpenRouter")
    parser.add_argument("--exam-dir", type=Path, required=True,
                        help="Path to exam directory")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--score-only", action="store_true",
                        help="Just recompute scores from existing results")
    cli_args = parser.parse_args()

    exam_dir = cli_args.exam_dir.resolve()
    EXAM_PATH = exam_dir / "bank.json"
    RESULTS_DIR = exam_dir / "calibration_results"
    RESULTS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH = exam_dir / "bank.json"

    with open(EXAM_PATH) as f:
        exam = json.load(f)
    print(f"Loaded {len(exam)} questions", flush=True)

    score_only = cli_args.score_only
    workers = cli_args.workers

    if not score_only:
        api_key = load_api_key(exam_dir)
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        # Filter to models not yet done
        to_run = []
        for model in FREE_MODELS:
            safe_name = model.replace("/", "_").replace(":", "_")
            if not (RESULTS_DIR / f"{safe_name}.json").exists():
                to_run.append(model)

        print(f"\n{len(to_run)} models to run, {len(FREE_MODELS) - len(to_run)} already done, "
              f"using {workers} workers\n", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_model, client, m, exam): m for m in to_run}
            for future in as_completed(futures):
                model = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  ERROR {model}: {e}", flush=True)

    # Compute ensemble scores from ALL results (local + remote)
    exam = compute_ensemble_scores(exam)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(exam, f, indent=2)

    # Summary by tier
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
                  f"avg difficulty: {avg:.3f}", flush=True)

    all_scores = [q["difficulty_score"] for q in exam if q.get("difficulty_score") is not None]
    if all_scores:
        print(f"\n  Overall: {len(exam)} questions, "
              f"avg: {sum(all_scores)/len(all_scores):.3f}, "
              f"min: {min(all_scores):.3f}, max: {max(all_scores):.3f}", flush=True)

    print(f"\n  Saved: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
