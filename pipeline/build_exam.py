#!/usr/bin/env python3
"""
Build a complete adaptive exam from a PDF.

Drop a PDF into an exam directory and run this script. It will:
1. Render pages as images and generate easy/medium/hard questions (Sonnet via OpenRouter)
2. Validate questions with Haiku + Sonnet (text-based, with source pages)
3. Filter out bad questions and merge
4. Classify by Bloom's taxonomy (Haiku)
5. Calibrate difficulty via free OpenRouter models
6. Output a ready-to-serve exam bank

Usage:
    source .venv/bin/activate
    python3 -m pipeline.build_exam exams/my_exam/ --pdf source.pdf
    python3 -m pipeline.build_exam exams/my_exam/            # if PDF already in dir
    python3 -m pipeline.build_exam exams/my_exam/ --resume   # resume interrupted build
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PIPELINE_DIR = Path(__file__).parent


def find_pdf(exam_dir: Path, pdf_arg: str | None) -> Path:
    if pdf_arg:
        src = Path(pdf_arg)
        if not src.exists():
            print(f"ERROR: PDF not found: {src}")
            sys.exit(1)
        dest = exam_dir / "reference.pdf"
        if not dest.exists():
            shutil.copy2(src, dest)
            print(f"Copied {src} -> {dest}")
        return dest
    # Look for existing PDF
    pdfs = list(exam_dir.glob("*.pdf"))
    if pdfs:
        ref = exam_dir / "reference.pdf"
        if not ref.exists():
            shutil.copy2(pdfs[0], ref)
        return ref
    print("ERROR: No PDF found. Provide --pdf or put a PDF in the exam directory.")
    sys.exit(1)


def ensure_exam_config(exam_dir: Path, pdf_path: Path):
    config_path = exam_dir / "exam.json"
    if config_path.exists():
        return
    exam_id = exam_dir.name
    config = {
        "id": exam_id,
        "title": exam_id.replace("_", " ").replace("-", " ").title(),
        "subtitle": "Adaptive Exam",
        "description": f"Adaptive exam generated from {pdf_path.name}",
        "bank_file": "bank.json",
        "reference_pdf": "reference.pdf",
        "pdf_page_offset": 0,
        "cat_config": {
            "initial_theta": -2.0,
            "se_threshold": 0.3,
            "min_questions": 5,
            "recency_half_life": 8,
        },
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Created {config_path}")


def run_step(name, cmd, cwd=None):
    print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
    result = subprocess.run(cmd, cwd=cwd or str(PROJECT_DIR))
    if result.returncode != 0:
        print(f"WARNING: {name} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Build a complete adaptive exam from a PDF")
    parser.add_argument("exam_dir", type=str, help="Path to exam directory (e.g. exams/my_exam/)")
    parser.add_argument("--pdf", type=str, default=None, help="Path to source PDF")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted build")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent workers for generation/validation")
    parser.add_argument("--skip-calibrate", action="store_true", help="Skip model calibration")
    args = parser.parse_args()

    exam_dir = Path(args.exam_dir).resolve()
    exam_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    pdf_path = find_pdf(exam_dir, args.pdf)
    ensure_exam_config(exam_dir, pdf_path)
    print(f"Exam directory: {exam_dir}")
    print(f"PDF: {pdf_path}")

    venv_python = str(PROJECT_DIR / ".venv" / "bin" / "python3")
    if not Path(venv_python).exists():
        venv_python = "python3"

    gen_dir = exam_dir / "generated_questions"
    val_dir = exam_dir / "validation_results"
    bank_path = exam_dir / "bank.json"

    exam_arg = ["--exam-dir", str(exam_dir)]

    # Step 1: Generate questions
    if not bank_path.exists() or args.resume:
        run_step("Step 1: Generating questions", [
            venv_python, str(PIPELINE_DIR / "generate.py"),
            *exam_arg, "--workers", str(args.workers),
        ])

    # Step 2: Validate (streaming — picks up chunks as they appear)
    all_gen = gen_dir / "all_generated_questions.json"
    if all_gen.exists() and not (val_dir / "all_validated.json").exists():
        run_step("Step 2: Validating questions", [
            venv_python, str(PIPELINE_DIR / "validate.py"),
            *exam_arg, "--workers", str(args.workers),
            "--timeout", "60",
        ])

    # Step 3: Filter and merge
    if (val_dir / "all_validated.json").exists() and not bank_path.exists():
        run_step("Step 3: Filtering and merging", [
            venv_python, str(PIPELINE_DIR / "filter_merge.py"),
            *exam_arg,
        ])

    # Step 4: Classify Bloom's taxonomy
    if bank_path.exists():
        run_step("Step 4: Classifying Bloom's taxonomy", [
            venv_python, str(PIPELINE_DIR / "classify_blooms.py"),
            *exam_arg, "--workers", "1",
        ])

    # Step 5: Calibrate difficulty (optional)
    if not args.skip_calibrate and bank_path.exists():
        run_step("Step 5: Calibrating difficulty (remote models)", [
            venv_python, str(PIPELINE_DIR / "calibrate_remote.py"),
            *exam_arg, "--workers", "1",
        ])

    # Summary
    if bank_path.exists():
        with open(bank_path) as f:
            bank = json.load(f)
        print(f"\n{'='*60}")
        print(f"EXAM READY: {len(bank)} questions")
        print(f"  Directory: {exam_dir}")
        print(f"  Bank:      {bank_path}")
        print(f"  Start server: python3 app.py --port 8080")
        print(f"{'='*60}")
    else:
        print("\nExam build incomplete. Run with --resume to continue.")


if __name__ == "__main__":
    main()
