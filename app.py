#!/usr/bin/env python3
"""
Frostbit — Computerized Adaptive Testing platform.

Multi-exam Flask app. Each exam lives in exams/<id>/ with exam.json, bank.json,
and an optional reference.pdf.

Usage:
    source .venv/bin/activate
    python3 app.py --port 8080
"""

import csv
import hmac
import io
import json
import os
import re
import secrets
import sqlite3
import subprocess
import uuid
from functools import wraps
from pathlib import Path

from flask import (
    Flask, Response, g, jsonify, redirect, render_template,
    request, send_file, session, url_for,
)

from cat_engine import (
    CATSession, Item, get_performance_summary, process_answer,
    select_next_item, should_stop, skip_item,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
EXAMS_DIR = PROJECT_DIR / "exams"
DB_PATH = PROJECT_DIR / "frostbit.db"

# Load .env.local
_env_path = PROJECT_DIR / ".env.local"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

app = Flask(__name__, template_folder=str(PROJECT_DIR / "templates"),
            static_folder=str(PROJECT_DIR / "static"))
app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_SESSION_SECURE") == "1",
)

# Stable secret so sessions survive server restarts
_secret_path = PROJECT_DIR / ".flask_secret"
if _secret_path.exists():
    app.secret_key = _secret_path.read_bytes()
else:
    app.secret_key = os.urandom(24)
    _secret_path.write_bytes(app.secret_key)
    os.chmod(str(_secret_path), 0o600)

INSTRUCTOR_AUTH = os.environ.get("INSTRUCTOR_AUTH", "")


def get_csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


@app.context_processor
def inject_template_helpers():
    return {"csrf_token": get_csrf_token}


@app.before_request
def protect_csrf():
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    if request.endpoint == "static":
        return None

    expected = get_csrf_token()
    provided = request.headers.get("X-CSRF-Token")
    if not provided:
        provided = request.form.get("csrf_token")
    if not provided and request.is_json:
        payload = request.get_json(silent=True) or {}
        provided = payload.get("csrf_token")

    if provided and hmac.compare_digest(provided, expected):
        return None

    if request.is_json or request.path.startswith("/api/") or request.path.startswith("/admin/api/"):
        return jsonify({"error": "csrf validation failed"}), 400
    return "CSRF validation failed", 400


@app.after_request
def set_security_headers(response):
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "frame-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'self'; "
        "worker-src 'self' blob: https://cdnjs.cloudflare.com",
    )
    return response


# ---------------------------------------------------------------------------
# Exam loading
# ---------------------------------------------------------------------------
class Exam:
    def __init__(self, exam_dir: Path):
        self.dir = exam_dir
        with open(exam_dir / "exam.json") as f:
            self.config = json.load(f)
        self.id = self.config["id"]
        self.title = self.config["title"]
        self.subtitle = self.config.get("subtitle", "")
        self.description = self.config.get("description", "")
        self.pdf_offset = self.config.get("pdf_page_offset", 0)
        self.published = self.config.get("published", False)
        self.show_reference = self.config.get("show_reference", True)
        self.cat_config = self.config.get("cat_config", {})
        self.items = self._load_items()
        self.items_by_id = {item.id: item for item in self.items}

    def _load_items(self) -> list[Item]:
        bank_path = self.dir / self.config.get("bank_file", "bank.json")
        if not bank_path.exists():
            return []
        with open(bank_path) as f:
            raw = json.load(f)

        items = []
        for q in raw:
            if not q.get("enabled", True):
                continue
            # Build difficulty from available signals
            signals = []
            h = q.get("haiku_difficulty")
            s = q.get("sonnet_difficulty")
            if h and s:
                signals.append(((h + s) / 2 - 3) * 1.2)
            er = q.get("ensemble_difficulty_rating")
            if er:
                signals.append((er - 3) * 1.2)
            es = q.get("difficulty_score")
            if es is not None and q.get("models_tested", 0) >= 3:
                signals.append((es * 6) - 3)
            if signals:
                diff = sum(signals) / len(signals)
            else:
                tier_defaults = {"easy": -1.5, "medium": 0.0, "hard": 1.5}
                diff = tier_defaults.get(q.get("difficulty", "medium"), 0.0)

            page_ref = ""
            m = re.search(r'\(pp?\.\s*[\d,\s\-/]+\)', q.get("question", ""))
            if m:
                page_ref = m.group(0)

            items.append(Item(
                id=q["id"],
                question=re.sub(r'\s*\(pp?\.\s*[\d,\s\-/]+\)', '', q.get("question", "")).strip(),
                choices={"A": q["A"], "B": q["B"], "C": q["C"], "D": q["D"], "E": q["E"]},
                answer=q["answer"],
                difficulty=diff,
                tier=q.get("difficulty", "unknown"),
                page_ref=page_ref,
                blooms_level=q.get("blooms_level", 0),
                blooms_name=q.get("blooms_name", ""),
            ))
        return items

    def has_pdf(self):
        pdf_path = self.dir / self.config.get("reference_pdf", "reference.pdf")
        return pdf_path.exists()

    def pdf_path(self):
        return self.dir / self.config.get("reference_pdf", "reference.pdf")


def load_exams() -> dict[str, Exam]:
    exams = {}
    if EXAMS_DIR.exists():
        for d in sorted(EXAMS_DIR.iterdir()):
            if d.is_dir() and (d / "exam.json").exists():
                try:
                    exam = Exam(d)
                    exams[exam.id] = exam
                    print(f"  Loaded exam '{exam.id}': {len(exam.items)} items", flush=True)
                except Exception as e:
                    print(f"  Error loading {d.name}: {e}", flush=True)
    return exams


def build_exam_config(exam_id: str, exam_dir: Path, existing_exam: Exam | None = None) -> dict:
    title = existing_exam.title if existing_exam else exam_id.replace("_", " ").replace("-", " ").title()
    subtitle = existing_exam.subtitle if existing_exam else "Adaptive Exam"
    description = existing_exam.description if existing_exam else ""
    pdf_name = "reference.pdf"
    if existing_exam and existing_exam.config.get("reference_pdf"):
        pdf_name = existing_exam.config["reference_pdf"]
    elif any(exam_dir.glob("*.pdf")):
        pdf_name = sorted(exam_dir.glob("*.pdf"))[0].name

    cat_config = {
        "initial_theta": -2.0,
        "se_threshold": 0.3,
        "min_questions": 5,
        "recency_half_life": 8,
        "skip_budget": 0,
    }
    if existing_exam:
        cat_config.update(existing_exam.cat_config)

    return {
        "id": exam_id,
        "title": title,
        "subtitle": subtitle,
        "description": description,
        "bank_file": "bank.json",
        "reference_pdf": pdf_name,
        "pdf_page_offset": existing_exam.pdf_offset if existing_exam else 0,
        "published": existing_exam.published if existing_exam else False,
        "show_reference": existing_exam.show_reference if existing_exam else True,
        "cat_config": cat_config,
    }


def get_pdf_page_count(pdf_path: Path) -> int | None:
    if not pdf_path.exists():
        return None
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def ensure_exam_config_file(exam: Exam) -> Path:
    config_path = exam.dir / "exam.json"
    if not config_path.exists():
        config = build_exam_config(exam.id, exam.dir, exam)
        page_count = get_pdf_page_count(exam.pdf_path())
        if page_count is not None:
            config["total_pages"] = page_count
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    return config_path


EXAMS = load_exams()
print(f"Loaded {len(EXAMS)} exam(s)", flush=True)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            exam_id TEXT NOT NULL,
            student_id TEXT NOT NULL,
            student_name TEXT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finished_at TIMESTAMP,
            theta REAL,
            se REAL,
            total_questions INTEGER,
            correct INTEGER,
            estimated_ability REAL,
            archived BOOLEAN DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            item_id INTEGER NOT NULL,
            chosen TEXT NOT NULL,
            correct BOOLEAN NOT NULL,
            skipped BOOLEAN NOT NULL DEFAULT 0,
            theta_after REAL,
            se_after REAL,
            elapsed REAL DEFAULT 0,
            answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_responses_session ON responses(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_exam ON sessions(exam_id);
    """)

    session_columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)").fetchall()}
    if "resume_token" not in session_columns:
        db.execute("ALTER TABLE sessions ADD COLUMN resume_token TEXT")

    missing_tokens = db.execute(
        "SELECT id FROM sessions WHERE resume_token IS NULL OR resume_token = ''"
    ).fetchall()
    for session_row in missing_tokens:
        db.execute(
            "UPDATE sessions SET resume_token=? WHERE id=?",
            (secrets.token_urlsafe(16), session_row[0]),
        )

    response_columns = {row[1] for row in db.execute("PRAGMA table_info(responses)").fetchall()}
    if "skipped" not in response_columns:
        db.execute("ALTER TABLE responses ADD COLUMN skipped BOOLEAN NOT NULL DEFAULT 0")

    db.commit()
    db.close()


init_db()

# In-memory session store
active_sessions: dict[str, CATSession] = {}


def rebuild_session(session_id: str) -> CATSession | None:
    db_conn = sqlite3.connect(str(DB_PATH))
    db_conn.row_factory = sqlite3.Row
    sess = db_conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not sess:
        db_conn.close()
        return None

    exam = EXAMS.get(sess["exam_id"])
    if not exam:
        db_conn.close()
        return None

    responses = db_conn.execute(
        "SELECT * FROM responses WHERE session_id=? ORDER BY answered_at",
        (session_id,),
    ).fetchall()
    db_conn.close()

    cat = CATSession(
        student_id=sess["student_id"],
        items_pool=list(exam.items),
        theta=exam.cat_config.get("initial_theta", -2.0),
        routing_theta=exam.cat_config.get("initial_theta", -2.0),
        resume_token=sess["resume_token"] or "",
        initial_theta=exam.cat_config.get("initial_theta", -2.0),
        recency_half_life=exam.cat_config.get("recency_half_life", 8.0),
        skip_budget=int(exam.cat_config.get("skip_budget", 0) or 0),
    )
    for r in responses:
        item = exam.items_by_id.get(r["item_id"])
        if item:
            cat.current_item_id = item.id
            if r["skipped"]:
                skip_item(cat, item)
            else:
                process_answer(cat, item, r["chosen"])
    cat.finished = bool(sess["finished_at"])
    return cat


def get_exam_for_session(session_id: str) -> Exam | None:
    db = get_db()
    sess = db.execute("SELECT exam_id FROM sessions WHERE id=?", (session_id,)).fetchone()
    if sess:
        return EXAMS.get(sess["exam_id"])
    return None


def is_exam_accessible(exam: Exam | None) -> bool:
    return bool(exam and (exam.published or session.get("is_admin")))


def build_student_question_payload(exam: Exam | None, cat: CATSession, item: Item) -> dict:
    pdf_page = None
    if item.page_ref and exam:
        match = re.search(r'(\d+)', item.page_ref)
        if match:
            pdf_page = int(match.group(1)) + exam.pdf_offset

    return {
        "done": False,
        "question": item.question,
        "choices": item.choices,
        "question_number": len(cat.responses) + 1,
        "theta": round(cat.theta, 2),
        "se": round(cat.se, 2),
        "pdf_page": pdf_page,
        "skip_remaining": max(0, cat.skip_budget - cat.skip_count),
        "skip_budget": cat.skip_budget,
    }


def create_exam_session(
    exam: Exam,
    student_name: str,
    student_id: str,
    *,
    show_answers: bool = False,
) -> str:
    session_id = str(uuid.uuid4())
    resume_token = secrets.token_urlsafe(16)
    cat = CATSession(
        student_id=student_id,
        items_pool=list(exam.items),
        theta=exam.cat_config.get("initial_theta", -2.0),
        routing_theta=exam.cat_config.get("initial_theta", -2.0),
        resume_token=resume_token,
        initial_theta=exam.cat_config.get("initial_theta", -2.0),
        recency_half_life=exam.cat_config.get("recency_half_life", 8.0),
        skip_budget=int(exam.cat_config.get("skip_budget", 0) or 0),
    )
    active_sessions[session_id] = cat

    db = get_db()
    db.execute(
        "INSERT INTO sessions (id, exam_id, student_id, student_name, resume_token) VALUES (?, ?, ?, ?, ?)",
        (session_id, exam.id, student_id, student_name, resume_token),
    )
    db.commit()

    is_admin = bool(session.get("is_admin"))
    session.clear()
    if is_admin:
        session["is_admin"] = True
    session["session_id"] = session_id
    session["student_name"] = student_name
    session["exam_id"] = exam.id
    session["resume_token"] = resume_token
    session["show_answers"] = show_answers
    return session_id


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    public_exams = [e for e in EXAMS.values() if e.published]
    return render_template("home.html", exams=public_exams)


@app.route("/exam/<exam_id>")
def exam_landing(exam_id):
    exam = EXAMS.get(exam_id)
    if not is_exam_accessible(exam):
        return redirect(url_for("index"))
    return render_template("index.html", exam=exam)


@app.route("/exam/<exam_id>/start", methods=["POST"])
def start_exam(exam_id):
    exam = EXAMS.get(exam_id)
    if not is_exam_accessible(exam):
        return redirect(url_for("index"))

    student_name = request.form.get("student_name", "").strip()
    student_id = request.form.get("student_id", "").strip()
    if not student_name or not student_id:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    create_exam_session(exam, student_name, student_id, show_answers=False)
    return redirect(url_for("exam_page", exam_id=exam_id))


@app.route("/exam/<exam_id>/continue", methods=["POST"])
def continue_exam(exam_id):
    exam = EXAMS.get(exam_id)
    if not is_exam_accessible(exam):
        return redirect(url_for("index"))

    resume_token = request.form.get("resume_token", "").strip()
    if not resume_token:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    db = get_db()
    sess = db.execute(
        "SELECT * FROM sessions WHERE exam_id=? AND resume_token=? AND finished_at IS NULL "
        "ORDER BY started_at DESC LIMIT 1",
        (exam_id, resume_token),
    ).fetchone()

    if not sess:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    cat = rebuild_session(sess["id"])
    if not cat:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    active_sessions[sess["id"]] = cat
    session["session_id"] = sess["id"]
    session["student_name"] = sess["student_name"]
    session["exam_id"] = exam_id
    session["resume_token"] = sess["resume_token"]
    session["show_answers"] = False
    return redirect(url_for("exam_page", exam_id=exam_id))


@app.route("/exam/<exam_id>/take")
def exam_page(exam_id):
    exam = EXAMS.get(exam_id)
    if not is_exam_accessible(exam):
        return redirect(url_for("index"))

    session_id = session.get("session_id")
    if not session_id or session.get("exam_id") != exam_id:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    if session_id not in active_sessions:
        cat = rebuild_session(session_id)
        if cat:
            active_sessions[session_id] = cat
        else:
            return redirect(url_for("exam_landing", exam_id=exam_id))

    if not session.get("resume_token"):
        session["resume_token"] = active_sessions[session_id].resume_token

    return render_template("exam.html", student_name=session.get("student_name", ""),
                           exam=exam, resume_token=session.get("resume_token", ""),
                           show_answers=bool(session.get("show_answers") and session.get("is_admin")))


@app.route("/exam/<exam_id>/pdf")
def serve_pdf(exam_id):
    exam = EXAMS.get(exam_id)
    if not is_exam_accessible(exam) or not exam.has_pdf():
        return "No PDF", 404
    resp = send_file(str(exam.pdf_path()), mimetype="application/pdf")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/exam/<exam_id>/results")
def results_page(exam_id):
    session_id = session.get("session_id")
    exam = EXAMS.get(exam_id)
    if not session_id or session.get("exam_id") != exam_id or not exam:
        return redirect(url_for("index"))
    if session_id not in active_sessions:
        cat = rebuild_session(session_id)
        if not cat:
            return redirect(url_for("index"))
        active_sessions[session_id] = cat
    if not session.get("resume_token"):
        session["resume_token"] = active_sessions[session_id].resume_token
    return render_template("results.html", student_name=session.get("student_name", ""),
                           exam=exam, resume_token=session.get("resume_token", ""))


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/api/next")
def api_next_question():
    session_id = session.get("session_id")
    exam_id = session.get("exam_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400

    exam = EXAMS.get(exam_id)
    cat = active_sessions[session_id]

    if cat.finished or should_stop(cat):
        return jsonify({"done": True})

    if cat.current_item_id is not None:
        item = exam.items_by_id.get(cat.current_item_id) if exam else None
        if not item:
            cat.current_item_id = None
            return jsonify({"error": "item not found"}), 400
        payload = build_student_question_payload(exam, cat, item)
        if session.get("show_answers") and session.get("is_admin"):
            payload.update({
                "item_id": item.id,
                "answer": item.answer,
                "difficulty": round(item.difficulty, 2),
                "tier": item.tier,
                "blooms_level": item.blooms_level,
                "blooms_name": item.blooms_name,
            })
        return jsonify(payload)

    item = select_next_item(cat)
    if not item:
        return jsonify({"done": True})

    cat.current_item_id = item.id
    payload = build_student_question_payload(exam, cat, item)
    if session.get("show_answers") and session.get("is_admin"):
        payload.update({
            "item_id": item.id,
            "answer": item.answer,
            "difficulty": round(item.difficulty, 2),
            "tier": item.tier,
            "blooms_level": item.blooms_level,
            "blooms_name": item.blooms_name,
        })
    return jsonify(payload)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400

    cat = active_sessions[session_id]
    exam = get_exam_for_session(session_id)
    data = request.get_json(silent=True) or {}
    chosen = data.get("chosen", "").upper()
    try:
        elapsed = max(0.0, float(data.get("elapsed", 0)))
    except (TypeError, ValueError):
        elapsed = 0.0

    if cat.finished:
        return jsonify({"error": "session already finished"}), 400
    if cat.current_item_id is None:
        return jsonify({"error": "no pending question"}), 400
    if chosen not in "ABCDE":
        return jsonify({"error": "invalid answer"}), 400

    item = exam.items_by_id.get(cat.current_item_id) if exam else None
    if not item:
        return jsonify({"error": "item not found"}), 400

    try:
        resp = process_answer(cat, item, chosen)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    db = get_db()
    db.execute(
        "INSERT INTO responses (session_id, item_id, chosen, correct, skipped, theta_after, se_after, elapsed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, item.id, chosen, resp.correct, 0, resp.theta_after, resp.se_after, elapsed),
    )
    db.commit()

    done = should_stop(cat)
    result = {
        "correct": resp.correct,
        "correct_answer": item.answer,
        "theta": round(cat.theta, 2),
        "se": round(cat.se, 2),
        "done": done,
        "question_number": len(cat.responses),
        "blooms_level": item.blooms_level,
        "blooms_name": item.blooms_name,
        "skip_remaining": max(0, cat.skip_budget - cat.skip_count),
    }

    if done:
        summary = finish_session(session_id)
        result["summary"] = summary

    return jsonify(result)


@app.route("/api/skip", methods=["POST"])
def api_skip():
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400

    cat = active_sessions[session_id]
    exam = get_exam_for_session(session_id)
    data = request.get_json(silent=True) or {}
    try:
        elapsed = max(0.0, float(data.get("elapsed", 0)))
    except (TypeError, ValueError):
        elapsed = 0.0

    if cat.finished:
        return jsonify({"error": "session already finished"}), 400
    if cat.current_item_id is None:
        return jsonify({"error": "no pending question"}), 400
    if cat.skip_count >= cat.skip_budget:
        return jsonify({"error": "no skips remaining"}), 400

    item = exam.items_by_id.get(cat.current_item_id) if exam else None
    if not item:
        return jsonify({"error": "item not found"}), 400

    try:
        resp = skip_item(cat, item)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    db = get_db()
    db.execute(
        "INSERT INTO responses (session_id, item_id, chosen, correct, skipped, theta_after, se_after, elapsed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (session_id, item.id, resp.chosen, resp.correct, 1, resp.theta_after, resp.se_after, elapsed),
    )
    db.commit()

    return jsonify({
        "skipped": True,
        "skip_remaining": max(0, cat.skip_budget - cat.skip_count),
    })


@app.route("/api/finish", methods=["POST"])
def api_finish():
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400
    if active_sessions[session_id].finished:
        return jsonify({"done": True, "summary": get_performance_summary(active_sessions[session_id])})
    summary = finish_session(session_id)
    return jsonify({"done": True, "summary": summary})


@app.route("/api/history")
def api_history():
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400

    cat = active_sessions[session_id]
    exam = get_exam_for_session(session_id)

    db = get_db()
    db_responses = db.execute(
        "SELECT elapsed, item_id, skipped FROM responses WHERE session_id=? ORDER BY answered_at",
        (session_id,),
    ).fetchall()

    history = []
    for i, r in enumerate(cat.responses):
        elapsed = db_responses[i]["elapsed"] if i < len(db_responses) else (i + 1) * 15
        item_id = db_responses[i]["item_id"] if i < len(db_responses) else None
        skipped = bool(db_responses[i]["skipped"]) if i < len(db_responses) else False
        if skipped:
            continue
        item = exam.items_by_id.get(item_id) if exam and item_id else None
        history.append({
            "question": len(history) + 1,
            "theta": round(r.theta_after, 2),
            "correct": r.correct,
            "elapsed": elapsed or (i + 1) * 15,
            "blooms_level": item.blooms_level if item else 0,
            "blooms_name": item.blooms_name if item else "",
        })
    return jsonify({
        "history": history,
        "theta": round(cat.theta, 2),
        "skip_remaining": max(0, cat.skip_budget - cat.skip_count),
        "skip_budget": cat.skip_budget,
    })


@app.route("/api/results")
def api_results():
    session_id = session.get("session_id")
    if not session_id:
        return jsonify({"error": "no active session"}), 400
    if session_id not in active_sessions:
        cat = rebuild_session(session_id)
        if not cat:
            return jsonify({"error": "no active session"}), 400
        active_sessions[session_id] = cat
    cat = active_sessions[session_id]
    return jsonify(get_performance_summary(cat))


def finish_session(session_id: str) -> dict:
    cat = active_sessions[session_id]
    cat.finished = True
    cat.current_item_id = None
    summary = get_performance_summary(cat)
    db = get_db()
    db.execute(
        "UPDATE sessions SET finished_at=CURRENT_TIMESTAMP, theta=?, se=?, "
        "total_questions=?, correct=?, estimated_ability=? WHERE id=?",
        (summary["theta"], summary["se"], summary["total_questions"],
         summary["correct"], summary["estimated_ability"], session_id),
    )
    db.commit()
    return summary


def finish_incomplete_sessions_for_exam(exam_id: str) -> int:
    db = get_db()
    rows = db.execute(
        "SELECT id FROM sessions WHERE exam_id=? AND finished_at IS NULL",
        (exam_id,),
    ).fetchall()

    finished_count = 0
    for row in rows:
        session_id = row["id"]
        if session_id not in active_sessions:
            cat = rebuild_session(session_id)
            if not cat:
                continue
            active_sessions[session_id] = cat
        finish_session(session_id)
        finished_count += 1

    return finished_count


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------
def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        token = request.form.get("token", "").strip()
        if token and INSTRUCTOR_AUTH and hmac.compare_digest(token, INSTRUCTOR_AUTH):
            session.clear()
            session["is_admin"] = True
            return redirect(url_for("admin"))
        error = "Invalid token"
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout", methods=["POST"])
def admin_logout():
    session.clear()
    return redirect(url_for("admin_login"))


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------
@app.route("/admin")
@require_admin
def admin():
    exam_id = request.args.get("exam_id")

    if not exam_id:
        return render_template("admin_exams.html", exams=EXAMS.values())

    exam = EXAMS.get(exam_id)
    if not exam:
        return redirect(url_for("admin"))

    db = get_db()
    show_archived = request.args.get("show_archived") == "1"
    if show_archived:
        sessions = db.execute(
            "SELECT * FROM sessions WHERE exam_id=? ORDER BY started_at DESC",
            (exam_id,),
        ).fetchall()
    else:
        sessions = db.execute(
            "SELECT * FROM sessions WHERE exam_id=? AND archived=0 ORDER BY started_at DESC",
            (exam_id,),
        ).fetchall()

    finished = [s for s in sessions if s["finished_at"]]
    thetas = [s["theta"] for s in finished if s["theta"] is not None]
    stats = {
        "total_sessions": len(sessions),
        "finished": len(finished),
        "in_progress": len(sessions) - len(finished),
        "avg_theta": round(sum(thetas) / len(thetas), 2) if thetas else None,
        "avg_questions": round(sum(s["total_questions"] for s in finished if s["total_questions"]) / len(finished), 1) if finished else None,
        "avg_accuracy": round(sum(s["correct"] / s["total_questions"] * 100 for s in finished if s["total_questions"]) / len(finished), 1) if finished else None,
    }

    # Item bank stats
    blooms = {}
    bank_path = exam.dir / exam.config.get("bank_file", "bank.json")
    if bank_path.exists():
        with open(bank_path) as f:
            bank_raw = json.load(f)
        for q in bank_raw:
            bl = q.get("blooms_name", "Unclassified")
            blooms[bl] = blooms.get(bl, 0) + 1

    tiers = {}
    for item in exam.items:
        tiers[item.tier] = tiers.get(item.tier, 0) + 1

    return render_template("admin.html", sessions=sessions, stats=stats,
                           blooms=blooms, tiers=tiers, show_archived=show_archived,
                           exam=exam)


@app.route("/admin/session/<session_id>")
@require_admin
def admin_session(session_id):
    db = get_db()
    sess = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not sess:
        return redirect(url_for("admin"))

    exam = EXAMS.get(sess["exam_id"])
    responses = db.execute(
        "SELECT * FROM responses WHERE session_id=? ORDER BY answered_at",
        (session_id,),
    ).fetchall()

    enriched = []
    for r in responses:
        item = exam.items_by_id.get(r["item_id"]) if exam else None
        enriched.append({
            "index": len(enriched) + 1,
            "item_id": r["item_id"],
            "question": item.question[:100] if item else "?",
            "chosen": r["chosen"],
            "correct_answer": item.answer if item else "?",
            "correct": bool(r["correct"]),
            "skipped": bool(r["skipped"]),
            "theta_after": r["theta_after"],
            "elapsed": r["elapsed"],
            "difficulty": round(item.difficulty, 2) if item else None,
            "tier": item.tier if item else "?",
        })

    return render_template("admin_session.html", session=sess, responses=enriched,
                           exam=exam)


@app.route("/admin/session/<session_id>/archive", methods=["POST"])
@require_admin
def admin_archive(session_id):
    db = get_db()
    sess = db.execute("SELECT exam_id FROM sessions WHERE id=?", (session_id,)).fetchone()
    db.execute("UPDATE sessions SET archived=1 WHERE id=?", (session_id,))
    db.commit()
    return redirect(url_for("admin", exam_id=sess["exam_id"] if sess else None))


@app.route("/admin/session/<session_id>/unarchive", methods=["POST"])
@require_admin
def admin_unarchive(session_id):
    db = get_db()
    sess = db.execute("SELECT exam_id FROM sessions WHERE id=?", (session_id,)).fetchone()
    db.execute("UPDATE sessions SET archived=0 WHERE id=?", (session_id,))
    db.commit()
    return redirect(url_for("admin", exam_id=sess["exam_id"] if sess else None, show_archived="1"))


@app.route("/admin/api/session/<session_id>/trajectory")
@require_admin
def admin_trajectory(session_id):
    db = get_db()
    sess = db.execute("SELECT student_name FROM sessions WHERE id=?", (session_id,)).fetchone()
    responses = db.execute(
        "SELECT theta_after, correct, elapsed, skipped FROM responses WHERE session_id=? ORDER BY answered_at",
        (session_id,),
    ).fetchall()
    return jsonify({
        "session_id": session_id,
        "name": sess["student_name"] if sess else "?",
        "points": [
            {"q": i + 1, "theta": r["theta_after"], "correct": bool(r["correct"]), "elapsed": r["elapsed"]}
            for i, r in enumerate([row for row in responses if not row["skipped"]])
        ],
    })


@app.route("/admin/questions")
@require_admin
def admin_questions():
    exam_id = request.args.get("exam_id")
    if not exam_id and len(EXAMS) == 1:
        exam_id = list(EXAMS.keys())[0]
    exam = EXAMS.get(exam_id)
    if not exam:
        return redirect(url_for("admin"))

    bank_path = exam.dir / exam.config.get("bank_file", "bank.json")
    if not bank_path.exists():
        return render_template("admin_questions.html", questions=[], exam=exam,
                               tiers={}, blooms={}, total_q=0)
    with open(bank_path) as f:
        bank = json.load(f)

    db = get_db()
    q_stats = db.execute("""
        SELECT r.item_id, COUNT(*) as times_asked, SUM(r.correct) as times_correct,
               AVG(r.correct) as accuracy
        FROM responses r JOIN sessions s ON r.session_id = s.id
        WHERE s.exam_id = ? AND COALESCE(r.skipped, 0) = 0
        GROUP BY r.item_id
    """, (exam_id,)).fetchall()
    stats_map = {r["item_id"]: dict(r) for r in q_stats}

    questions = []
    for q in bank:
        s = stats_map.get(q["id"], {})
        questions.append({
            "id": q["id"],
            "question": q["question"],
            "question_short": q["question"][:80],
            "A": q.get("A", ""), "B": q.get("B", ""), "C": q.get("C", ""),
            "D": q.get("D", ""), "E": q.get("E", ""),
            "answer": q.get("answer", ""),
            "enabled": q.get("enabled", True),
            "difficulty": q.get("difficulty", "?"),
            "difficulty_score": q.get("difficulty_score"),
            "haiku_difficulty": q.get("haiku_difficulty"),
            "sonnet_difficulty": q.get("sonnet_difficulty"),
            "ensemble_difficulty_rating": q.get("ensemble_difficulty_rating"),
            "blooms_level": q.get("blooms_level"),
            "blooms_name": q.get("blooms_name", ""),
            "ensemble_flags": q.get("ensemble_flags"),
            "ensemble_flag_count": q.get("ensemble_flag_count", 0),
            "times_asked": s.get("times_asked", 0),
            "times_correct": s.get("times_correct", 0),
            "student_accuracy": round(s["accuracy"] * 100, 1) if s.get("accuracy") is not None else None,
        })

    # Tier and Bloom's distributions
    tiers = {}
    blooms = {}
    for q in bank:
        t = q.get("difficulty", "unknown")
        tiers[t] = tiers.get(t, 0) + 1
        bl = q.get("blooms_name", "Unclassified")
        blooms[bl] = blooms.get(bl, 0) + 1
    total_q = len(bank)

    return render_template("admin_questions.html", questions=questions, exam=exam,
                           tiers=tiers, blooms=blooms, total_q=total_q)


@app.route("/admin/build", methods=["GET", "POST"])
@require_admin
def admin_build():
    global EXAMS
    error = None
    success = None

    if request.method == "POST":
        exam_id = request.form.get("exam_id", "").strip()
        title = request.form.get("title", "").strip()
        subtitle = request.form.get("subtitle", "").strip()
        description = request.form.get("description", "").strip()
        pdf_offset = int(request.form.get("pdf_offset", 0))
        page_start = request.form.get("page_start", "").strip()
        page_end = request.form.get("page_end", "").strip()
        auto_build = request.form.get("auto_build") == "on"
        pdf_file = request.files.get("pdf")

        if not exam_id or not title or not pdf_file:
            error = "Exam ID, title, and PDF are required"
        elif not re.match(r'^[a-z0-9_-]+$', exam_id):
            error = "Exam ID must be lowercase letters, numbers, hyphens, underscores only"
        elif (EXAMS_DIR / exam_id).exists():
            error = f"Exam '{exam_id}' already exists"
        else:
            # Create exam directory
            exam_dir = EXAMS_DIR / exam_id
            exam_dir.mkdir(parents=True)

            # Save PDF
            pdf_path = exam_dir / "reference.pdf"
            pdf_file.save(str(pdf_path))

            # Get page count
            total_pages = get_pdf_page_count(pdf_path)

            # Auto-detect PDF page offset by finding first printed page number
            detected_offset = 0
            if total_pages:
                try:
                    for pdf_page in range(1, min(total_pages + 1, 20)):
                        result = subprocess.run(
                            ["pdftotext", "-f", str(pdf_page), "-l", str(pdf_page),
                             "-layout", str(pdf_path), "-"],
                            capture_output=True, text=True
                        )
                        # Look for a standalone page number (typically at bottom)
                        lines = result.stdout.strip().splitlines()
                        for line in reversed(lines[-5:] if len(lines) >= 5 else lines):
                            stripped = line.strip()
                            if stripped.isdigit() and 1 <= int(stripped) <= 500:
                                printed_page = int(stripped)
                                detected_offset = pdf_page - printed_page
                                break
                        if detected_offset != 0:
                            break
                except Exception:
                    pass

            # Use user override if provided, otherwise auto-detected
            if pdf_offset == 0 and detected_offset != 0:
                pdf_offset = detected_offset

            # Write config
            config = {
                "id": exam_id,
                "title": title,
                "subtitle": subtitle,
                "description": description,
                "bank_file": "bank.json",
                "reference_pdf": "reference.pdf",
                "pdf_page_offset": pdf_offset,
                "pdf_page_offset_auto": detected_offset,
                "total_pages": total_pages,
                "cat_config": {
                    "initial_theta": -2.0,
                    "se_threshold": 0.3,
                    "min_questions": 5,
                    "recency_half_life": 8,
                    "skip_budget": 0,
                },
            }
            with open(exam_dir / "exam.json", "w") as f:
                json.dump(config, f, indent=2)

            # Reload exams
            EXAMS = load_exams()

            if auto_build:
                build_cmd = [
                    str(PROJECT_DIR / ".venv" / "bin" / "python3"),
                    "-m", "pipeline.build_exam", str(exam_dir),
                    "--workers", "5",
                ]
                if page_start and page_end:
                    build_cmd.extend(["--pages", f"{page_start}-{page_end}"])

                subprocess.Popen(
                    build_cmd,
                    cwd=str(PROJECT_DIR),
                    stdout=open(str(exam_dir / "build.log"), "w"),
                    stderr=subprocess.STDOUT,
                )
                return redirect(url_for("admin_build_status", exam_id=exam_id))
            else:
                return redirect(url_for("admin"))

    return render_template("admin_build.html", error=error, success=success)


@app.route("/admin/build/<exam_id>/status")
@require_admin
def admin_build_status(exam_id):
    exam_dir = EXAMS_DIR / exam_id
    if not exam_dir.exists():
        return redirect(url_for("admin_build"))

    # Read config
    config = {}
    config_path = exam_dir / "exam.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Check build progress
    status = {"stage": "pending", "details": {}}

    gen_dir = exam_dir / "generated_questions"
    val_dir = exam_dir / "validation_results"
    bank_path = exam_dir / "bank.json"
    log_path = exam_dir / "build.log"

    # Count chunks
    gen_chunks = len(list(gen_dir.glob("chunk_*.json"))) if gen_dir.exists() else 0
    val_chunks = len(list(val_dir.glob("val_chunk_*.json"))) if val_dir.exists() else 0
    total_pages = config.get("total_pages", 0)
    expected_chunks = (total_pages + 4) // 5 if total_pages else 0

    if bank_path.exists():
        with open(bank_path) as f:
            bank = json.load(f)
        has_blooms = sum(1 for q in bank if q.get("blooms_level"))
        has_calibration = sum(1 for q in bank if q.get("difficulty_score") is not None)
        status["stage"] = "complete" if has_blooms and has_calibration else "post-processing"
        status["details"] = {
            "questions": len(bank),
            "blooms_classified": has_blooms,
            "calibrated": has_calibration,
        }
    elif val_chunks > 0:
        status["stage"] = "validating"
        status["details"] = {"validated": val_chunks, "total": gen_chunks}
    elif gen_chunks > 0:
        status["stage"] = "generating"
        status["details"] = {"generated": gen_chunks, "expected": expected_chunks}
    else:
        status["stage"] = "starting"

    # Read last lines of build log
    log_tail = ""
    if log_path.exists():
        lines = log_path.read_text().splitlines()
        log_tail = "\n".join(lines[-30:])

    return render_template("admin_build_status.html",
                           exam_id=exam_id, config=config, status=status,
                           log_tail=log_tail)


@app.route("/admin/exam/<exam_id>/take")
@require_admin
def admin_take_exam(exam_id):
    exam = EXAMS.get(exam_id)
    if not exam:
        return redirect(url_for("admin"))

    student_name = f"Instructor Test ({exam_id})"
    student_id = f"admin-test-{uuid.uuid4().hex[:8]}"
    create_exam_session(exam, student_name, student_id, show_answers=True)
    return redirect(url_for("exam_page", exam_id=exam_id))


@app.route("/admin/exam/<exam_id>/settings", methods=["GET", "POST"])
@require_admin
def admin_exam_settings(exam_id):
    global EXAMS
    exam = EXAMS.get(exam_id)
    if not exam:
        return redirect(url_for("admin"))

    config_path = ensure_exam_config_file(exam)
    saved = False

    if request.method == "POST":
        with open(config_path) as f:
            config = json.load(f)

        was_published = bool(config.get("published", False))

        config["title"] = request.form.get("title", config["title"])
        config["subtitle"] = request.form.get("subtitle", config.get("subtitle", ""))
        config["description"] = request.form.get("description", config.get("description", ""))
        config["published"] = request.form.get("published") == "on"
        config["show_reference"] = request.form.get("show_reference") == "on"
        config["pdf_page_offset"] = int(request.form.get("pdf_offset", 0))

        cat = config.get("cat_config", {})
        cat["initial_theta"] = float(request.form.get("initial_theta", -2.0))
        cat["recency_half_life"] = float(request.form.get("half_life", 8))
        cat["skip_budget"] = max(0, int(request.form.get("skip_budget", 0) or 0))
        config["cat_config"] = cat

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Reload
        EXAMS[exam_id] = Exam(exam.dir)
        exam = EXAMS[exam_id]

        if was_published and not config["published"]:
            finish_incomplete_sessions_for_exam(exam_id)

        saved = True

    with open(config_path) as f:
        config = json.load(f)
    if config.get("total_pages") in (None, "", "?"):
        page_count = get_pdf_page_count(exam.pdf_path())
        if page_count is not None:
            config["total_pages"] = page_count
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            EXAMS[exam_id] = Exam(exam.dir)
            exam = EXAMS[exam_id]

    demo_item_difficulties = [round(item.difficulty, 3) for item in exam.items]
    return render_template(
        "admin_exam_settings.html",
        exam=exam,
        saved=saved,
        demo_item_difficulties=demo_item_difficulties,
    )


@app.route("/admin/api/question/edit", methods=["POST"])
@require_admin
def admin_edit_question():
    global EXAMS
    data = request.get_json(silent=True) or {}
    exam_id = data.get("exam_id")
    qid = data.get("id")
    exam = EXAMS.get(exam_id)
    if not exam:
        return jsonify({"ok": False, "error": "exam not found"})

    bank_path = exam.dir / exam.config.get("bank_file", "bank.json")
    with open(bank_path) as f:
        bank = json.load(f)

    # Find and update the question
    updated = False
    for q in bank:
        if q["id"] == qid:
            for field in ["question", "A", "B", "C", "D", "E"]:
                if field in data:
                    q[field] = data[field]
            updated = True
            break

    if not updated:
        return jsonify({"ok": False, "error": "question not found"})

    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=2)

    # Reload exam items
    EXAMS[exam_id] = Exam(exam.dir)

    return jsonify({"ok": True})


@app.route("/admin/api/question/toggle", methods=["POST"])
@require_admin
def admin_toggle_question():
    global EXAMS
    data = request.get_json(silent=True) or {}
    exam_id = data.get("exam_id")
    qid = data.get("id")
    enabled = data.get("enabled", True)
    exam = EXAMS.get(exam_id)
    if not exam:
        return jsonify({"ok": False, "error": "exam not found"})

    bank_path = exam.dir / exam.config.get("bank_file", "bank.json")
    with open(bank_path) as f:
        bank = json.load(f)

    for q in bank:
        if q["id"] == qid:
            q["enabled"] = enabled
            break
    else:
        return jsonify({"ok": False, "error": "question not found"})

    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=2)
    EXAMS[exam_id] = Exam(exam.dir)
    return jsonify({"ok": True})


@app.route("/admin/api/question/delete", methods=["POST"])
@require_admin
def admin_delete_question():
    global EXAMS
    data = request.get_json(silent=True) or {}
    exam_id = data.get("exam_id")
    qid = data.get("id")
    exam = EXAMS.get(exam_id)
    if not exam:
        return jsonify({"ok": False, "error": "exam not found"})

    bank_path = exam.dir / exam.config.get("bank_file", "bank.json")
    with open(bank_path) as f:
        bank = json.load(f)

    bank = [q for q in bank if q["id"] != qid]

    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=2)
    EXAMS[exam_id] = Exam(exam.dir)
    return jsonify({"ok": True})


@app.route("/admin/export")
@require_admin
def admin_export():
    exam_id = request.args.get("exam_id")
    if not exam_id and len(EXAMS) == 1:
        exam_id = list(EXAMS.keys())[0]

    db = get_db()
    rows = db.execute("""
        SELECT s.student_name, s.student_id, s.exam_id, s.started_at, s.finished_at,
               s.theta, s.se, s.total_questions, s.correct, s.estimated_ability,
               r.item_id, r.chosen, r.correct as is_correct, r.skipped, r.theta_after, r.elapsed
        FROM sessions s
        LEFT JOIN responses r ON s.id = r.session_id
        WHERE (? IS NULL OR s.exam_id = ?)
        ORDER BY s.student_id, r.answered_at
    """, (exam_id, exam_id)).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["student_name", "student_id", "exam_id", "started_at", "finished_at",
                     "final_theta", "final_se", "total_questions", "total_correct",
                     "estimated_ability", "item_id", "chosen", "is_correct", "skipped",
                     "theta_after", "elapsed_seconds"])
    for r in rows:
        writer.writerow(list(r))

    fname = f"frostbit_{exam_id or 'all'}_results.csv"
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    host = "127.0.0.1" if args.debug else "0.0.0.0"
    app.run(host=host, port=args.port, debug=args.debug)
