#!/usr/bin/env python3
"""
Frostbit — Computerized Adaptive Testing platform.

Multi-exam Flask app. Each exam lives in exams/<id>/ with exam.json, bank.json,
and an optional reference.pdf.

Usage:
    source .venv/bin/activate
    python3 app.py --port 8080 --debug
"""

import csv
import io
import json
import os
import re
import sqlite3
import uuid
from functools import wraps
from pathlib import Path

from flask import (
    Flask, Response, g, jsonify, redirect, render_template,
    request, send_file, session, url_for,
)

from cat_engine import (
    CATSession, Item, get_performance_summary, process_answer,
    select_next_item, should_stop,
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

# Stable secret so sessions survive server restarts
_secret_path = PROJECT_DIR / ".flask_secret"
if _secret_path.exists():
    app.secret_key = _secret_path.read_bytes()
else:
    app.secret_key = os.urandom(24)
    _secret_path.write_bytes(app.secret_key)
    os.chmod(str(_secret_path), 0o600)

INSTRUCTOR_AUTH = os.environ.get("INSTRUCTOR_AUTH", "")


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
            theta_after REAL,
            se_after REAL,
            elapsed REAL DEFAULT 0,
            answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE INDEX IF NOT EXISTS idx_responses_session ON responses(session_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_exam ON sessions(exam_id);
    """)
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
    )
    for r in responses:
        item = exam.items_by_id.get(r["item_id"])
        if item:
            process_answer(cat, item, r["chosen"])
    return cat


def get_exam_for_session(session_id: str) -> Exam | None:
    db = get_db()
    sess = db.execute("SELECT exam_id FROM sessions WHERE id=?", (session_id,)).fetchone()
    if sess:
        return EXAMS.get(sess["exam_id"])
    return None


# ---------------------------------------------------------------------------
# Public routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("home.html", exams=EXAMS.values())


@app.route("/exam/<exam_id>")
def exam_landing(exam_id):
    exam = EXAMS.get(exam_id)
    if not exam:
        return redirect(url_for("index"))
    return render_template("index.html", exam=exam)


@app.route("/exam/<exam_id>/start", methods=["POST"])
def start_exam(exam_id):
    exam = EXAMS.get(exam_id)
    if not exam:
        return redirect(url_for("index"))

    student_name = request.form.get("student_name", "").strip()
    student_id = request.form.get("student_id", "").strip()
    if not student_name or not student_id:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    session_id = str(uuid.uuid4())
    cat = CATSession(
        student_id=student_id,
        items_pool=list(exam.items),
        theta=exam.cat_config.get("initial_theta", -2.0),
    )
    active_sessions[session_id] = cat

    db = get_db()
    db.execute(
        "INSERT INTO sessions (id, exam_id, student_id, student_name) VALUES (?, ?, ?, ?)",
        (session_id, exam_id, student_id, student_name),
    )
    db.commit()

    session["session_id"] = session_id
    session["student_name"] = student_name
    session["exam_id"] = exam_id
    return redirect(url_for("exam_page", exam_id=exam_id))


@app.route("/exam/<exam_id>/continue", methods=["POST"])
def continue_exam(exam_id):
    student_id = request.form.get("student_id", "").strip()
    if not student_id:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    db = get_db()
    sess = db.execute(
        "SELECT * FROM sessions WHERE exam_id=? AND student_id=? AND finished_at IS NULL "
        "ORDER BY started_at DESC LIMIT 1",
        (exam_id, student_id),
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
    return redirect(url_for("exam_page", exam_id=exam_id))


@app.route("/exam/<exam_id>/take")
def exam_page(exam_id):
    exam = EXAMS.get(exam_id)
    if not exam:
        return redirect(url_for("index"))

    session_id = session.get("session_id")
    if not session_id:
        return redirect(url_for("exam_landing", exam_id=exam_id))

    if session_id not in active_sessions:
        cat = rebuild_session(session_id)
        if cat:
            active_sessions[session_id] = cat
        else:
            return redirect(url_for("exam_landing", exam_id=exam_id))

    return render_template("exam.html", student_name=session.get("student_name", ""),
                           exam=exam)


@app.route("/exam/<exam_id>/pdf")
def serve_pdf(exam_id):
    exam = EXAMS.get(exam_id)
    if not exam or not exam.has_pdf():
        return "No PDF", 404
    resp = send_file(str(exam.pdf_path()), mimetype="application/pdf")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


@app.route("/exam/<exam_id>/results")
def results_page(exam_id):
    session_id = session.get("session_id")
    if not session_id:
        return redirect(url_for("index"))
    exam = EXAMS.get(exam_id)
    return render_template("results.html", student_name=session.get("student_name", ""),
                           exam=exam)


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

    if should_stop(cat):
        return jsonify({"done": True})

    item = select_next_item(cat)
    if not item:
        return jsonify({"done": True})

    pdf_page = None
    if item.page_ref and exam:
        m = re.search(r'(\d+)', item.page_ref)
        if m:
            pdf_page = int(m.group(1)) + exam.pdf_offset

    return jsonify({
        "done": False,
        "item_id": item.id,
        "question": item.question,
        "choices": item.choices,
        "answer": item.answer,
        "question_number": len(cat.responses) + 1,
        "theta": round(cat.theta, 2),
        "se": round(cat.se, 2),
        "pdf_page": pdf_page,
        "difficulty": round(item.difficulty, 2),
        "tier": item.tier,
        "blooms_level": item.blooms_level,
        "blooms_name": item.blooms_name,
    })


@app.route("/api/answer", methods=["POST"])
def api_answer():
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400

    cat = active_sessions[session_id]
    exam = get_exam_for_session(session_id)
    data = request.get_json()
    item_id = data.get("item_id")
    chosen = data.get("chosen", "").upper()
    elapsed = data.get("elapsed", 0)

    if not item_id or chosen not in "ABCDE":
        return jsonify({"error": "invalid answer"}), 400

    item = exam.items_by_id.get(item_id) if exam else None
    if not item:
        return jsonify({"error": "item not found"}), 400

    resp = process_answer(cat, item, chosen)

    db = get_db()
    db.execute(
        "INSERT INTO responses (session_id, item_id, chosen, correct, theta_after, se_after, elapsed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, item_id, chosen, resp.correct, resp.theta_after, resp.se_after, elapsed),
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
    }

    if done:
        summary = finish_session(session_id)
        result["summary"] = summary

    return jsonify(result)


@app.route("/api/finish", methods=["POST"])
def api_finish():
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400
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
        "SELECT elapsed, item_id FROM responses WHERE session_id=? ORDER BY answered_at",
        (session_id,),
    ).fetchall()

    history = []
    for i, r in enumerate(cat.responses):
        elapsed = db_responses[i]["elapsed"] if i < len(db_responses) else (i + 1) * 15
        item_id = db_responses[i]["item_id"] if i < len(db_responses) else None
        item = exam.items_by_id.get(item_id) if exam and item_id else None
        history.append({
            "question": i + 1,
            "theta": round(r.theta_after, 2),
            "correct": r.correct,
            "elapsed": elapsed or (i + 1) * 15,
            "blooms_level": item.blooms_level if item else 0,
            "blooms_name": item.blooms_name if item else "",
        })
    return jsonify({"history": history, "theta": round(cat.theta, 2)})


@app.route("/api/results")
def api_results():
    session_id = session.get("session_id")
    if not session_id or session_id not in active_sessions:
        return jsonify({"error": "no active session"}), 400
    cat = active_sessions[session_id]
    return jsonify(get_performance_summary(cat))


def finish_session(session_id: str) -> dict:
    cat = active_sessions[session_id]
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
        if token and token == INSTRUCTOR_AUTH:
            session["is_admin"] = True
            return redirect(url_for("admin"))
        error = "Invalid token"
    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


# ---------------------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------------------
@app.route("/admin")
@require_admin
def admin():
    # If only one exam, go directly to it
    exam_id = request.args.get("exam_id")
    if not exam_id and len(EXAMS) == 1:
        exam_id = list(EXAMS.keys())[0]

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
    with open(exam.dir / exam.config.get("bank_file", "bank.json")) as f:
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
        "SELECT theta_after, correct, elapsed FROM responses WHERE session_id=? ORDER BY answered_at",
        (session_id,),
    ).fetchall()
    return jsonify({
        "session_id": session_id,
        "name": sess["student_name"] if sess else "?",
        "points": [
            {"q": i + 1, "theta": r["theta_after"], "correct": bool(r["correct"]), "elapsed": r["elapsed"]}
            for i, r in enumerate(responses)
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

    with open(exam.dir / exam.config.get("bank_file", "bank.json")) as f:
        bank = json.load(f)

    db = get_db()
    q_stats = db.execute("""
        SELECT r.item_id, COUNT(*) as times_asked, SUM(r.correct) as times_correct,
               AVG(r.correct) as accuracy
        FROM responses r JOIN sessions s ON r.session_id = s.id
        WHERE s.exam_id = ?
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

    return render_template("admin_questions.html", questions=questions, exam=exam)


@app.route("/admin/build", methods=["GET", "POST"])
@require_admin
def admin_build():
    import subprocess
    error = None
    success = None

    if request.method == "POST":
        exam_id = request.form.get("exam_id", "").strip()
        title = request.form.get("title", "").strip()
        subtitle = request.form.get("subtitle", "").strip()
        description = request.form.get("description", "").strip()
        pdf_offset = int(request.form.get("pdf_offset", 0))
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

            # Write config
            config = {
                "id": exam_id,
                "title": title,
                "subtitle": subtitle,
                "description": description,
                "bank_file": "bank.json",
                "reference_pdf": "reference.pdf",
                "pdf_page_offset": pdf_offset,
                "cat_config": {
                    "initial_theta": -2.0,
                    "se_threshold": 0.3,
                    "min_questions": 5,
                    "recency_half_life": 8,
                },
            }
            with open(exam_dir / "exam.json", "w") as f:
                json.dump(config, f, indent=2)

            if auto_build:
                # Kick off build in background
                venv_python = str(PROJECT_DIR / ".venv" / "bin" / "python3")
                subprocess.Popen(
                    [venv_python, "-m", "pipeline.build_exam", str(exam_dir)],
                    cwd=str(PROJECT_DIR),
                    stdout=open(str(exam_dir / "build.log"), "w"),
                    stderr=subprocess.STDOUT,
                )
                success = f"Exam '{exam_id}' created! Build running in background — check {exam_dir}/build.log"
            else:
                success = f"Exam '{exam_id}' created. Run: python3 -m pipeline.build_exam exams/{exam_id}/"

            # Reload exams
            global EXAMS
            EXAMS = load_exams()

    return render_template("admin_build.html", error=error, success=success)


@app.route("/admin/api/question/edit", methods=["POST"])
@require_admin
def admin_edit_question():
    data = request.get_json()
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
    global EXAMS
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
               r.item_id, r.chosen, r.correct as is_correct, r.theta_after, r.elapsed
        FROM sessions s
        LEFT JOIN responses r ON s.id = r.session_id
        WHERE (? IS NULL OR s.exam_id = ?)
        ORDER BY s.student_id, r.answered_at
    """, (exam_id, exam_id)).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["student_name", "student_id", "exam_id", "started_at", "finished_at",
                     "final_theta", "final_se", "total_questions", "total_correct",
                     "estimated_ability", "item_id", "chosen", "is_correct",
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
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
