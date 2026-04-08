# Frostbit

Adaptive testing platform with AI-generated questions and IRT-based computerized adaptive testing (CAT).

Drop a PDF, run one command, get a full adaptive exam site.

## Quick Start

```bash
# Setup
git clone https://github.com/rec3141/frostbit.git
cd frostbit
python3 -m venv .venv
source .venv/bin/activate
pip install flask anthropic openai

# System dependency (PDF rendering)
brew install poppler  # macOS
# apt install poppler-utils  # Linux

# Config
cp .env.example .env.local
# Edit .env.local with your OpenRouter API key

# Build an exam from a PDF
python3 -m pipeline.build_exam exams/my-exam/ --pdf /path/to/textbook.pdf

# Run the server
python3 app.py --port 8080
```

Open http://localhost:8080 to take exams, http://localhost:8080/admin for the instructor dashboard.

## How It Works

### For Students

1. Select an exam from the front page
2. Enter name and student ID
3. Answer questions — the system adapts difficulty to your level
4. Reference PDF is shown alongside questions (open book)
5. Challenge level chart updates in real time
6. Click "Finished" when done, or come back later with "Continue"

### For Instructors

1. Log in at `/admin` with the `INSTRUCTOR_AUTH` token
2. Upload a PDF and metadata at `/admin/build` to create a new exam
3. The pipeline generates questions, validates them, classifies by Bloom's taxonomy, and calibrates difficulty
4. Monitor student sessions, compare theta trajectories, export CSV
5. Edit individual questions at `/admin/questions`

## Exam Building Pipeline

The pipeline turns a PDF into a calibrated question bank:

```
PDF → Page Images → Question Generation → Validation → Filtering → Bloom's Classification → Difficulty Calibration
```

### Step by step

| Step | Script | What it does |
|------|--------|-------------|
| Generate | `pipeline/generate.py` | Renders 5-page chunks as images, sends to Sonnet for 5 easy + 5 medium + 5 hard questions each |
| Validate | `pipeline/validate.py` | Haiku + Sonnet answer each question with source text, rate difficulty 1-5, flag quality issues |
| Filter | `pipeline/filter_merge.py` | Drops questions where validators got the answer wrong or flagged issues |
| Bloom's | `pipeline/classify_blooms.py` | Haiku classifies each question by Bloom's Taxonomy level (1-6) |
| Calibrate | `pipeline/calibrate_remote.py` | Runs questions through free OpenRouter models for ensemble difficulty scores |
| Calibrate | `pipeline/calibrate_local.py` | Same but via local LM Studio models |

All steps are resumable — they save intermediate results and skip completed work.

### One-command build

```bash
python3 -m pipeline.build_exam exams/my-exam/ --pdf textbook.pdf --workers 5
```

Or build from the admin UI: `/admin/build`

## CAT Engine

The adaptive testing engine uses a 1-parameter logistic (Rasch) IRT model:

- **Item difficulty** — calibrated from multiple signals: validator ratings, ensemble model accuracy, generation tier
- **Ability estimation** — Maximum A Posteriori (MAP) with exponential recency weighting (half-life of 8 questions)
- **Item selection** — picks items closest to current ability estimate with jitter for variety
- **Starts easy** — initial theta at -2.0 (bottom of scale), students work up

The challenge level chart shows the student's trajectory over time, colored by Bloom's taxonomy level.

## Configuration

### `.env.local`

```
OPENROUTER_API_KEY=sk-or-v1-...
INSTRUCTOR_AUTH=your-secret-token
```

### Exam config (`exams/<id>/exam.json`)

```json
{
  "id": "bio101",
  "title": "BIO101 Introduction to Biology",
  "subtitle": "Midterm Exam",
  "description": "Adaptive exam covering chapters 1-5",
  "bank_file": "bank.json",
  "reference_pdf": "reference.pdf",
  "pdf_page_offset": 0,
  "cat_config": {
    "initial_theta": -2.0,
    "recency_half_life": 8
  }
}
```

`pdf_page_offset` is the difference between PDF page numbers and printed page numbers (e.g., if the PDF has 6 pages of front matter before page 1, set this to 6).

## Project Structure

```
frostbit/
├── app.py              # Flask web app (multi-exam, admin auth)
├── cat_engine.py       # IRT adaptive testing engine
├── pipeline/           # Exam building tools
│   ├── build_exam.py   # One-command pipeline runner
│   ├── generate.py     # Question generation (Sonnet + vision)
│   ├── validate.py     # Validation (Haiku + Sonnet)
│   ├── filter_merge.py # Filter and merge question banks
│   ├── classify_blooms.py  # Bloom's taxonomy classification
│   ├── calibrate_remote.py # Difficulty calibration (OpenRouter)
│   └── calibrate_local.py  # Difficulty calibration (LM Studio)
├── templates/          # Jinja2 HTML templates
├── static/             # CSS, PDF viewer
├── exams/              # One directory per exam
│   └── <exam-id>/
│       ├── exam.json   # Exam configuration
│       ├── bank.json   # Question bank
│       └── reference.pdf
└── .env.local          # API keys (not committed)
```

## Tech Stack

- **Backend**: Flask, SQLite
- **Frontend**: Vanilla JS, Canvas charts, pdf.js
- **Question generation**: Claude Sonnet (via OpenRouter) with PDF page images
- **Validation**: Claude Haiku + Sonnet with source text
- **Calibration**: Ensemble of free OpenRouter models + local LM Studio models
- **IRT model**: 1PL Rasch with MAP estimation and recency weighting

## Dark Mode

Follows system preference automatically via `prefers-color-scheme` CSS media query.

## License

MIT
