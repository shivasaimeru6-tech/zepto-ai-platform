# Zepto Data & AI Platform — Capstone Project

One repository, three linked modules: a data-engineering pipeline
(`/data_pipeline`), an analytics + predictive-modeling pipeline
(`/analytics`), and a GenAI support assistant (`/support_assistant`).

## Setup

A single consolidated `requirements.txt` is provided at the repo root
(there is also a module-local `support_assistant/requirements.txt` covering
just that module, if you prefer to install module-by-module).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Network access is required for parts of this project that this development
environment itself doesn't have: scraping `books.toscrape.com` (Module 1),
seaborn's first-time Titanic dataset fetch (Module 2), and downloading the
`all-MiniLM-L6-v2` model weights the first time `sentence-transformers` runs
(Module 3). All three cache/save locally after their first successful run.

## How to run each module

### Module 1 — Data Pipeline
```bash
cd data_pipeline
python scrape_clean_load.py
```
Scrapes ≥60 books across ≥3 categories from books.toscrape.com, cleans and
converts prices (fixed rate 1 GBP = 105.50 INR), loads a normalized SQLite
DB (`books.db`), and runs the required SQL + pandas queries. See
`data_pipeline/README.md` for design decisions.

### Module 2 — Analytics Pipeline
```bash
cd analytics
python 01_eda.py        # loads Titanic once, profiles, cleans, tells the data story
python 02_modeling.py   # reads the SAME cleaned data, runs the modeling pipeline
```
See `analytics/README.md` for design decisions, and `analytics/charts/`
(generated) for saved plots. All written interpretations required by the
brief are printed by the scripts as they run.

### Module 3 — Support Assistant
```bash
cd support_assistant
uvicorn main:app --reload
# or: docker build -t zepto-support-assistant . && docker run -p 7860:7860 zepto-support-assistant
```
`MOCK_LLM` defaults to `1` (offline, deterministic — this is the graded
baseline; leave it unset). See `support_assistant/README.md` for the full
architecture description and example call transcripts.

## Design decisions summary
- **Module 1**: fixed-rate GBP→INR conversion (no live lookup needed);
  two-table normalized schema (`categories` ↔ `books`); rows with
  unparseable ratings are dropped, residual missing prices are
  median-imputed. Full detail in `data_pipeline/README.md`.
- **Module 2**: missing-value threshold rule (<5% drop, 5–30% impute, very
  high → drop column); all preprocessing fit on the training split only via
  a `ColumnTransformer`/`Pipeline`; three classifiers compared with a full
  metric suite plus an imbalance-handling and hyperparameter-tuning study;
  a regression side-task on `fare`. Full detail in `analytics/README.md`.
- **Module 3**: local, free, keyless embeddings (`sentence-transformers`)
  and vector storage (ChromaDB); a 3-node LangGraph router; every LLM call
  gated behind `MOCK_LLM` (default: fully offline mock, which is what's
  graded); a Pydantic-validated JSON response schema. Full detail in
  `support_assistant/README.md`.

## Git workflow
This repository's history includes a feature branch created, committed to
at least twice, and merged back into `main` — visible via
`git log --graph --all`. See `git log` for the actual commit history.
