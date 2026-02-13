# AGENTS.md

This document provides guidance for autonomous coding agents operating in this repository.
It describes the tech stack, build/test commands, conventions, and architectural expectations.

---

## 1. Tech Stack Overview

Primary language:
- Python 3.9+

Web framework:
- FastAPI
- Uvicorn (ASGI server)

Database:
- PostgreSQL 14+
- pgvector extension
- psycopg (v3)
- asyncpg

AI / RAG stack:
- sentence-transformers
- OpenAI SDK (optional provider)
- Ollama SDK (local provider)
- LangChain (core + openai + text splitters + experimental)
- NumPy

Document parsing:
- PyPDF2
- python-docx

Configuration:
- python-dotenv (.env file)

The application is a Retrieval-Augmented Generation (RAG) classification API.

---

## 2. Project Structure

High-level layout:

- `app.py` – FastAPI entry point
- `database/` – DB connection and data access
- `embedding/` – Embedding model logic
- `rag/` – Retrieval and classification pipeline
- `utils/` – Logging and helpers
- `data/` – Training documents
- `classification_relational.sql` – Database schema
- `ingest_documents.py` – Embeds and stores training docs
- `test_classification.py` – Script-based API test

Agents should preserve this separation of concerns.

---

## 3. Environment Setup

Create virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy configuration:

```powershell
cp .env.example .env
```

Do NOT hardcode secrets. Always rely on environment variables.

---

## 4. Database Setup

Requirements:
- PostgreSQL running
- pgvector extension enabled

Apply schema manually using:

`classification_relational.sql`

Optional initialization:

```powershell
python setup_db.py
```

The system will not function correctly until the database is initialized.

---

## 5. Data Ingestion (Required Before Classification)

Training documents must be embedded before classification works:

```powershell
python ingest_documents.py
```

If not run, classification may return:
"No similar documents found. Ensure training documents are ingested."

Agents modifying retrieval logic must preserve chunk storage and similarity scoring.

---

## 6. Running the API

Default:

```powershell
python app.py
```

Custom port + reload:

```powershell
python app.py --port 8001 --reload
```

Alternative using uvicorn directly:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Production (Linux/macOS example):

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

---

## 7. Testing

There is no pytest or unittest suite.
Testing is script-based.

Run classification test script:

```powershell
python test_classification.py
```

Important:
- The API must be running.
- The script expects `http://localhost:8001`.

Start server accordingly:

```powershell
python app.py --port 8001
```

### Running a Single Test Scenario

Options:

1. Modify `number_of_tests` inside `test_classification.py`.
2. Reduce the sample list to one entry.
3. Call classification directly:

```powershell
python classify_document.py "Example text"
```

4. Use curl:

```bash
curl -X POST "http://localhost:8000/classify/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Example text", "include_context": false}'
```

Agents adding tests should prefer pytest and keep tests isolated from running server where possible.

---

## 8. Code Style Guidelines

### General

- Use snake_case for files and functions.
- Use PascalCase for classes.
- Use ALL_CAPS for constants and environment variables.
- Follow existing module boundaries.

### Imports

- Standard library first.
- Third-party packages second.
- Local imports last.
- Avoid circular imports.

Do not use wildcard imports.

### Typing

- Use Python type hints consistently.
- Prefer built-in generics (e.g., `list[str]`).
- Use `Optional[T]` or `T | None` consistently.
- Pydantic models define request/response schemas.

Do not remove type hints unless strictly necessary.

### Async Usage

- API endpoints should be async.
- Use async DB calls when available.
- Do not mix blocking operations inside async routes.

### Error Handling

- Use `HTTPException` in FastAPI routes.
- Log errors with stack traces (`exc_info=True`).
- Avoid swallowing exceptions silently.

### Logging

- Use utilities from `utils/`.
- Maintain structured, informative startup logs.
- Do not remove important system diagnostics.

---

## 9. Architectural Conventions

The classification flow:

1. Document is chunked.
2. Chunks are embedded.
3. Similar chunks retrieved from DB.
4. Tags aggregated by similarity.
5. LLM produces final classification.

Agents must preserve:
- Similarity-weighted aggregation
- Top-K retrieval logic
- Provider abstraction (OpenAI vs local)

Do not hardcode a specific LLM provider.

---

## 10. Linting and Formatting

No formatter or linter is currently configured.

Observed style:
- 4-space indentation
- Clear, descriptive function names
- Minimal inline comments

If adding linting tools (e.g., ruff, black, mypy), do so in a separate commit.

---

## 11. Cursor and Copilot Rules

No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` files are present.

If added in the future, agents must follow them strictly.

---

## 12. Agent Behavior Guidelines

- Do not modify database schema without updating SQL file.
- Do not remove ingestion requirement.
- Do not hardcode secrets.
- Maintain modular boundaries.
- Preserve async correctness.
- Prefer incremental changes over large refactors.

When uncertain, follow existing patterns in the repository.

---

End of AGENTS.md
