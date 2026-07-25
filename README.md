# RAG Document Q&A

A production-style RAG (Retrieval-Augmented Generation) system that lets you upload documents and ask questions about them in natural language.

Built as an AI Engineering portfolio project — every architectural decision is documented.

---

## Architecture

```
Documents (PDF/DOCX/TXT)
        │
        ▼
   [Ingest Pipeline]
   TextLoader / PyPDFLoader
        │ split
        ▼
   RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
        │ embed
        ▼
   all-MiniLM-L6-v2  (local, no API cost)
        │ store
        ▼
   ChromaDB  (persisted to disk)
        │
        │  query time
        ▼
   Similarity Search (top-k chunks)
        │
        ▼
   Claude claude-sonnet-4-6  →  Answer + Sources
```

**Services:**
- `api` — FastAPI backend (port 8000), RAG pipeline, document ingestion
- `app` — Streamlit frontend (port 8501), ChatGPT-like interface

---

## Quick Start

### Option 1 — Docker (recommended)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd rag-portfolio

# 2. Set your API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 3. Start everything
docker compose up --build

# 4. Open the chat interface
open http://localhost:8501
```

The first startup downloads the embedding model (~90MB). Subsequent starts are fast.

### Option 2 — Local (development)

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env with your key

# Index sample documents
python src/phase4_multi_doc.py

# Terminal 1 — API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — UI
streamlit run src/app.py
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | API status + chunk count |
| POST | `/chat` | Ask a question |
| GET | `/chat/history` | Conversation history |
| DELETE | `/chat/history` | Clear history |
| POST | `/documents/upload` | Upload PDF/DOCX/TXT |
| GET | `/documents` | List indexed documents |

Interactive docs: `http://localhost:8000/docs`

---

## Project Structure

```
rag-portfolio/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app, lifespan, chat endpoints
│   │   ├── documents.py     # Upload + list endpoints
│   │   ├── ingest.py        # Incremental indexing logic
│   │   └── models.py        # Pydantic request/response models
│   ├── app.py               # Streamlit UI
│   ├── phase1_hello_rag.py  # Phase 1: local pipeline
│   ├── phase2_rag_with_claude.py
│   ├── phase3_persistent_rag.py
│   ├── phase4_multi_doc.py
│   └── phase5_chat_history.py
├── tests/
│   ├── conftest.py
│   ├── test_security.py
│   ├── test_chunking.py
│   └── test_api.py
├── data/
│   └── sample_docs/         # Drop your documents here
├── docs/
│   └── architecture.md      # All ADRs documented
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Running Tests

```bash
pytest
pytest --cov=src --cov-report=term-missing
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude claude-sonnet-4-6 (Anthropic) |
| Embeddings | all-MiniLM-L6-v2 (local, HuggingFace) |
| Vector DB | ChromaDB (disk-persisted) |
| Orchestration | LangChain 0.3 |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containers | Docker Compose |
| Testing | pytest + TestClient |
