# RAG Document Q&A

A production-style RAG (Retrieval-Augmented Generation) system that lets you upload documents and ask questions about them in natural language.

Built as an AI Engineering portfolio project — every architectural decision is documented.

![CI](https://github.com/MateusPortoo/rag-portfolio/actions/workflows/ci.yml/badge.svg)

---

## Architecture

```
Documents (PDF/DOCX/TXT)
        │
        ▼
   [Ingest Pipeline]
   TextLoader / PyPDFLoader / Docx2txtLoader
        │ split
        ▼
   RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
        │ enrich metadata
        ▼
   source · doc_type · chunk_index · total_chunks
        │ embed
        ▼
   all-MiniLM-L6-v2  (local, no API cost)
        │ store
        ▼
   ChromaDB — HNSW (cosine, M=16, ef_construction=200, search_ef=100)
        │
        │  query time
        ▼
   Similarity Search (top-4 chunks) + optional metadata filter
        │
        ▼
   Llama 3.1 8B via Groq  →  Answer + Sources
```

**Services:**
- `api` — FastAPI backend (port 8000), RAG pipeline, document ingestion
- `app` — Streamlit frontend (port 8501), chat interface

---

## Quick Start

### Option 1 — Docker (recommended)

```bash
git clone <repo-url>
cd rag-portfolio

cp .env.example .env
# Edit .env: GROQ_API_KEY=gsk_...
# Get a free key at https://console.groq.com/keys

docker compose up --build

open http://localhost:8501
```

### Option 2 — Local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Groq key

python src/phase4_multi_doc.py   # index sample documents

uvicorn src.api.main:app --reload --port 8000   # terminal 1
streamlit run src/app.py                         # terminal 2
```

---

## HNSW Configuration

ChromaDB uses HNSW (Hierarchical Navigable Small World) as its vector index. This project configures it explicitly instead of using defaults:

| Parameter | Value | Why |
|-----------|-------|-----|
| `hnsw:space` | `cosine` | Correct metric for L2-normalized embeddings |
| `hnsw:construction_ef` | `200` | Higher recall during indexing (default is 100) |
| `hnsw:M` | `16` | Connections per node — standard for small datasets |
| `hnsw:search_ef` | `100` | Better recall at query time (default is 10) |

These parameters are set in `src/api/ingest.py → HNSW_CONFIG` and apply when the collection is first created.

**Trade-off:** higher `ef` values improve recall but increase latency. The values above are tuned for a portfolio dataset (< 10k chunks). For millions of chunks, tune with a benchmark first.

---

## Metadata Filtering

Every chunk stored in ChromaDB carries four metadata fields:

| Field | Type | Example | Use case |
|-------|------|---------|----------|
| `source` | string | `"politica_empresa.txt"` | Isolate a specific file |
| `doc_type` | string | `"pdf"`, `"txt"`, `"docx"` | Filter by file type |
| `chunk_index` | int | `0`, `1`, `2` | Select position within document |
| `total_chunks` | int | `12` | Know how big the document is |

**Usage via `build_retriever()`:**

```python
from src.api.ingest import build_retriever

# Search only in one file
retriever = build_retriever(vector_store, filter={"source": "politica_empresa.txt"})

# Search only in PDFs
retriever = build_retriever(vector_store, filter={"doc_type": "pdf"})

# Search only in early chunks (document intro / summary)
retriever = build_retriever(vector_store, filter={"chunk_index": {"$lte": 3}})

# Combine filters (ChromaDB $and syntax)
retriever = build_retriever(vector_store, filter={
    "$and": [{"doc_type": "txt"}, {"chunk_index": {"$gte": 2}}]
})
```

Metadata filters execute **before** the embedding search — they are O(1) index lookups with no LLM cost.
---

## Hybrid Search (BM25 + Semantic + RRF)

The retrieval pipeline in src/phase6_hybrid_rag.py combines **lexical search (BM25)** with **semantic search (ChromaDB)** via **Reciprocal Rank Fusion (RRF)**.

| Method | Strength |
|--------|----------|
| BM25 | Exact terms, acronyms, proper nouns, numbers |
| Semantic | Concepts, paraphrases, natural language |
| RRF | Combines both rankings without weight tuning |

**RRF formula:** score(d) = sum(1 / (k + rank(d))) where k=60.

---

## HyDE - Hypothetical Document Embeddings

Instead of embedding the query directly, the LLM generates a **hypothetical answer** paragraph first. That paragraph is embedded and used to search ChromaDB.

`
query -> [LLM] -> hypothetical document -> [embed] -> similarity_search_by_vector -> real chunks
`

**Why it works:** A hypothetical answer lives in the same vector space as real documents (assertive, document-style prose), so its embedding sits closer to relevant chunks than the query embedding directly.

Reference: Gao et al. 2022 - Precise Zero-Shot Dense Retrieval without Relevance Labels

The API startup (src/api/main.py) uses HyDE by default.
---

## Reranking (Cross-Encoder)

After initial retrieval, a **cross-encoder** scores each (query, chunk) pair together and reorders the candidates by true relevance before they reach the LLM.

`
HyDE retriever → top-15 candidates → cross-encoder → top-5 reranked → LLM
`

| Stage | Model type | Speed | Accuracy |
|-------|-----------|-------|----------|
| Retrieval (HyDE) | Bi-encoder (embed separately) | Fast | Good |
| Reranking | Cross-encoder (joint attention) | Slower | Better |

**Why cross-encoders are more accurate:** they attend to both the query and the document simultaneously, capturing term interactions that bi-encoders miss.

**Model:** cross-encoder/ms-marco-MiniLM-L-6-v2 (6-layer MiniLM, ~22M params, runs on CPU in <1s for 15 candidates).





---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | API status + chunk count |
| POST | `/chat` | Ask a question, get full response |
| POST | `/chat/stream` | Ask a question, get streaming SSE response |
| GET | `/chat/history` | Conversation history |
| DELETE | `/chat/history` | Clear history |
| POST | `/documents/upload` | Upload PDF/DOCX/TXT |
| GET | `/documents` | List indexed documents |

Interactive docs: `http://localhost:8000/docs`

### Streaming endpoint

`POST /chat/stream` returns `text/event-stream`. Each event is a JSON object:

```
data: {"token": "The answer"}
data: {"token": " is 30 days."}
data: {"done": true, "sources": ["politica_empresa.txt"]}
data: {"error": "..."}   ← only on failure
```

---

## Project Structure

```
rag-portfolio/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions — pytest + coverage
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI app, lifespan, chat + stream endpoints
│   │   ├── documents.py         # Upload + list endpoints
│   │   ├── ingest.py            # Indexing: HNSW config, metadata enrichment, retriever builder
│   │   └── models.py            # Pydantic request/response models
│   ├── app.py                   # Streamlit UI
│   ├── phase1_hello_rag.py      # Phase 1: local pipeline without LLM
│   ├── phase2_rag_with_claude.py  # Phase 2: historical prototype (discontinued)
│   ├── phase3_persistent_rag.py
│   ├── phase4_multi_doc.py
│   └── phase5_chat_history.py
├── tests/
│   ├── conftest.py
│   ├── test_security.py
│   ├── test_chunking.py
│   └── test_api.py
├── data/
│   └── sample_docs/             # Drop your documents here
├── docs/
│   └── architecture.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

> **Note:** `phase2_rag_with_claude.py` is a historical prototype from early development
> that used Anthropic's Claude API. It is not part of the production pipeline and requires
> `langchain-anthropic` (not in `requirements.txt`). The production LLM is Llama 3.1 8B via Groq.

---

## Running Tests

```bash
pytest                                          # run all tests
pytest --cov=src --cov-report=term-missing      # with coverage report
pytest tests/test_security.py -v               # single file
```

Coverage target: **80%+**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Llama 3.1 8B via Groq (free tier) |
| Embeddings | all-MiniLM-L6-v2 (local, HuggingFace) |
| Vector DB | ChromaDB with explicit HNSW config |
| Orchestration | LangChain 0.3 |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containers | Docker Compose |
| Testing | pytest + coverage (CI via GitHub Actions) |
