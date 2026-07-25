# Architecture Decisions — RAG Portfolio

## ADR-001 · LLM Provider: Anthropic Claude

**Decision:** Use `claude-sonnet-4-6` via the Anthropic SDK (not OpenAI).

**Reasons:**
- 200k context window — handles large document sets without truncation
- Strong instruction-following in Portuguese
- `langchain-anthropic` integrates natively with the LangChain pipeline

**Trade-off:** Requires a paid API key (no free tier). Mitigated by Phase 1
running 100% locally without any API calls.

---

## ADR-002 · Vector Database: ChromaDB (local)

**Decision:** ChromaDB over Pinecone or Weaviate.

**Reasons:**
- Zero infrastructure — runs in-memory or on disk, no cloud account
- Python-native API, integrates with LangChain in one call
- Fast enough for document sets up to ~100k chunks on a laptop

**Trade-off:** Does not scale to production workloads; not distributed.
For a portfolio demo this is the correct choice — operational simplicity
beats raw scalability.

**Migration path:** If the project grows, swap `Chroma` for `PineconeVectorStore`
in a single constructor call; the rest of the pipeline stays identical.

---

## ADR-003 · Embedding Model: all-MiniLM-L6-v2

**Decision:** `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace.

**Reasons:**
- Runs entirely on CPU — no GPU required, no API cost
- 384-dimensional vectors — small enough to be fast, large enough to be accurate
- State-of-the-art for English; acceptable for Portuguese (bilingual corpus)

**Trade-off:** Lower accuracy on Portuguese than a dedicated multilingual model
(e.g., `paraphrase-multilingual-MiniLM-L12-v2`). ADR-007 will revisit this
if retrieval quality is poor.

---

## ADR-004 · Chunking: RecursiveCharacterTextSplitter (1000 / 200)

**Decision:** 1000-character chunks with 200-character overlap.

**Reasoning:**
- 1000 chars ≈ one dense paragraph — enough context for Claude, small enough
  to be precise when retrieved
- 200-char overlap prevents concepts from being split across a boundary and
  lost in both chunks
- `RecursiveCharacterTextSplitter` prefers `\n\n` → `\n` → ` ` before
  cutting mid-word, preserving paragraph semantics

**What we traded off:** Semantic chunking (e.g., splitting by heading) would
give higher precision but requires document-type-specific parsers. Will revisit
in Phase 4 (multi-document support).

---

## ADR-005 · Backend Framework: FastAPI

**Decision:** FastAPI (Phase 6) over Flask, Django, or Express.

**Reasons:**
- Auto-generates OpenAPI docs (useful for portfolio presentation)
- Native `async`/`await` — matches LangChain's async chain calls
- Pydantic validation keeps request/response contracts explicit

---

## ADR-006 · Frontend: Streamlit (Phase 8)

**Decision:** Streamlit over React, Gradio, or a bare HTML page.

**Reasons:**
- Zero JS — write UI in pure Python
- Chat interface built-in (`st.chat_message`, `st.chat_input`)
- Deploys to Streamlit Cloud for free (useful for sharing the portfolio)

**Trade-off:** Limited customization vs. React. This is a technical demo, not
a product — Streamlit's fast iteration beats React's flexibility here.

---

## Phase Roadmap

| Phase | Milestone | Status |
|-------|-----------|--------|
| 1 | Local pipeline: load → chunk → embed → ChromaDB → retrieve | ✅ Done |
| 2 | Add Claude generation: retrieve + prompt → answer | ✅ Done |
| 3 | Persist vector store to disk (survive restarts) | ✅ Done |
| 4 | Multi-document support + source citation | ✅ Done |
| 5 | Chat history within session | ✅ Done |
| 6 | FastAPI REST backend | ✅ Done |
| 7 | Document upload endpoint | ✅ Done |
| 8 | Streamlit UI | ✅ Done |
| 9 | Testing (pytest + unit + integration) | ✅ Done |
| 10 | Docker Compose (FastAPI + Streamlit) | ✅ Done |
