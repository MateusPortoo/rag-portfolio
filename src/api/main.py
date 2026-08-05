"""
Phase 6 — FastAPI Backend
==========================
Transforma o RAG das fases anteriores em uma API REST.

Endpoints:
  GET  /health            → verifica se a API está no ar
  POST /chat              → envia uma pergunta (resposta completa), recebe resposta + fontes
  GET  /chat/history      → lista as mensagens trocadas na sessão
  POST /chat/stream       → streaming SSE token a token
  GET  /chat/history      → lista as mensagens trocadas na sessão
  DELETE /chat/history    → limpa o histórico (começa nova conversa)

Para rodar:
  uvicorn src.api.main:app --reload --port 8000

Depois abra: http://localhost:8000/docs  (documentação interativa automática)
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Adiciona a raiz do projeto ao path para importar módulos locais
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from src.api.models import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    HistoryMessage,
    StatusResponse,
)
from src.phase5_chat_history import ConversationalRAG   # reusa a classe da Fase 5
from src.phase7_hyde import build_hyde_retriever
from src.phase8_reranker import build_reranking_retriever
from src.api.documents import router as documents_router

load_dotenv()

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────

PERSIST_DIR     = Path(__file__).parent.parent.parent / "data" / "chroma_db_v4"
COLLECTION_NAME = "multi_doc_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"


# ── ESTADO GLOBAL DA APLICAÇÃO ───────────────────────────────────────────────
# Para uma API single-user (sem autenticação, fase de portfólio), um único
# objeto RAG global é suficiente.
# Em produção multi-usuário, cada sessão teria seu próprio ConversationalRAG.

app_state: dict = {}


# ── LIFESPAN: startup e shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    lifespan() substitui os decorators @app.on_event("startup") deprecados.
    Tudo antes do 'yield' roda ao iniciar. Tudo depois roda ao desligar.

    Por que inicializar aqui e não no primeiro request?
    Carregar o modelo de embeddings leva ~3 segundos. Se carregássemos no
    primeiro request, aquele usuário esperaria 3s a mais. Fazer no startup
    garante que a API só sobe quando estiver pronta.
    """
    print("[startup] Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    db_file = PERSIST_DIR / "chroma.sqlite3"
    if not db_file.exists():
        raise RuntimeError(
            f"Banco vetorial não encontrado em '{PERSIST_DIR}'.\n"
            "Execute primeiro: python src/phase4_multi_doc.py"
        )

    print("[startup] Carregando banco vetorial do disco...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    chunk_count = vector_store._collection.count()
    print(f"[startup] {chunk_count} chunks indexados.")

    print(f"[startup] Conectando ao Groq ({GROQ_MODEL})...")
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=1024)

    print("[startup] Construindo pipeline HyDE + reranking...")
    hyde_retriever = build_hyde_retriever(vector_store, llm, embeddings, k=15)
    reranking_retriever = build_reranking_retriever(hyde_retriever, k_final=5, k_candidates=15, threshold=0.65)

    # Armazena no estado global para os endpoints acessarem
    app_state["rag"] = ConversationalRAG(vector_store, llm, retriever=reranking_retriever)
    app_state["chunk_count"] = chunk_count

    print("[startup] API pronta! Acesse http://localhost:8000/docs\n")

    yield  # ← API roda aqui

    # Shutdown (cleanup se necessário)
    print("[shutdown] Encerrando...")
    app_state.clear()


# ── CRIAÇÃO DA APP ────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Q&A API",
    description="API para perguntas e respostas sobre documentos usando RAG + Groq (Llama 3.1 8B).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: permite que o Streamlit (Fase 8) faça requisições para esta API
# Em produção, substitua "*" pelo domínio específico do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registra o router de documentos (upload + listagem)
app.include_router(documents_router)


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=StatusResponse, tags=["Sistema"])
async def health_check():
    """
    Verifica se a API está no ar e o banco vetorial carregado.
    Útil para monitoramento e para o frontend saber se pode fazer requests.
    """
    rag: ConversationalRAG = app_state.get("rag")
    if not rag:
        raise HTTPException(status_code=503, detail="RAG não inicializado.")

    return StatusResponse(
        status="ok",
        chunks_indexed=app_state.get("chunk_count", 0),
        model=GROQ_MODEL,
    )


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Envia uma pergunta sobre os documentos indexados.

    O sistema:
    1. Reformula a pergunta usando o histórico da conversa
    2. Busca os chunks mais relevantes no banco vetorial
    3. Passa os chunks + histórico para o Llama 3.1 8B via Groq gerar uma resposta
    4. Retorna a resposta e os arquivos consultados
    """
    rag: ConversationalRAG = app_state.get("rag")
    if not rag:
        raise HTTPException(status_code=503, detail="RAG não inicializado.")

    try:
        result = rag.ask(request.question)
    except Exception as e:
        # Loga o erro real no servidor mas não expõe detalhes ao cliente
        print(f"[error] Falha ao processar pergunta: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erro ao processar a pergunta. Tente novamente.",
        )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        question=request.question,
    )


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Streaming de resposta via Server-Sent Events (SSE).

    Cada evento tem formato: data: <json>\\n\\n

    Tipos de evento:
      {"token": "..."} → fragmento de texto gerado pelo LLM em tempo real
      {"done": true, "sources": [...]} → fim da resposta com fontes consultadas
      {"error": "..."} → falha durante a geração
    """
    rag: ConversationalRAG = app_state.get("rag")
    if not rag:
        raise HTTPException(status_code=503, detail="RAG não inicializado.")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in rag.ask_stream(request.question):
                if isinstance(chunk, str):
                    yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
                elif isinstance(chunk, dict):
                    yield f"data: {json.dumps({'done': True, 'sources': chunk['sources']}, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"[error] Streaming falhou: {e}")
            yield f"data: {json.dumps({'error': 'Erro ao gerar resposta.'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/chat/history", response_model=HistoryResponse, tags=["Chat"])
async def get_history():
    """
    Retorna o histórico de mensagens da conversa atual.
    O histórico existe apenas em memória — é perdido ao reiniciar a API.
    """
    rag: ConversationalRAG = app_state.get("rag")
    if not rag:
        raise HTTPException(status_code=503, detail="RAG não inicializado.")

    messages = []
    for msg in rag.history.messages:
        if isinstance(msg, HumanMessage):
            messages.append(HistoryMessage(role="human", content=msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(HistoryMessage(role="ai", content=msg.content))

    return HistoryResponse(messages=messages, total=len(messages))


@app.delete("/chat/history", tags=["Chat"])
async def clear_history():
    """
    Limpa o histórico da conversa. Útil para começar um novo contexto
    sem reiniciar a API (que demoraria por causa do carregamento do modelo).
    """
    rag: ConversationalRAG = app_state.get("rag")
    if not rag:
        raise HTTPException(status_code=503, detail="RAG não inicializado.")

    rag.clear_history()
    return {"message": "Histórico limpo com sucesso."}

