"""
Phase 8 — Interface Streamlit estilo ChatGPT
=============================================
Roda separado da API FastAPI.

Para usar:
  Terminal 1: uvicorn src.api.main:app --port 8000
  Terminal 2: streamlit run src/app.py
"""

import json
import requests
import streamlit as st

# ── CONFIGURAÇÃO DA PÁGINA ────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
API_BASE = os.getenv("API_BASE", "http://localhost:8000")

# ── CSS CUSTOMIZADO ───────────────────────────────────────────────────────────
# Aproxima o visual do ChatGPT: fonte limpa, bubbles sutis, input fixo

st.markdown("""
<style>
/* Fundo geral mais escuro */
.stApp { background-color: #0d1117; }

/* Sidebar mais escura */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* Título da sidebar */
.sidebar-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e6edf3;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 1rem;
}

/* Seções da sidebar */
.sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.2rem 0 0.4rem 0;
}

/* Card de documento indexado */
.doc-card {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    color: #c9d1d9;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Badge de fonte na resposta da IA */
.source-badge {
    display: inline-block;
    background: #21262d;
    border: 1px solid #388bfd;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #58a6ff;
    margin: 0.4rem 0.2rem 0 0;
}

/* Mensagem de boas-vindas */
.welcome-box {
    text-align: center;
    padding: 3rem 1rem;
    color: #8b949e;
}
.welcome-box h2 { color: #e6edf3; margin-bottom: 0.5rem; }

/* Remove borda padrão do chat_input */
[data-testid="stChatInput"] textarea {
    background-color: #21262d !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

/* Bubbles de mensagem */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS DE API ────────────────────────────────────────────────────────────

def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.json() if r.ok else None
    except requests.exceptions.ConnectionError:
        return None


def api_chat_stream(question: str):
    """
    Chama /chat/stream e retorna (generator_de_tokens, lista_de_fontes).
    O generator é passado para st.write_stream; fontes são preenchidas
    via closure durante a iteração e lidas depois que o stream termina.
    """
    sources: list[str] = []

    def token_generator():
        try:
            with requests.post(
                f"{API_BASE}/chat/stream",
                json={"question": question},
                stream=True,
                timeout=60,
            ) as r:
                for raw_line in r.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if "token" in event:
                        yield event["token"]
                    elif event.get("done"):
                        sources.extend(event.get("sources", []))
                    elif "error" in event:
                        yield f"\n\n⚠️ {event['error']}"
        except requests.exceptions.ConnectionError:
            yield "⚠️ Não foi possível conectar à API."

    return token_generator(), sources


def api_list_documents() -> list[dict]:
    try:
        r = requests.get(f"{API_BASE}/documents", timeout=5)
        return r.json().get("documents", []) if r.ok else []
    except requests.exceptions.ConnectionError:
        return []


def api_upload(files) -> dict | None:
    try:
        file_tuples = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        r = requests.post(f"{API_BASE}/documents/upload", files=file_tuples, timeout=120)
        return r.json() if r.ok else {"message": f"Erro: {r.text}", "indexed": [], "skipped": []}
    except requests.exceptions.ConnectionError:
        return None


def api_clear_history():
    try:
        requests.delete(f"{API_BASE}/chat/history", timeout=5)
    except requests.exceptions.ConnectionError:
        pass


# ── ESTADO DA SESSÃO ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # lista de {"role", "content", "sources"}

if "api_ok" not in st.session_state:
    st.session_state.api_ok = False


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-title">📄 RAG Chat</div>', unsafe_allow_html=True)

    # Status da API
    health = api_health()
    if health:
        st.session_state.api_ok = True
        st.success(f"API online · {health['chunks_indexed']} chunks", icon="✅")
    else:
        st.session_state.api_ok = False
        st.error("API offline", icon="🔴")
        st.caption("Rode: `uvicorn src.api.main:app --port 8000`")

    # ── Upload de documentos ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Upload de documentos</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Arraste ou selecione arquivos",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.button("Indexar documentos", use_container_width=True, type="primary"):
        if not st.session_state.api_ok:
            st.error("API offline — não é possível indexar.")
        else:
            with st.spinner("Indexando..."):
                result = api_upload(uploaded)
            if result:
                if result.get("indexed"):
                    st.success(result["message"])
                else:
                    st.info(result["message"])
                st.rerun()

    # ── Documentos indexados ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Documentos indexados</div>', unsafe_allow_html=True)

    docs = api_list_documents()
    if docs:
        for doc in docs:
            st.markdown(
                f'<div class="doc-card">📄 {doc["filename"]}'
                f'<span style="color:#8b949e;font-size:0.75rem;margin-left:auto">'
                f'{doc["size_kb"]} KB</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Nenhum documento indexado ainda.")

    # ── Ações ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Conversa</div>', unsafe_allow_html=True)

    if st.button("🗑 Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        api_clear_history()
        st.rerun()

    st.markdown("---")
    st.caption("RAG Portfolio · Fase 8")
    st.caption("Groq llama-3.1-8b · ChromaDB · LangChain")


# ── ÁREA DE CHAT PRINCIPAL ────────────────────────────────────────────────────

# Mensagem de boas-vindas quando não há histórico
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <h2>📄 RAG Chat</h2>
        <p>Envie documentos na barra lateral e faça perguntas sobre eles.<br>
        As respostas vêm diretamente do conteúdo dos seus arquivos.</p>
    </div>
    """, unsafe_allow_html=True)

# Renderiza o histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Mostra as fontes abaixo das mensagens da IA
        if msg["role"] == "assistant" and msg.get("sources"):
            sources_html = "".join(
                f'<span class="source-badge">📎 {s}</span>'
                for s in msg["sources"]
            )
            st.markdown(
                f'<div style="margin-top:0.5rem">{sources_html}</div>',
                unsafe_allow_html=True,
            )

# ── INPUT DO USUÁRIO ─────────────────────────────────────────────────────────

if question := st.chat_input(
    "Faça uma pergunta sobre seus documentos...",
    disabled=not st.session_state.api_ok,
):
    # Exibe a mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "sources": [],
    })

    # Streaming: exibe tokens em tempo real com st.write_stream
    with st.chat_message("assistant"):
        token_gen, sources = api_chat_stream(question)
        answer = st.write_stream(token_gen)   # retorna o texto completo ao fim

        if not answer:
            answer = "⚠️ Não foi possível conectar à API. Verifique se ela está rodando."

        if sources:
            sources_html = "".join(
                f'<span class="source-badge">📎 {s}</span>'
                for s in sources
            )
            st.markdown(
                f'<div style="margin-top:0.5rem">{sources_html}</div>',
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
