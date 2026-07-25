"""
Phase 5 — RAG com Histórico de Chat
=====================================
Novidades desta fase:
  1. O sistema lembra das perguntas anteriores dentro da sessão
  2. Perguntas de acompanhamento ("E para dependentes?") são reformuladas
     automaticamente em perguntas completas antes do retrieval
  3. O histórico é mantido em memória (some ao fechar o script)

Pipeline completo:
  nova_pergunta + histórico
        │
        ▼
  [Reformulador] — LLM leve que reescreve a pergunta como standalone
        │
        ▼
  pergunta_standalone → [Retriever] → chunks
        │
        ▼
  [Claude] ← chunks + histórico + pergunta original
        │
        ▼
  resposta → salva no histórico → próxima rodada
"""

import os
import textwrap
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser


# ── 1. CONFIGURAÇÃO ──────────────────────────────────────────────────────────

load_dotenv()

# Reusa o banco da Fase 4 (já tem os dois documentos indexados)
PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db_v4"

COLLECTION_NAME = "multi_doc_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.1-8b-instant"
K_CHUNKS        = 4

# ── PROMPT 1: Reformulador ────────────────────────────────────────────────────
# Recebe o histórico e a nova pergunta.
# Reescreve a pergunta como se fosse a primeira — sem depender do histórico.
# Se a pergunta já for standalone ("Quais são os benefícios?"), retorna igual.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Dado um histórico de conversa e a pergunta mais recente do usuário, "
        "reformule a pergunta para que ela seja completamente compreensível "
        "sem precisar do histórico. "
        "Se a pergunta já for autossuficiente, retorne-a sem modificações. "
        "NÃO responda à pergunta — apenas reformule ou repita.",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# ── PROMPT 2: Resposta com histórico ─────────────────────────────────────────
# Técnica: prefilled messages + stop sequences
#   • ("ai", "") → prefill vazio: força o modelo a começar respondendo diretamente,
#     sem preamble ("Com base nos documentos...", "Claro!", etc.)
#   • stop=[...] aplicado no LLM → interrompe antes de seções de notas/fontes
#     que o modelo às vezes adiciona espontaneamente
# Técnica: prefilled messages + stop sequences
# O prefill é implementado via instrução direta no system prompt ("Comece sua resposta
# diretamente") — equivalente funcional para modelos OpenAI-compat como Groq,
# onde passar um AIMessage vazio como último turno pode truncar o astream.
# As stop sequences são aplicadas via llm.bind(stop=[...]) no __init__.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Você é um assistente que responde perguntas sobre documentos internos. "
        "Use APENAS as informações dos trechos abaixo para responder. "
        "Cite o arquivo de origem entre colchetes, ex: [politica_empresa.txt]. "
        "Se a resposta não estiver nos trechos, diga que não encontrou. "
        "Comece sua resposta diretamente, sem introduções como 'Com base nos documentos' ou 'Claro!'. "
        "Leve em conta o histórico da conversa para manter coerência.\n\n"
        "Trechos recuperados:\n{context}",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# ── 2. CARREGAMENTO DO BANCO (criado na Fase 4) ───────────────────────────────

def load_vector_store(embeddings) -> Chroma:
    db_file = PERSIST_DIR / "chroma.sqlite3"
    if not db_file.exists():
        raise FileNotFoundError(
            f"Banco não encontrado em '{PERSIST_DIR}'.\n"
            "Execute primeiro: python src/phase4_multi_doc.py"
        )
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    count = vs._collection.count()
    print(f"[store] {count} chunk(s) carregado(s) do disco.")
    return vs


# ── 3. CONVERSATIONAL RAG ────────────────────────────────────────────────────

class ConversationalRAG:
    """
    Encapsula o pipeline RAG com histórico de chat.

    Por que uma classe aqui?
    A classe mantém estado entre chamadas (histórico, retriever, llm).
    Uma função pura não consegue guardar o histórico entre invocações.
    """

    def __init__(self, vector_store: Chroma, llm: ChatGroq):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": K_CHUNKS})
        self.llm = llm
        self.history = ChatMessageHistory()  # começa vazio, cresce durante a sessão

        # Sub-chain do reformulador
        # StrOutputParser converte AIMessage → string pura
        self.contextualizer = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()

        # stop sequences: interrompe o LLM antes que ele adicione seções extras
        _llm_with_stop = llm.bind(stop=["\n\nNota:", "\n\nObservação:", "\n\n---"])

        # Sub-chain da resposta final
        self.answerer = ANSWER_PROMPT | _llm_with_stop | StrOutputParser()

    def _format_context(self, docs) -> str:
        """Formata os chunks com nome do arquivo para o Claude citar."""
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[Trecho {i} — {source}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    def _get_sources(self, docs) -> list[str]:
        seen = []
        for doc in docs:
            s = doc.metadata.get("source", "desconhecido")
            if s not in seen:
                seen.append(s)
        return seen

    def ask(self, question: str) -> dict:
        """
        Fluxo completo para uma pergunta:
        1. Se houver histórico → reformula a pergunta
        2. Usa a pergunta reformulada para buscar no vector store
        3. Gera resposta com Claude (recebe histórico + contexto)
        4. Salva pergunta e resposta no histórico
        """
        chat_history = self.history.messages

        # Passo 1: reformulação (só faz sentido quando há histórico)
        if chat_history:
            standalone_question = self.contextualizer.invoke({
                "chat_history": chat_history,
                "question": question,
            })
            print(f"\n[reform] '{question}'")
            print(f"      → '{standalone_question}'")
        else:
            # Primeira pergunta: já é standalone, não precisa reformular
            standalone_question = question

        # Passo 2: retrieval com a pergunta reformulada
        docs = self.retriever.invoke(standalone_question)
        context = self._format_context(docs)
        sources = self._get_sources(docs)

        # Passo 3: resposta com Claude (usa pergunta ORIGINAL + histórico)
        # Por quê a pergunta original e não a standalone?
        # Para manter o tom natural da conversa ("E para dependentes?"
        # é mais natural que a versão reformulada completa).
        answer = self.answerer.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question,
        })

        # Passo 4: salva no histórico para a próxima rodada
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)

        return {"answer": answer, "sources": sources}

    async def ask_stream(self, question: str) -> AsyncGenerator:
        """
        Versão streaming de ask().
        Yields: tokens de texto (str) enquanto o LLM gera a resposta,
                seguido de um dict {"sources": [...]} como último item.

        O chamador distingue pelo tipo:
          str  → token para exibir em tempo real
          dict → metadados finais (fontes consultadas)
        """
        chat_history = self.history.messages

        if chat_history:
            standalone_question = self.contextualizer.invoke({
                "chat_history": chat_history,
                "question": question,
            })
        else:
            standalone_question = question

        docs = self.retriever.invoke(standalone_question)
        context = self._format_context(docs)
        sources = self._get_sources(docs)

        full_answer = ""
        async for token in self.answerer.astream({
            "context": context,
            "chat_history": chat_history,
            "question": question,
        }):
            full_answer += token
            yield token

        self.history.add_user_message(question)
        self.history.add_ai_message(full_answer)
        yield {"sources": sources}

    def clear_history(self):
        """Limpa o histórico para começar uma nova conversa."""
        self.history.clear()
        print("[hist]  Histórico limpo. Nova conversa iniciada.")

    def show_history(self):
        """Mostra o histórico acumulado."""
        if not self.history.messages:
            print("[hist]  Histórico vazio.")
            return
        print(f"\n[hist]  {len(self.history.messages)} mensagem(ns) no histórico:")
        for msg in self.history.messages:
            role = "Você" if isinstance(msg, HumanMessage) else "IA"
            preview = msg.content[:80].replace("\n", " ")
            print(f"  [{role}] {preview}...")


# ── 4. LOOP INTERATIVO ────────────────────────────────────────────────────────

def print_result(result: dict):
    bar = "─" * 70
    print(bar)
    print(textwrap.fill(result["answer"].strip(), width=70))
    print(f"\nFontes: {', '.join(result['sources'])}")
    print()


def interactive_loop(rag: ConversationalRAG):
    bar = "=" * 70
    print(f"\n{bar}")
    print("  Chat com RAG — comandos especiais:")
    print("  /limpar  → nova conversa  |  /historico → ver mensagens  |  sair → encerrar")
    print(f"{bar}\n")

    while True:
        query = input("Você: ").strip()

        if not query:
            continue
        if query.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break
        if query.lower() == "/limpar":
            rag.clear_history()
            continue
        if query.lower() == "/historico":
            rag.show_history()
            continue

        result = rag.ask(query)
        print_result(result)


# ── 5. PIPELINE PRINCIPAL ────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  FASE 5 — RAG com Histórico de Chat")
    print("=" * 70 + "\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY não encontrada no .env")

    print("[embed] Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = load_vector_store(embeddings)

    print(f"[llm]   Conectando ao Groq ({GROQ_MODEL})...")
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, max_tokens=1024)

    rag = ConversationalRAG(vector_store, llm)
    print("[llm]   Pronto.\n")

    # Demo automática mostrando perguntas de acompanhamento
    print("--- Demonstração: Perguntas de Acompanhamento ---\n")

    demo_pairs = [
        ("Qual é o plano de saúde oferecido?", False),
        ("E para dependentes? Qual o custo?", True),   # ← acompanhamento
        ("Quando acontece a rescisão do contrato de TI?", False),
        ("Qual é a multa por atraso de pagamento?", True),  # ← acompanhamento
    ]

    for question, is_followup in demo_pairs:
        label = "(acompanhamento)" if is_followup else ""
        print(f"Você: {question} {label}")
        result = rag.ask(question)
        print_result(result)

    # Limpa para o modo interativo começar do zero
    rag.clear_history()

    interactive_loop(rag)


if __name__ == "__main__":
    main()
