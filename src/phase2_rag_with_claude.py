"""
Phase 2 — RAG com Claude
=========================
Adiciona geração ao pipeline da Fase 1:
  load → chunk → embed → vectorstore → retrieve → prompt → Claude → resposta

Requer: ANTHROPIC_API_KEY no arquivo .env
"""

import os
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ── 1. CONFIGURAÇÃO ──────────────────────────────────────────────────────────

load_dotenv()  # Lê .env e injeta ANTHROPIC_API_KEY no os.environ

SAMPLE_DOC = Path(__file__).parent.parent / "data" / "sample_docs" / "politica_empresa.txt"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modelo Claude — confirmado nos docs do langchain-anthropic (Context7)
CLAUDE_MODEL = "claude-sonnet-4-6"

# Quantos chunks recuperar por query
K_CHUNKS = 3

# Prompt que instrui o Claude a usar APENAS o contexto fornecido
# {context} = chunks recuperados  |  {question} = pergunta do usuário
RAG_PROMPT = ChatPromptTemplate.from_template("""
Você é um assistente especializado em responder perguntas sobre documentos internos.
Responda a pergunta abaixo usando APENAS as informações do contexto fornecido.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação no documento."
Seja direto e objetivo.

Contexto:
{context}

Pergunta: {question}

Resposta:""")

# Perguntas de demo (modo não-interativo)
DEMO_QUERIES = [
    "Quantos dias de férias tenho por ano?",
    "Quanto recebo de vale-refeição por dia?",
    "Posso trabalhar 100% em home office?",
    "Quando acontece a distribuição de PLR?",
    "O que acontece se eu violar o NDA?",
    "Quanto custa o plano de saúde para mim?",  # resposta está no doc
    "Qual é a data de fundação da empresa?",    # NÃO está no doc — teste do "não sei"
]


# ── 2. PIPELINE DE INGESTÃO (igual Fase 1) ───────────────────────────────────

def load_document(path: Path):
    suffix = path.suffix.lower()
    loader = TextLoader(str(path), encoding="utf-8") if suffix == ".txt" else PyPDFLoader(str(path))
    docs = loader.load()
    print(f"[load]  {len(docs)} documento(s) de '{path.name}'")
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[split] {len(chunks)} chunk(s)")
    return chunks


def build_embeddings():
    print(f"[embed] Carregando '{EMBEDDING_MODEL}'...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks, embeddings):
    print(f"[store] Indexando {len(chunks)} chunk(s)...")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="phase2_demo",
    )


# ── 3. CONSTRUÇÃO DA RAG CHAIN (LCEL) ────────────────────────────────────────

def build_rag_chain(vector_store):
    """
    Monta o pipeline completo usando LCEL (LangChain Expression Language).

    LCEL usa o operador | (pipe) para encadear componentes:
      retriever | prompt | llm | parser

    Fluxo para cada query:
      1. retriever.invoke(question)  → [Document, Document, Document]
      2. format_context(docs)        → string com os 3 chunks concatenados
      3. RAG_PROMPT.invoke(...)      → ChatPromptValue com o prompt completo
      4. llm.invoke(prompt)          → AIMessage com a resposta do Claude
      5. StrOutputParser()           → string final

    Por que RunnablePassthrough?
    O LCEL precisa passar a question tanto para o retriever quanto para o prompt.
    RunnablePassthrough() copia o input sem modificar — é um "fio direto".
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY não encontrada.\n"
            "Copie .env.example para .env e coloque sua chave."
        )

    # Retriever: interface de busca sobre o vector store
    # as_retriever() converte Chroma → objeto com método .invoke(query)
    retriever = vector_store.as_retriever(search_kwargs={"k": K_CHUNKS})

    # LLM: Claude via langchain-anthropic
    # temperature=0 → respostas determinísticas (menos criativas, mais factuais)
    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        temperature=0,
        max_tokens=1024,
    )

    def format_context(docs):
        """
        Converte lista de Documents em string única para o {context}.
        Adiciona separador entre chunks para o Claude distinguir as fontes.
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            source = Path(doc.metadata.get("source", "doc")).name
            parts.append(f"[Trecho {i} — {source}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    # Chain LCEL:
    # - "context" é processado pelo retriever + formatador
    # - "question" passa direto (RunnablePassthrough) para o prompt
    rag_chain = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── 4. EXIBIÇÃO ───────────────────────────────────────────────────────────────

def print_answer(query: str, answer: str):
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"PERGUNTA: {query}")
    print(bar)
    wrapped = textwrap.fill(answer.strip(), width=70)
    print(wrapped)
    print()


# ── 5. PIPELINE PRINCIPAL ────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  FASE 2 — RAG com Claude")
    print("=" * 70 + "\n")

    if not SAMPLE_DOC.exists():
        raise FileNotFoundError(f"Documento não encontrado: {SAMPLE_DOC}")

    # Ingestão (igual Fase 1)
    documents = load_document(SAMPLE_DOC)
    chunks = split_documents(documents)
    embeddings = build_embeddings()
    vector_store = build_vector_store(chunks, embeddings)

    # NOVO: monta a chain com Claude
    print(f"\n[llm]   Conectando ao Claude ({CLAUDE_MODEL})...")
    rag_chain = build_rag_chain(vector_store)
    print("[llm]   Chain pronta.\n")

    print("--- Demonstração: Perguntas e Respostas ---")
    for query in DEMO_QUERIES:
        answer = rag_chain.invoke(query)
        print_answer(query, answer)

    # Modo interativo: permite fazer perguntas livres
    print("=" * 70)
    print("  Modo interativo — digite sua pergunta (ou 'sair' para encerrar)")
    print("=" * 70)

    while True:
        query = input("\nPergunta: ").strip()
        if not query or query.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break
        answer = rag_chain.invoke(query)
        print_answer(query, answer)


if __name__ == "__main__":
    main()
