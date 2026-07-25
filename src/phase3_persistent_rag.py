"""
Phase 3 — RAG com Persistência em Disco
=========================================
Novidade desta fase: o ChromaDB salva os vetores no disco.
Na primeira execução: processa o documento e salva.
Nas seguintes: carrega do disco diretamente, sem reprocessar.

Estrutura em disco criada:
  data/chroma_db/
    ├── chroma.sqlite3      ← metadados e textos
    └── <uuid>/             ← vetores binários (arquivos .bin)
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

load_dotenv()

SAMPLE_DOC = Path(__file__).parent.parent / "data" / "sample_docs" / "politica_empresa.txt"

# Onde o ChromaDB vai salvar os arquivos no disco
# Path relativo à raiz do projeto
PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db"

COLLECTION_NAME = "politica_empresa"  # nome da coleção dentro do ChromaDB

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"
K_CHUNKS = 3

RAG_PROMPT = ChatPromptTemplate.from_template("""
Você é um assistente especializado em responder perguntas sobre documentos internos.
Responda usando APENAS as informações do contexto fornecido.
Se a resposta não estiver no contexto, diga "Não encontrei essa informação no documento."
Seja direto e objetivo.

Contexto:
{context}

Pergunta: {question}

Resposta:""")


# ── 2. EMBEDDINGS (reutilizado em ingestão e consulta) ───────────────────────

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── 3. INGESTÃO (só roda se o banco não existe) ───────────────────────────────

def ingest_document(path: Path, embeddings) -> Chroma:
    """
    Processa o documento e salva os vetores no disco.
    Só é chamado quando PERSIST_DIR não existe ou está vazio.
    """
    print("[ingest] Processando documento pela primeira vez...")

    # Carrega
    suffix = path.suffix.lower()
    loader = TextLoader(str(path), encoding="utf-8") if suffix == ".txt" else PyPDFLoader(str(path))
    documents = loader.load()
    print(f"[ingest] {len(documents)} documento(s) carregado(s)")

    # Divide em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] {len(chunks)} chunk(s) gerado(s)")

    # Gera embeddings e salva no disco
    # persist_directory faz o ChromaDB gravar automaticamente após from_documents
    print(f"[ingest] Gerando embeddings e salvando em '{PERSIST_DIR}'...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    print(f"[ingest] Salvo com sucesso. Próximas execuções serão instantâneas.")
    return vector_store


# ── 4. CARREGAMENTO DO DISCO (roda quando o banco já existe) ─────────────────

def load_from_disk(embeddings) -> Chroma:
    """
    Carrega o vector store do disco sem reprocessar nada.
    É equivalente a abrir um banco de dados já existente.
    """
    print(f"[load]  Banco encontrado em '{PERSIST_DIR}'. Carregando do disco...")
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    count = vector_store._collection.count()
    print(f"[load]  {count} chunk(s) carregado(s) do disco.")
    return vector_store


# ── 5. DECISÃO: INGERIR OU CARREGAR ──────────────────────────────────────────

def get_vector_store(embeddings) -> Chroma:
    """
    Ponto central de decisão:
    - Se o banco existe no disco → carrega
    - Se não existe → processa e salva

    Isso é chamado de "idempotência": rodar várias vezes dá o mesmo resultado
    sem duplicar o trabalho.
    """
    # Verifica se o diretório existe E contém o arquivo principal do ChromaDB
    db_file = PERSIST_DIR / "chroma.sqlite3"
    if PERSIST_DIR.exists() and db_file.exists():
        return load_from_disk(embeddings)
    else:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        return ingest_document(SAMPLE_DOC, embeddings)


# ── 6. RAG CHAIN (igual Fase 2) ──────────────────────────────────────────────

def build_rag_chain(vector_store: Chroma):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY não encontrada no .env")

    retriever = vector_store.as_retriever(search_kwargs={"k": K_CHUNKS})

    llm = ChatAnthropic(
        model=CLAUDE_MODEL,
        temperature=0,
        max_tokens=1024,
    )

    def format_context(docs):
        parts = []
        for i, doc in enumerate(docs, 1):
            source = Path(doc.metadata.get("source", "doc")).name
            parts.append(f"[Trecho {i} — {source}]\n{doc.page_content.strip()}")
        return "\n\n".join(parts)

    return (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


# ── 7. LOOP INTERATIVO ────────────────────────────────────────────────────────

def interactive_loop(rag_chain):
    bar = "─" * 70
    print(f"\n{bar}")
    print("  Modo interativo — digite sua pergunta (ou 'sair' para encerrar)")
    print(f"{bar}\n")

    while True:
        query = input("Pergunta: ").strip()
        if not query or query.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break

        answer = rag_chain.invoke(query)
        print(f"\n{bar}")
        wrapped = textwrap.fill(answer.strip(), width=70)
        print(wrapped)
        print(f"{bar}\n")


# ── 8. PIPELINE PRINCIPAL ────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  FASE 3 — RAG com Persistência em Disco")
    print("=" * 70 + "\n")

    # Embeddings (necessário tanto para ingerir quanto para carregar)
    print("[embed] Carregando modelo de embeddings...")
    embeddings = build_embeddings()

    # Decide automaticamente: ingerir ou carregar do disco
    vector_store = get_vector_store(embeddings)

    # Monta a chain com Claude
    print(f"\n[llm]   Conectando ao Claude ({CLAUDE_MODEL})...")
    rag_chain = build_rag_chain(vector_store)
    print("[llm]   Pronto.\n")

    interactive_loop(rag_chain)


if __name__ == "__main__":
    main()
