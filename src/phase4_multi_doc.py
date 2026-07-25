"""
Phase 4 — Multi-Documento com Citação de Fonte
================================================
Novidades desta fase:
  1. Carrega TODOS os arquivos de data/sample_docs/ (TXT, PDF, DOCX)
  2. Cada chunk mantém o metadado 'source' com o nome do arquivo
  3. A resposta cita de qual arquivo cada informação veio
  4. Detecta novos arquivos e adiciona ao banco sem reprocessar os antigos
"""

import os
import json
import textwrap
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ── 1. CONFIGURAÇÃO ──────────────────────────────────────────────────────────

load_dotenv()

DOCS_DIR    = Path(__file__).parent.parent / "data" / "sample_docs"
PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db_v4"

# Arquivo que registra quais docs já foram indexados
# Isso evita reprocessar arquivos antigos quando um novo é adicionado
MANIFEST_FILE = PERSIST_DIR / "manifest.json"

COLLECTION_NAME  = "multi_doc_collection"
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL     = "claude-sonnet-4-6"
K_CHUNKS         = 4   # mais chunks porque temos mais documentos

# Prompt adaptado para citar fontes
# A instrução "cite o arquivo" é crucial — sem ela o Claude pode omitir
RAG_PROMPT = ChatPromptTemplate.from_template("""
Você é um assistente que responde perguntas sobre documentos internos.
Responda usando APENAS as informações dos trechos abaixo.
Para cada informação que usar, cite o arquivo entre colchetes, por exemplo: [politica_empresa.txt]
Se a resposta não estiver nos trechos, diga "Não encontrei essa informação nos documentos."

Trechos recuperados:
{context}

Pergunta: {question}

Resposta (cite as fontes):""")


# ── 2. MANIFEST — controle de quais arquivos já foram indexados ──────────────

def load_manifest() -> set:
    """Retorna conjunto de nomes de arquivo já indexados."""
    if MANIFEST_FILE.exists():
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        return set(data.get("indexed_files", []))
    return set()


def save_manifest(indexed_files: set):
    """Salva o conjunto atualizado de arquivos indexados."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"indexed_files": sorted(indexed_files)}
    MANIFEST_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 3. CARREGAMENTO DE DOCUMENTOS ────────────────────────────────────────────

def load_documents(file_paths: list[Path]) -> list:
    """
    Carrega uma lista de arquivos usando o loader correto para cada tipo.
    Cada Document recebe metadata["source"] com o caminho completo.
    Depois simplificamos para apenas o nome do arquivo.
    """
    all_docs = []
    for path in file_paths:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(str(path))
        else:
            print(f"[load]  Formato não suportado, pulando: {path.name}")
            continue

        docs = loader.load()

        # Simplifica o source para apenas o nome do arquivo
        # (o LangChain coloca o caminho absoluto por padrão)
        for doc in docs:
            doc.metadata["source"] = path.name

        all_docs.extend(docs)
        print(f"[load]  {path.name} → {len(docs)} página(s)/documento(s)")

    return all_docs


def get_files_in_dir() -> dict[str, Path]:
    """Retorna dict {nome_arquivo: path} de todos os arquivos suportados."""
    supported = {".txt", ".pdf", ".docx", ".doc"}
    return {
        p.name: p
        for p in DOCS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in supported
    }


# ── 4. INGESTÃO INCREMENTAL ───────────────────────────────────────────────────

def get_vector_store(embeddings) -> Chroma:
    """
    Estratégia incremental:
    1. Verifica quais arquivos já foram indexados (manifest)
    2. Identifica arquivos NOVOS (estão no diretório mas não no manifest)
    3. Processa apenas os novos e adiciona ao banco existente
    4. Se o banco não existe, cria do zero

    Vantagem: adicionar 1 arquivo novo não reprocessa os outros 99.
    """
    available_files = get_files_in_dir()
    already_indexed = load_manifest()

    new_files = {
        name: path
        for name, path in available_files.items()
        if name not in already_indexed
    }

    # Banco já existe e não há arquivos novos → só carrega
    db_exists = (PERSIST_DIR / "chroma.sqlite3").exists()
    if db_exists and not new_files:
        print(f"[store] Banco atualizado. Carregando {len(already_indexed)} arquivo(s) do disco...")
        vs = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        print(f"[store] {vs._collection.count()} chunk(s) no banco.")
        return vs

    # Há arquivos novos (ou banco não existe) → processa os novos
    if new_files:
        print(f"\n[ingest] {len(new_files)} arquivo(s) novo(s) encontrado(s):")
        for name in new_files:
            print(f"         + {name}")

        documents = load_documents(list(new_files.values()))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[ingest] {len(chunks)} chunk(s) gerado(s)")

        if db_exists:
            # Banco existe → adiciona os novos chunks sem apagar os antigos
            print("[ingest] Adicionando ao banco existente...")
            vs = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(PERSIST_DIR),
            )
            vs.add_documents(chunks)
        else:
            # Banco não existe → cria do zero
            PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            print("[ingest] Criando banco do zero...")
            vs = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(PERSIST_DIR),
            )

        # Atualiza o manifest com os arquivos recém-indexados
        updated = already_indexed | set(new_files.keys())
        save_manifest(updated)
        print(f"[ingest] Manifest atualizado: {len(updated)} arquivo(s) indexado(s).")
        return vs


# ── 5. RAG CHAIN COM CITAÇÃO ──────────────────────────────────────────────────

def build_rag_chain(vector_store: Chroma):
    """
    Novidade na Fase 4: a chain retorna DOIS valores:
      - answer: texto gerado pelo Claude (com citações inline)
      - sources: lista dos arquivos que foram consultados

    Para isso usamos RunnableLambda — permite criar um passo customizado
    que retorna um dict em vez de uma string simples.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY não encontrada no .env")

    retriever = vector_store.as_retriever(search_kwargs={"k": K_CHUNKS})

    llm = ChatAnthropic(model=CLAUDE_MODEL, temperature=0, max_tokens=1024)

    def format_context_and_sources(docs):
        """
        Retorna contexto formatado E lista de fontes em um dict.
        O contexto inclui o nome do arquivo em cada trecho
        para que o Claude saiba qual arquivo citar.
        """
        parts = []
        sources = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "desconhecido")
            parts.append(f"[Trecho {i} — {source}]\n{doc.page_content.strip()}")
            if source not in sources:
                sources.append(source)
        return {
            "context_text": "\n\n".join(parts),
            "sources": sources,
            "docs": docs,
        }

    # Chain principal: retrieval → formatação → Claude → resposta + fontes
    def full_chain(question: str) -> dict:
        # 1. Recupera chunks relevantes
        docs = retriever.invoke(question)

        # 2. Formata contexto e coleta fontes
        formatted = format_context_and_sources(docs)

        # 3. Monta o prompt e chama o Claude
        prompt_value = RAG_PROMPT.invoke({
            "context": formatted["context_text"],
            "question": question,
        })
        answer = llm.invoke(prompt_value)
        answer_text = StrOutputParser().invoke(answer)

        return {
            "answer": answer_text,
            "sources": formatted["sources"],
        }

    return full_chain


# ── 6. EXIBIÇÃO ───────────────────────────────────────────────────────────────

def print_result(query: str, result: dict):
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"PERGUNTA: {query}")
    print(bar)
    print(textwrap.fill(result["answer"].strip(), width=70))
    print(f"\nFontes consultadas: {', '.join(result['sources'])}")
    print()


# ── 7. PIPELINE PRINCIPAL ────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  FASE 4 — Multi-Documento com Citação de Fonte")
    print("=" * 70 + "\n")

    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        raise FileNotFoundError(f"Nenhum documento encontrado em '{DOCS_DIR}'")

    print("[embed] Carregando modelo de embeddings...")
    embeddings = build_embeddings_model()

    vector_store = get_vector_store(embeddings)

    print(f"\n[llm]   Conectando ao Claude ({CLAUDE_MODEL})...")
    rag_chain = build_rag_chain(vector_store)
    print("[llm]   Pronto.\n")

    # Perguntas que cruzam os dois documentos
    demo_queries = [
        "Quantos dias de férias tenho por ano?",           # politica_empresa.txt
        "Qual é o prazo de pagamento do contrato de TI?",  # contrato_servico.txt
        "O que acontece se eu violar a confidencialidade?",# ambos têm cláusulas
        "Qual é o SLA para chamados críticos?",            # contrato_servico.txt
        "Quando é pago o PLR e quanto custa o plano de saúde?",  # politica_empresa.txt
    ]

    print("--- Demonstração Multi-Documento ---")
    for query in demo_queries:
        result = rag_chain(query)
        print_result(query, result)

    # Modo interativo
    bar = "=" * 70
    print(f"{bar}")
    print("  Modo interativo — 'sair' para encerrar")
    print(f"{bar}\n")

    while True:
        query = input("Pergunta: ").strip()
        if not query or query.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break
        result = rag_chain(query)
        print_result(query, result)


def build_embeddings_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    main()
