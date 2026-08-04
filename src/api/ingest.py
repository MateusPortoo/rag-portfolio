"""
Lógica de ingestão reutilizável pela API e pelos scripts de linha de comando.

Fase 6+:
  - HNSW configurado explicitamente (cosine, ef=200, M=16, search_ef=100)
  - Metadata enriquecida por chunk: source, doc_type, chunk_index, total_chunks
  - build_retriever() com suporte a filtro de metadata
"""

import json
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

PERSIST_DIR     = Path(__file__).parent.parent.parent / "data" / "chroma_db_v4"
MANIFEST_FILE   = PERSIST_DIR / "manifest.json"
COLLECTION_NAME = "multi_doc_collection"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
N_RESULTS       = 4

# HNSW: parâmetros do índice vetorial no ChromaDB
# hnsw:space          → cosine é ideal para embeddings L2-normalizados
# hnsw:construction_ef → ef durante indexação; maior = melhor recall, build mais lento
# hnsw:M               → conexões por nó no grafo; 16 é o padrão; 32+ melhora recall em datasets grandes
# hnsw:search_ef       → ef durante busca; maior = melhor recall, busca mais lenta
# Estes valores só têm efeito quando a coleção é criada pela primeira vez.
HNSW_CONFIG = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:M": 16,
    "hnsw:search_ef": 100,
}


def load_manifest() -> set:
    if MANIFEST_FILE.exists():
        data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        return set(data.get("indexed_files", []))
    return set()


def save_manifest(indexed_files: set):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps({"indexed_files": sorted(indexed_files)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_file(path: Path) -> list:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in (".docx", ".doc"):
        loader = Docx2txtLoader(str(path))
    else:
        raise ValueError(f"Formato não suportado: {suffix}")

    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path.name
    return docs


def _enrich_metadata(chunks: list, path: Path) -> list:
    """
    Adiciona campos de metadata a cada chunk após o split:
      - source:        nome do arquivo (garantido aqui mesmo que o loader omita)
      - doc_type:      extensão sem ponto — "pdf", "txt", "docx"
      - chunk_index:   posição 0-based dentro do documento
      - total_chunks:  total de chunks deste documento

    Por que enriquecer?
    Permite filtros no retrieval ANTES do embedding search — O(1) no banco,
    sem custo adicional de LLM ou tokens. Exemplos práticos:
      {"source": "contrato.txt"}          → isola um arquivo
      {"doc_type": "pdf"}                 → só PDFs
      {"chunk_index": {"$lte": 3}}        → só os primeiros 4 chunks (sumário)
    """
    total = len(chunks)
    doc_type = path.suffix.lower().lstrip(".")
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = path.name
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = total
    return chunks


def build_retriever(vector_store: Chroma, filter: dict | None = None):
    """
    Cria um retriever com n_results=4 e filtro opcional de metadata.

    Exemplos de filtro (sintaxe ChromaDB):
      {"source": "politica_empresa.txt"}      → arquivo específico
      {"doc_type": "pdf"}                     → só PDFs
      {"chunk_index": {"$lte": 5}}            → início do documento
      {"$and": [{"doc_type": "txt"},
                {"chunk_index": {"$gte": 2}}]} → TXTs a partir do chunk 2
    """
    search_kwargs: dict = {"k": N_RESULTS}
    if filter:
        search_kwargs["filter"] = filter
    return vector_store.as_retriever(search_kwargs=search_kwargs)


def ingest_new_file(path: Path, embeddings=None) -> int:
    """
    Indexa um único arquivo novo no banco vetorial existente.
    Retorna o número de chunks adicionados.
    """
    from src.api.main import app_state

    if embeddings is None:
        rag = app_state.get("rag")
        if not rag:
            raise RuntimeError("API não inicializada. Rode o servidor primeiro.")
        embeddings = rag.retriever.vectorstore._embedding_function

    db_file = PERSIST_DIR / "chroma.sqlite3"
    if not db_file.exists():
        raise FileNotFoundError(
            "Banco vetorial não encontrado. Execute phase4_multi_doc.py primeiro."
        )

    documents = _load_file(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    chunks = _enrich_metadata(chunks, path)

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_metadata=HNSW_CONFIG,
    )
    vs.add_documents(chunks)

    indexed = load_manifest()
    indexed.add(path.name)
    save_manifest(indexed)

    from src.api.main import app_state
    app_state["chunk_count"] = vs._collection.count()

    print(f"[ingest] '{path.name}' → {len(chunks)} chunks adicionados.")
    return len(chunks)
