"""
Lógica de ingestão reutilizável pela API e pelos scripts de linha de comando.

Extrai as funções de load_document, chunking e indexação da Fase 4
para que possam ser chamadas tanto pelos scripts (phase4_multi_doc.py)
quanto pelos endpoints FastAPI (documents.py) sem duplicar código.
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


def ingest_new_file(path: Path, embeddings=None) -> int:
    """
    Indexa um único arquivo novo no banco vetorial existente.
    Retorna o número de chunks adicionados.

    Se o banco não existir ainda, levanta FileNotFoundError.
    Use phase4_multi_doc.py para criar o banco do zero.

    embeddings: se None, usa o objeto armazenado no app_state da API.
    """
    # Importa aqui para evitar import circular com main.py
    from src.api.main import app_state

    if embeddings is None:
        # Pega o embeddings já carregado no startup da API
        rag = app_state.get("rag")
        if not rag:
            raise RuntimeError("API não inicializada. Rode o servidor primeiro.")
        # Acessa o embedding function via retriever → vector store
        embeddings = rag.retriever.vectorstore._embedding_function

    db_file = PERSIST_DIR / "chroma.sqlite3"
    if not db_file.exists():
        raise FileNotFoundError(
            f"Banco vetorial não encontrado. Execute phase4_multi_doc.py primeiro."
        )

    # Carrega e divide o documento
    documents = _load_file(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Adiciona ao banco existente (não apaga os outros documentos)
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    vs.add_documents(chunks)

    # Atualiza o manifest
    indexed = load_manifest()
    indexed.add(path.name)
    save_manifest(indexed)

    # Atualiza o contador no app_state
    from src.api.main import app_state
    app_state["chunk_count"] = vs._collection.count()

    print(f"[ingest] '{path.name}' → {len(chunks)} chunks adicionados.")
    return len(chunks)
