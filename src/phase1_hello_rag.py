"""
Phase 1 — Hello World do RAG
=============================
Pipeline completo sem API: load → chunk → embed → store → query → print.
Nenhuma chamada de rede. Tudo roda localmente.
"""

import os
import textwrap
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── 1. CONFIGURAÇÃO ──────────────────────────────────────────────────────────

# Documento de amostra incluído no repositório
SAMPLE_DOC = Path(__file__).parent.parent / "data" / "sample_docs" / "politica_empresa.txt"

# Parâmetros de chunking acordados na entrevista de design
CHUNK_SIZE = 1000      # caracteres por chunk
CHUNK_OVERLAP = 200    # sobreposição entre chunks contíguos

# Modelo de embeddings local (baixa ~90 MB na primeira execução)
# Produz vetores de 384 dimensões; sem GPU necessária
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Perguntas para demonstrar o retrieval
DEMO_QUERIES = [
    "Quantos dias de férias o colaborador tem por ano?",
    "Como funciona o vale-refeição?",
    "Posso trabalhar 100% em home office?",
    "O que acontece se eu violar a confidencialidade?",
]


# ── 2. CARREGAR DOCUMENTO ────────────────────────────────────────────────────

def load_document(path: Path):
    """
    Carrega um arquivo .txt ou .pdf e retorna lista de Document do LangChain.
    Cada Document tem .page_content (texto) e .metadata (source, page, etc.).
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        # TextLoader lê o arquivo inteiro como um único Document
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".pdf":
        # PyPDFLoader retorna um Document por página
        loader = PyPDFLoader(str(path))
    else:
        raise ValueError(f"Formato não suportado: {suffix}")

    documents = loader.load()
    print(f"[load] {len(documents)} documento(s) carregado(s) de '{path.name}'")
    return documents


# ── 3. CHUNKING ──────────────────────────────────────────────────────────────

def split_documents(documents):
    """
    Divide os documentos em chunks menores.

    RecursiveCharacterTextSplitter tenta cortar em "\n\n", depois "\n",
    depois " ", depois caractere — priorizando cortes naturais do texto.
    Isso preserva parágrafos inteiros sempre que possível.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separadores em ordem de preferência (do mais natural ao mais forçado)
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[split] {len(chunks)} chunk(s) gerado(s) "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── 4. EMBEDDINGS ────────────────────────────────────────────────────────────

def build_embeddings():
    """
    Carrega o modelo de embeddings do HuggingFace Hub.
    Na primeira execução baixa ~90 MB e salva em cache (~/.cache/huggingface/).
    Execuções seguintes são instantâneas.
    """
    print(f"[embed] Carregando modelo '{EMBEDDING_MODEL}' (pode demorar na 1ª vez)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # Garante reprodutibilidade — sem randomness nos vetores
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("[embed] Modelo pronto.")
    return embeddings


# ── 5. VECTOR STORE ──────────────────────────────────────────────────────────

def build_vector_store(chunks, embeddings):
    """
    Cria um ChromaDB em memória, embeda todos os chunks e os indexa.

    Chroma.from_documents() faz as três operações em uma chamada:
      1. Chama embeddings.embed_documents() em cada chunk
      2. Armazena os vetores + texto + metadata
      3. Cria o índice ANN para busca rápida por similaridade

    Fase 1 usa persist_directory=None (in-memory) — os dados somem ao sair.
    A partir da Fase 3 usaremos persist_directory para salvar no disco.
    """
    print(f"[store] Embedando e indexando {len(chunks)} chunk(s)...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        # collection_name identifica esta coleção dentro do ChromaDB
        collection_name="phase1_demo",
    )
    print("[store] Vector store pronta.")
    return vector_store


# ── 6. RETRIEVAL ─────────────────────────────────────────────────────────────

def retrieve(vector_store, query: str, k: int = 3):
    """
    Converte a query em vetor e busca os k chunks mais similares.

    similarity_search_with_score retorna lista de (Document, score).
    Score = distância coseno: quanto MENOR, mais similar (escala 0→2).
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    return results


# ── 7. EXIBIÇÃO ───────────────────────────────────────────────────────────────

def print_results(query: str, results):
    """Formata e imprime os chunks recuperados de forma legível."""
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"QUERY: {query}")
    print(bar)

    for rank, (doc, score) in enumerate(results, start=1):
        # Score de distância coseno (0 = idêntico, 2 = oposto)
        # Convertemos para similaridade percentual aproximada
        similarity_pct = max(0.0, (1 - score / 2)) * 100
        source = doc.metadata.get("source", "desconhecido")

        print(f"\n[Resultado #{rank}]  Similaridade: {similarity_pct:.1f}%")
        print(f"Fonte: {Path(source).name}")
        print()
        # Quebra linhas longas para facilitar leitura no terminal
        wrapped = textwrap.fill(doc.page_content.strip(), width=70)
        print(wrapped)

    print(f"\n{bar}\n")


# ── 8. PIPELINE PRINCIPAL ────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  FASE 1 — Hello World do RAG")
    print("=" * 70 + "\n")

    # Verifica se o arquivo de amostra existe
    if not SAMPLE_DOC.exists():
        raise FileNotFoundError(
            f"Arquivo de amostra não encontrado: {SAMPLE_DOC}\n"
            "Execute a partir da raiz do projeto: python src/phase1_hello_rag.py"
        )

    # Pipeline: load → split → embed → store
    documents = load_document(SAMPLE_DOC)
    chunks = split_documents(documents)
    embeddings = build_embeddings()
    vector_store = build_vector_store(chunks, embeddings)

    print("\n--- Demonstração de Retrieval ---")

    # Executa cada query de demo e imprime os resultados
    for query in DEMO_QUERIES:
        results = retrieve(vector_store, query, k=2)
        print_results(query, results)

    print("Fase 1 concluída com sucesso!\n")
    print("Próximo passo → Fase 2: conectar Claude para gerar respostas em cima dos chunks.")


if __name__ == "__main__":
    main()
