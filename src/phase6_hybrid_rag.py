"""
Phase 6 — Busca Híbrida: BM25 + Semântica + RRF
=================================================
Combina retrieval lexical (BM25) com retrieval semântico (ChromaDB)
via Reciprocal Rank Fusion (RRF).

Por que híbrido?
  Semântico: ótimo para conceitos, paráfrases, linguagem natural
  BM25:      ótimo para termos exatos, siglas, nomes próprios, números
  RRF:       funde os dois rankings sem precisar tunar pesos manualmente

Pipeline:
  query
    ├── BM25 (lexical)     → top-2K índices
    └── Semântico (cosine) → top-2K docs
          │
          ▼
        RRF fusion → top-K docs (reordenados)
"""

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    """
    Retriever híbrido: BM25 lexical + semântico vetorial + RRF.

    Compatível com a interface LangChain (.invoke devolve list[Document]).
    Serve como drop-in para vector_store.as_retriever().
    """

    def __init__(self, vector_store, all_docs: list, k: int = 5, rrf_k: int = 60):
        self.vector_store = vector_store
        self.vectorstore = vector_store   # alias: compatibilidade com ingest_new_file
        self.all_docs = all_docs
        self.k = k
        self.rrf_k = rrf_k

        # Mapeia page_content → índice para correspondência dos resultados semânticos.
        # Colisão improvável em coleções normais (chunks são únicos).
        self._content_to_idx: dict = {
            doc.page_content: i for i, doc in enumerate(all_docs)
        }

        # Índice BM25: tokenização por whitespace (suficiente para português/inglês)
        tokenized = [doc.page_content.lower().split() for doc in all_docs]
        self.bm25 = BM25Okapi(tokenized)

    def invoke(self, query: str, filter: dict | None = None) -> list:
        """Executa busca híbrida e devolve top-k documentos fundidos via RRF."""
        top_n = self.k * 2  # candidatos extras antes do RRF

        # 1. BM25 — ranking lexical
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranking = sorted(
            range(len(self.all_docs)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:top_n]

        # 2. Semântico — ranking vetorial (cosine via ChromaDB)
        search_kwargs: dict = {"k": top_n}
        if filter:
            search_kwargs["filter"] = filter
        semantic_docs = self.vector_store.similarity_search(query, **search_kwargs)

        # Mapeia docs semânticos de volta para índices no all_docs
        semantic_ranking = []
        for doc in semantic_docs:
            idx = self._content_to_idx.get(doc.page_content)
            if idx is not None:
                semantic_ranking.append(idx)

        # 3. RRF — funde os dois rankings
        fused = self._rrf([bm25_ranking, semantic_ranking])[: self.k]
        return [self.all_docs[i] for i in fused]

    def _rrf(self, rankings: list) -> list:
        """
        Reciprocal Rank Fusion.
        score(d) = sum(1 / (k + rank(d)))  para cada ranking que contém d.
        rrf_k=60 é o padrão da literatura; controla quanto a posição importa.
        """
        scores: dict = {}
        for ranking in rankings:
            for rank, doc_idx in enumerate(ranking):
                scores[doc_idx] = scores.get(doc_idx, 0) + 1 / (self.rrf_k + rank + 1)
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def build_hybrid_retriever(vector_store, k: int = 5) -> "HybridRetriever":
    """
    Constrói um HybridRetriever carregando todos os docs do ChromaDB em memória.

    Trade-off: carrega O(N) textos na RAM para construir o índice BM25.
    Para coleções < 100k chunks isso é negligenciável (~100MB para 50k chunks).
    Para coleções maiores, serializar o índice BM25 com pickle + cache.
    """
    collection = vector_store._collection
    results = collection.get(include=["documents", "metadatas"])

    if not results["documents"]:
        raise RuntimeError(
            "Banco vetorial vazio. Execute phase4_multi_doc.py primeiro."
        )

    metadatas = results["metadatas"] or [{}] * len(results["documents"])
    all_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(results["documents"], metadatas)
    ]

    return HybridRetriever(vector_store, all_docs, k=k)


# ── DEMO STANDALONE ───────────────────────────────────────────────────────────

def main():
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    load_dotenv()

    PERSIST_DIR     = Path(__file__).parent.parent / "data" / "chroma_db_v4"
    COLLECTION_NAME = "multi_doc_collection"

    print("[embed] Carregando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(PERSIST_DIR),
    )

    print("[hybrid] Construindo índice BM25...")
    retriever = build_hybrid_retriever(vector_store, k=5)
    print(f"[hybrid] {len(retriever.all_docs)} docs indexados no BM25.")

    queries = [
        "plano de saúde para dependentes",
        "rescisão contrato TI prazo",
        "férias anuais remuneradas",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        docs = retriever.invoke(q)
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i}] {src}: {preview}...")


if __name__ == "__main__":
    main()
