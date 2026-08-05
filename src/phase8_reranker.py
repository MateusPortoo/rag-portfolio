"""
Phase 8 — Reranking com Cross-Encoder
======================================
Apos o retrieval inicial (HyDE, hibrido, ou semantico), um cross-encoder
avalia cada par (query, chunk) e reordena os candidatos por relevancia real.

Por que reranking?
  Retrievers bi-encoder (embeddings) codificam query e docs separadamente.
  Isso e rapido mas perde a interacao entre os termos da query e do doc.
  Cross-encoders analisam o par (query, doc) juntos -- mais lento, mas muito
  mais preciso para estimar relevancia.

Padrao de uso:
  retriever rapido (HyDE/hibrido) -> top-K candidatos
  cross-encoder                   -> reordena os K candidatos
  LLM                             -> responde com os top-N finais

Modelo usado: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 6 camadas MiniLM, ~22M parametros
  - Rapido o suficiente para rodar em CPU em <1s para 15 candidatos
  - Treinado em MS MARCO (relevancia de passagens para perguntas)
"""

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    """
    Wrapper que aplica reranking cross-encoder sobre qualquer retriever base.

    Fluxo:
      base_retriever.invoke(query)  ->  K candidatos (k_candidates)
      cross_encoder.predict(pairs)  ->  scores por par (query, chunk)
      sort by score + threshold     ->  top k_final docs
    """

    def __init__(self, base_retriever, k_final: int = 5, k_candidates: int = 15,
                 threshold: float = 0.0, model: str = CROSS_ENCODER_MODEL):
        self.base_retriever = base_retriever
        self.vectorstore = getattr(base_retriever, "vectorstore", None)
        self.k_final = k_final
        self.k_candidates = k_candidates
        self.threshold = threshold
        self.cross_encoder = CrossEncoder(model)

    def invoke(self, query: str) -> list:
        """
        1. Recupera k_candidates docs do retriever base
        2. Pontua cada (query, doc) com o cross-encoder
        3. Filtra por threshold e retorna os k_final melhores
        """
        candidates = self.base_retriever.invoke(query)

        if not candidates:
            return []

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [
            doc for score, doc in ranked[:self.k_final]
            if score >= self.threshold
        ]


def build_reranking_retriever(base_retriever, k_final: int = 5,
                              k_candidates: int = 15,
                              threshold: float = 0.0) -> RerankingRetriever:
    """
    Envolve qualquer retriever com reranking cross-encoder.

    k_candidates deve ser maior que k_final para que o reranker
    tenha candidatos suficientes para reordenar.
    threshold filtra chunks com score abaixo do minimo -- util para
    evitar que o LLM receba contexto irrelevante.
    """
    return RerankingRetriever(base_retriever, k_final=k_final,
                              k_candidates=k_candidates, threshold=threshold)


# -- DEMO STANDALONE -----------------------------------------------------------

def main():
    from pathlib import Path
    from dotenv import load_dotenv
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_groq import ChatGroq
    from src.phase7_hyde import build_hyde_retriever

    load_dotenv()

    PERSIST_DIR = Path(__file__).parent.parent / "data" / "chroma_db_v4"
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

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    print("[hyde]  Construindo HyDE retriever (k=15 candidatos)...")
    hyde_retriever = build_hyde_retriever(vector_store, llm, embeddings, k=15)

    print("[rank]  Carregando cross-encoder...")
    retriever = build_reranking_retriever(hyde_retriever, k_final=5,
                                         k_candidates=15, threshold=0.65)

    queries = [
        "plano de saude para dependentes",
        "rescisao contrato TI prazo",
        "ferias anuais remuneradas",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        docs = retriever.invoke(q)
        if not docs:
            print("  (nenhum doc acima do threshold)")
            continue
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"  [{i}] {src}: {preview}...")


if __name__ == "__main__":
    main()
