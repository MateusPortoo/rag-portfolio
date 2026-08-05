"""
Phase 7 — HyDE: Hypothetical Document Embeddings
==================================================
Em vez de embeddar a query diretamente, o LLM gera um documento hipotético
que responderia à pergunta. Esse documento hipotético é embeddado e usado
para buscar os chunks reais no ChromaDB.

Por que HyDE funciona melhor que busca direta?
  Query:               curta, interrogativa, sem contexto
  Documento hipotético: longo, assertivo, no estilo de documentos reais
  → O embedding do documento hipotético fica mais próximo dos chunks reais
    do que o embedding da query original.

Referência: Gao et al. 2022 — "Precise Zero-Shot Dense Retrieval without
            Relevance Labels" (https://arxiv.org/abs/2212.10496)

Pipeline:
  query
    │
    ▼
  [LLM] gera resposta hipotética (1 parágrafo)
    │
    ▼
  [Embeddings] embeda o documento hipotético
    │
    ▼
  ChromaDB.similarity_search_by_vector()  →  top-K chunks reais
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

HYDE_PROMPT = ChatPromptTemplate.from_template(
    "Escreva um parágrafo que responderia à pergunta abaixo, como se fosse "
    "um trecho de um documento técnico ou empresarial. "
    "Seja factual e objetivo. Não adicione disclaimers.\n\n"
    "Pergunta: {question}\n\n"
    "Resposta hipotética:"
)


class HyDERetriever:
    """
    Retriever que aplica Hypothetical Document Embeddings antes da busca.

    Compatível com a interface LangChain (.invoke → list[Document]).
    Serve como drop-in para vector_store.as_retriever().
    """

    def __init__(self, vector_store, llm, embeddings, k: int = 5):
        self.vector_store = vector_store
        self.vectorstore = vector_store  # alias: compatibilidade com ingest_new_file
        self.embeddings = embeddings
        self.k = k
        self._hyde_chain = HYDE_PROMPT | llm | StrOutputParser()

    def invoke(self, query: str) -> list:
        """
        1. Gera documento hipotético via LLM
        2. Embeda o documento hipotético
        3. Busca por vetor (não por string) no ChromaDB
        """
        hypothetical = self._hyde_chain.invoke({"question": query})
        embedding = self.embeddings.embed_query(hypothetical)
        return self.vector_store.similarity_search_by_vector(embedding, k=self.k)


def build_hyde_retriever(vector_store, llm, embeddings, k: int = 5) -> HyDERetriever:
    """Instancia o HyDERetriever com os objetos já carregados pela API."""
    return HyDERetriever(vector_store, llm, embeddings, k=k)


# ── DEMO STANDALONE ───────────────────────────────────────────────────────────

def main():
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_groq import ChatGroq

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

    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)

    retriever = build_hyde_retriever(vector_store, llm, embeddings, k=5)

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
