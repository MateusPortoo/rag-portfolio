"""
Testes do pipeline de reranking com golden set.

Estrutura:
  - Testes unitarios (sem ChromaDB, sem LLM) -- rodam no CI
  - Testes de integracao (requerem ChromaDB populado) -- marcados com skip
    Para rodar localmente: pytest tests/test_reranking.py -m integration -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.phase8_reranker import RerankingRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def _make_doc(content: str, source: str = "doc.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


def _make_retriever(docs: list) -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = docs
    mock.vectorstore = MagicMock()
    return mock


def _make_reranker(docs: list, scores: list, k_final: int = 3,
                   threshold: float = 0.0) -> RerankingRetriever:
    base = _make_retriever(docs)
    reranker = RerankingRetriever(base, k_final=k_final, threshold=threshold)
    reranker.cross_encoder = MagicMock()
    reranker.cross_encoder.predict.return_value = scores
    return reranker


class TestRerankingThreshold:

    def test_retorna_todos_acima_do_threshold(self):
        docs = [_make_doc(f"doc {i}") for i in range(3)]
        reranker = _make_reranker(docs, scores=[0.9, 0.8, 0.7], threshold=0.5)
        result = reranker.invoke("query")
        assert len(result) == 3

    def test_filtra_docs_abaixo_do_threshold(self):
        docs = [_make_doc(f"doc {i}") for i in range(3)]
        reranker = _make_reranker(docs, scores=[0.9, 0.3, 0.8], threshold=0.65)
        result = reranker.invoke("query")
        assert len(result) == 2

    def test_retorna_vazio_quando_todos_abaixo(self):
        docs = [_make_doc(f"doc {i}") for i in range(3)]
        reranker = _make_reranker(docs, scores=[0.1, 0.2, 0.3], threshold=0.65)
        result = reranker.invoke("query")
        assert result == []

    def test_k_final_limita_saida(self):
        docs = [_make_doc(f"doc {i}") for i in range(5)]
        reranker = _make_reranker(docs, scores=[0.9, 0.8, 0.7, 0.6, 0.5],
                                  k_final=3, threshold=0.0)
        result = reranker.invoke("query")
        assert len(result) == 3

    def test_reordena_por_score_decrescente(self):
        docs = [_make_doc(f"doc {i}") for i in range(3)]
        reranker = _make_reranker(docs, scores=[0.5, 0.3, 0.9], threshold=0.0)
        result = reranker.invoke("query")
        assert result[0].page_content == "doc 2"

    def test_threshold_exato_e_incluido(self):
        docs = [_make_doc("limite")]
        reranker = _make_reranker(docs, scores=[0.65], threshold=0.65)
        result = reranker.invoke("query")
        assert len(result) == 1

    def test_retorna_vazio_sem_candidatos(self):
        base = _make_retriever([])
        reranker = RerankingRetriever(base, k_final=5, threshold=0.0)
        reranker.cross_encoder = MagicMock()
        result = reranker.invoke("query")
        assert result == []

    def test_threshold_zero_nao_filtra_scores_negativos(self):
        docs = [_make_doc(f"doc {i}") for i in range(4)]
        reranker = _make_reranker(docs, scores=[-5.0, -3.0, -1.0, 0.0],
                                  threshold=0.0)
        result = reranker.invoke("query")
        assert len(result) == 4

    def test_vectorstore_alias_disponivel(self):
        base = _make_retriever([])
        reranker = RerankingRetriever(base, k_final=5)
        assert reranker.vectorstore is not None


class TestGoldenSetStructure:

    def test_arquivo_existe(self):
        assert GOLDEN_SET_PATH.exists()

    def test_e_lista_nao_vazia(self):
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0

    def test_campos_obrigatorios(self):
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        for item in data:
            assert "id" in item
            assert "question" in item
            assert "expected_sources" in item
            assert "expected_keywords" in item
            assert "category" in item

    def test_sources_sao_arquivos_conhecidos(self):
        known = {"politica_empresa.txt", "contrato_servico.txt"}
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        for item in data:
            for src in item["expected_sources"]:
                assert src in known

    def test_ids_unicos(self):
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        ids = [item["id"] for item in data]
        assert len(ids) == len(set(ids))

    def test_keywords_nao_vazias(self):
        data = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
        for item in data:
            assert len(item["expected_keywords"]) > 0


@pytest.mark.skip(reason="requer ChromaDB populado -- rode localmente: pytest -m integration")
class TestRerankingIntegration:

    @pytest.fixture(scope="class")
    def retriever(self):
        from pathlib import Path
        from dotenv import load_dotenv
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_groq import ChatGroq
        from src.phase7_hyde import build_hyde_retriever
        from src.phase8_reranker import build_reranking_retriever

        load_dotenv()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_store = Chroma(
            collection_name="multi_doc_collection",
            embedding_function=embeddings,
            persist_directory=str(
                Path(__file__).parent.parent / "data" / "chroma_db_v4"
            ),
        )
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=256)
        hyde = build_hyde_retriever(vector_store, llm, embeddings, k=15)
        return build_reranking_retriever(hyde, k_final=5, k_candidates=15, threshold=0.65)

    @pytest.fixture(scope="class")
    def golden(self):
        return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))

    def test_source_precision_at_1(self, retriever, golden):
        hits = sum(
            1 for item in golden
            if (docs := retriever.invoke(item["question"]))
            and docs[0].metadata.get("source") in item["expected_sources"]
        )
        precision = hits / len(golden)
        print(f"\nSource P@1: {precision:.2%} ({hits}/{len(golden)})")
        assert precision >= 0.70

    def test_keyword_recall_at_3(self, retriever, golden):
        def hit(item):
            docs = retriever.invoke(item["question"])[:3]
            text = " ".join(d.page_content for d in docs).lower()
            return any(kw.lower() in text for kw in item["expected_keywords"])

        hits = sum(1 for item in golden if hit(item))
        recall = hits / len(golden)
        print(f"\nKeyword R@3: {recall:.2%} ({hits}/{len(golden)})")
        assert recall >= 0.80

    def test_mrr(self, retriever, golden):
        def mrr(item):
            docs = retriever.invoke(item["question"])
            for rank, doc in enumerate(docs, 1):
                if doc.metadata.get("source") in item["expected_sources"]:
                    return 1.0 / rank
            return 0.0

        score = sum(mrr(item) for item in golden) / len(golden)
        print(f"\nMRR: {score:.4f}")
        assert score >= 0.65

    def test_k_final_respeitado(self, retriever, golden):
        for item in golden:
            assert len(retriever.invoke(item["question"])) <= 5
