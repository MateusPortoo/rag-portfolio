"""
Testes de integração — endpoints da FastAPI.

Usamos TestClient do FastAPI (não sobe um servidor real).
Mockamos o ConversationalRAG para não precisar de GPU, embeddings
nem chave da Anthropic durante os testes — os testes rodam offline.

Por que mockar o RAG?
  - Carregar o modelo de embeddings leva ~3s e ~90MB de RAM
  - Chamar o Claude custaria dinheiro real a cada `pytest`
  - Os testes de API verificam CONTRATOS (status codes, campos do JSON)
    não qualidade de resposta — o mock é suficiente para isso
"""

import io
from contextlib import asynccontextmanager

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ── FIXTURE: app com RAG mockado ─────────────────────────────────────────────

@pytest.fixture
def client():
    """
    Cria um TestClient com o app_state preenchido com um RAG falso.
    O mock imita a interface de ConversationalRAG sem fazer chamadas reais.

    Por que mockar o lifespan?
    O lifespan real carrega embeddings (~90MB) e exige ChromaDB em disco.
    Em CI não há banco nem chave real — o lifespan travaria sem o mock.
    """
    from src.api.main import app, app_state
    from langchain_core.messages import HumanMessage, AIMessage

    mock_rag = MagicMock()
    mock_rag.ask.return_value = {
        "answer": "Você tem direito a 30 dias de férias.",
        "sources": ["politica_empresa.txt"],
    }
    mock_rag.history.messages = [
        HumanMessage(content="Quantos dias de férias?"),
        AIMessage(content="30 dias corridos por ano."),
    ]
    mock_rag.clear_history = MagicMock()

    app_state["rag"] = mock_rag
    app_state["chunk_count"] = 42

    @asynccontextmanager
    async def _mock_lifespan(app):
        yield

    with patch("src.api.main.lifespan", _mock_lifespan):
        with TestClient(app) as c:
            yield c

    app_state.clear()


# ── GET /health ────────────────────────────────────────────────────────────────

class TestHealth:

    def test_retorna_200_quando_rag_inicializado(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_retorna_campos_esperados(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "chunks_indexed" in data
        assert "model" in data

    def test_status_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_chunks_indexados_correto(self, client):
        data = client.get("/health").json()
        assert data["chunks_indexed"] == 42

    def test_retorna_503_sem_rag(self):
        """Sem RAG no app_state, deve retornar 503 Service Unavailable."""
        from src.api.main import app, app_state
        app_state.clear()
        with TestClient(app) as c:
            response = c.get("/health")
        assert response.status_code == 503


# ── POST /chat ─────────────────────────────────────────────────────────────────

class TestChat:

    def test_pergunta_valida_retorna_200(self, client):
        response = client.post("/chat", json={"question": "Quantos dias de férias?"})
        assert response.status_code == 200

    def test_resposta_tem_campos_obrigatorios(self, client):
        data = client.post("/chat", json={"question": "Quantos dias?"}).json()
        assert "answer" in data
        assert "sources" in data
        assert "question" in data

    def test_sources_e_lista(self, client):
        data = client.post("/chat", json={"question": "Plano de saúde?"}).json()
        assert isinstance(data["sources"], list)

    def test_pergunta_vazia_retorna_422(self, client):
        """422 = Unprocessable Entity — validação do Pydantic rejeitou o input."""
        response = client.post("/chat", json={"question": ""})
        assert response.status_code == 422

    def test_pergunta_muito_longa_retorna_422(self, client):
        """Pergunta com mais de 1000 caracteres deve ser rejeitada."""
        response = client.post("/chat", json={"question": "x" * 1001})
        assert response.status_code == 422

    def test_body_sem_question_retorna_422(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_rag_ask_e_chamado_uma_vez(self, client):
        """Verifica que o pipeline RAG foi de fato invocado."""
        from src.api.main import app_state
        client.post("/chat", json={"question": "Quantos dias?"})
        app_state["rag"].ask.assert_called_once_with("Quantos dias?")


# ── GET /chat/history ──────────────────────────────────────────────────────────

class TestHistory:

    def test_retorna_200(self, client):
        response = client.get("/chat/history")
        assert response.status_code == 200

    def test_retorna_lista_de_mensagens(self, client):
        data = client.get("/chat/history").json()
        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_mensagens_tem_role_e_content(self, client):
        messages = client.get("/chat/history").json()["messages"]
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("human", "ai")

    def test_total_correto(self, client):
        data = client.get("/chat/history").json()
        assert data["total"] == len(data["messages"])


# ── DELETE /chat/history ───────────────────────────────────────────────────────

class TestClearHistory:

    def test_retorna_200(self, client):
        response = client.delete("/chat/history")
        assert response.status_code == 200

    def test_history_clear_e_chamado(self, client):
        from src.api.main import app_state
        client.delete("/chat/history")
        app_state["rag"].clear_history.assert_called_once()


# ── GET /documents ─────────────────────────────────────────────────────────────

class TestDocuments:

    def test_retorna_200(self, client):
        response = client.get("/documents")
        assert response.status_code == 200

    def test_retorna_lista(self, client):
        data = client.get("/documents").json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_total_consistente(self, client):
        data = client.get("/documents").json()
        assert data["total"] == len(data["documents"])


# ── POST /documents/upload ─────────────────────────────────────────────────────

class TestUpload:

    def _make_file(self, content: bytes, filename: str, content_type: str = "text/plain"):
        return ("files", (filename, io.BytesIO(content), content_type))

    def test_tipo_invalido_retorna_400(self, client):
        with patch("src.api.documents.ingest_new_file", return_value=5):
            response = client.post(
                "/documents/upload",
                files=[self._make_file(b"conteudo", "virus.exe", "application/octet-stream")],
            )
        assert response.status_code == 400

    def test_arquivo_vazio_retorna_400(self, client):
        with patch("src.api.documents.ingest_new_file", return_value=0):
            response = client.post(
                "/documents/upload",
                files=[self._make_file(b"", "vazio.txt", "text/plain")],
            )
        assert response.status_code == 400

    def test_nenhum_arquivo_retorna_400(self, client):
        response = client.post("/documents/upload", files=[])
        assert response.status_code in (400, 422)
