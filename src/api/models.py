"""
Modelos Pydantic — contratos de entrada e saída da API.

Pydantic funciona assim: você define classes com tipos Python normais.
FastAPI usa essas classes para:
  1. Validar o JSON que chega (entrada inválida → erro 422 automático)
  2. Serializar o objeto de saída para JSON
  3. Gerar a documentação /docs automaticamente
"""

from pydantic import BaseModel, Field


# ── REQUEST (o que a API recebe) ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Corpo do POST /chat"""

    question: str = Field(
        ...,                          # ... = obrigatório (sem valor padrão)
        min_length=1,
        max_length=1000,
        description="Pergunta em linguagem natural sobre os documentos.",
        examples=["Quantos dias de férias tenho por ano?"],
    )


# ── RESPONSE (o que a API devolve) ───────────────────────────────────────────

class ChatResponse(BaseModel):
    """Corpo da resposta do POST /chat"""

    answer: str = Field(description="Resposta gerada pelo Llama 3.1 8B via Groq.")
    sources: list[str] = Field(description="Arquivos consultados para gerar a resposta.")
    question: str = Field(description="Pergunta reformulada usada no retrieval.")


class HistoryMessage(BaseModel):
    """Uma mensagem no histórico da conversa."""

    role: str = Field(description="'human' ou 'ai'")
    content: str


class HistoryResponse(BaseModel):
    """Corpo da resposta do GET /chat/history"""

    messages: list[HistoryMessage]
    total: int = Field(description="Total de mensagens no histórico.")


class StatusResponse(BaseModel):
    """Corpo da resposta do GET /health"""

    status: str
    chunks_indexed: int = Field(description="Número de chunks no banco vetorial.")
    model: str = Field(description="Modelo LLM em uso.")

