"""
Phase 7 — Endpoints de Upload e Listagem de Documentos

Novos endpoints:
  POST /documents/upload  → recebe arquivo(s), salva e indexa
  GET  /documents         → lista documentos indexados com stats
"""

import re
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

# Importa o estado global e a lógica de ingestão incremental
# (definidos em main.py e phase4_multi_doc.py respectivamente)
from src.api.ingest import ingest_new_file

router = APIRouter(prefix="/documents", tags=["Documentos"])

# ── CONFIGURAÇÃO ──────────────────────────────────────────────────────────────

DOCS_DIR       = Path(__file__).parent.parent.parent / "data" / "sample_docs"
ALLOWED_TYPES  = {".pdf", ".docx", ".doc", ".txt"}
MAX_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB


# ── MODELOS ───────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename: str
    size_kb: float


class UploadResponse(BaseModel):
    indexed: list[str]
    skipped: list[str]
    message: str


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """
    Sanitiza o nome do arquivo para evitar path traversal.
    Exemplo de ataque: "../../etc/passwd" → rejeitado.
    Mantém apenas letras, números, hífens, underscores e pontos.
    """
    # Remove qualquer separador de diretório
    name = Path(name).name
    # Permite apenas caracteres seguros
    name = re.sub(r"[^\w\.\-]", "_", name)
    if not name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Nome de arquivo inválido.")
    return name


def _validate_file(file: UploadFile, content: bytes):
    """Valida tipo e tamanho do arquivo."""
    # Valida extensão
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo não suportado: '{suffix}'. Use: {', '.join(ALLOWED_TYPES)}",
        )
    # Valida tamanho
    if len(content) > MAX_SIZE_BYTES:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande: {size_mb:.1f}MB. Máximo: 10MB.",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: Annotated[list[UploadFile], File(description="Um ou mais arquivos PDF, DOCX ou TXT")],
):
    """
    Faz upload de um ou mais documentos e os indexa no banco vetorial.

    - Tipos aceitos: PDF, DOCX, DOC, TXT
    - Tamanho máximo: 10MB por arquivo
    - Arquivos já indexados são ignorados (não duplica)
    - A indexação acontece de forma síncrona (aguarde a resposta)
    """
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    indexed = []
    skipped = []

    for file in files:
        # Lê o conteúdo antes de qualquer validação
        # (precisamos do conteúdo para checar o tamanho)
        content = await file.read()

        # Validações de segurança
        _validate_file(file, content)
        safe_name = _safe_filename(file.filename or "upload")
        dest_path = DOCS_DIR / safe_name

        # Pula se o arquivo já existe (evita reprocessamento desnecessário)
        if dest_path.exists():
            skipped.append(safe_name)
            continue

        # Salva em disco
        dest_path.write_bytes(content)

        # Indexa no banco vetorial
        # ingest_new_file() adiciona ao ChromaDB e atualiza o manifest
        try:
            chunks_added = ingest_new_file(dest_path)
            indexed.append(f"{safe_name} ({chunks_added} chunks)")
        except Exception as e:
            # Se falhar na indexação, remove o arquivo salvo
            dest_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao indexar '{safe_name}': {str(e)}",
            )

    if not indexed and not skipped:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    parts = []
    if indexed:
        parts.append(f"{len(indexed)} indexado(s)")
    if skipped:
        parts.append(f"{len(skipped)} já existia(m)")

    return UploadResponse(
        indexed=indexed,
        skipped=skipped,
        message=", ".join(parts) + ".",
    )


@router.get("", response_model=DocumentsResponse)
async def list_documents():
    """
    Lista todos os documentos disponíveis na pasta de documentos.
    Mostra nome e tamanho de cada arquivo.
    """
    if not DOCS_DIR.exists():
        return DocumentsResponse(documents=[], total=0)

    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_TYPES:
            size_kb = path.stat().st_size / 1024
            docs.append(DocumentInfo(filename=path.name, size_kb=round(size_kb, 1)))

    return DocumentsResponse(documents=docs, total=len(docs))
