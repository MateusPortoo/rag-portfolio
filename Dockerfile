# ── Imagem base ───────────────────────────────────────────────────────────────
# python:3.11-slim: menor que a full, mas inclui compiladores C necessários
# pelo sentence-transformers. alpine seria menor ainda mas quebra no build.
FROM python:3.11-slim

# ── Dependências do sistema ────────────────────────────────────────────────────
# libgomp1: necessária pelo sentence-transformers (OpenMP para paralelismo)
# build-essential: compiladores C para alguns pacotes Python com extensões nativas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Diretório de trabalho ──────────────────────────────────────────────────────
WORKDIR /app

# ── Dependências Python ────────────────────────────────────────────────────────
# Copiamos requirements.txt ANTES do código fonte.
# Motivo: Docker usa cache por camada. Se o código mudar mas o requirements.txt
# não mudar, o Docker reutiliza a camada de instalação — build muito mais rápido.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código fonte ───────────────────────────────────────────────────────────────
COPY . .

# ── Usuário não-root (segurança) ───────────────────────────────────────────────
# Rodar como root dentro do container é um risco de segurança.
# Criamos um usuário dedicado para a aplicação.
RUN useradd --create-home appuser
RUN chown -R appuser:appuser /app
USER appuser

# ── Expõe as portas ────────────────────────────────────────────────────────────
# EXPOSE é documentação — não abre porta automaticamente. O docker-compose faz isso.
EXPOSE 8000
EXPOSE 8501
