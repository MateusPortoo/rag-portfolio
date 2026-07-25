"""
conftest.py — fixtures compartilhadas entre todos os testes.

O pytest carrega este arquivo automaticamente antes de rodar qualquer teste.
Fixtures são funções que preparam o ambiente de teste e podem ser
reutilizadas em múltiplos arquivos de teste.
"""

import pytest
from pathlib import Path


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """
    Cria um arquivo .txt temporário para testes.
    tmp_path é uma fixture built-in do pytest que fornece
    um diretório temporário único por teste (apagado depois).
    """
    content = (
        "SEÇÃO 1 — FÉRIAS\n\n"
        "Todo colaborador tem direito a 30 dias corridos de férias por ano.\n"
        "As férias podem ser divididas em até 3 períodos.\n\n"
        "SEÇÃO 2 — BENEFÍCIOS\n\n"
        "Vale-refeição: R$ 45,00 por dia útil trabalhado.\n"
        "Plano de saúde: Amil 400 Nacional com coparticipação de 20%.\n\n"
        "SEÇÃO 3 — HOME OFFICE\n\n"
        "Modelo híbrido: mínimo 3 dias presenciais por semana.\n"
        "Subsídio home office: R$ 150,00 por mês.\n"
    )
    file = tmp_path / "politica_rh.txt"
    file.write_text(content, encoding="utf-8")
    return file


@pytest.fixture
def oversized_content() -> bytes:
    """Conteúdo que excede o limite de 10MB."""
    return b"x" * (11 * 1024 * 1024)  # 11 MB


@pytest.fixture
def sample_docs_dir(tmp_path: Path, sample_txt: Path) -> Path:
    """
    Diretório temporário com um documento de amostra.
    Útil para testes de ingestão multi-doc.
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    # Copia o arquivo de amostra para o diretório
    import shutil
    shutil.copy(sample_txt, docs_dir / sample_txt.name)
    return docs_dir
