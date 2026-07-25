"""
Testes unitários — funções de segurança do endpoint de upload.

Estas funções são puras (sem side-effects), então os testes são simples:
entrada → saída esperada, sem mocks necessários.
"""

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from src.api.documents import _safe_filename, _validate_file


# ── _safe_filename ────────────────────────────────────────────────────────────

class TestSafeFilename:

    def test_nome_normal(self):
        assert _safe_filename("relatorio.pdf") == "relatorio.pdf"

    def test_nome_com_espacos_vira_underscore(self):
        result = _safe_filename("meu relatorio.pdf")
        assert " " not in result

    def test_path_traversal_bloqueado(self):
        """../etc/passwd é o ataque mais clássico de path traversal."""
        result = _safe_filename("../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_path_traversal_windows(self):
        result = _safe_filename("..\\..\\windows\\system32\\config")
        assert ".." not in result

    def test_nome_vazio_levanta_erro(self):
        with pytest.raises(HTTPException) as exc:
            _safe_filename("")
        assert exc.value.status_code == 400

    def test_nome_comecando_com_ponto_levanta_erro(self):
        with pytest.raises(HTTPException) as exc:
            _safe_filename(".hidden_file")
        assert exc.value.status_code == 400

    def test_caracteres_especiais_sanitizados(self):
        result = _safe_filename("arquivo<script>.txt")
        assert "<" not in result
        assert ">" not in result


# ── _validate_file ────────────────────────────────────────────────────────────

class TestValidateFile:

    def _make_upload_file(self, filename: str):
        """Cria um mock de UploadFile com o filename especificado."""
        mock = MagicMock()
        mock.filename = filename
        return mock

    def test_pdf_valido(self):
        """Não deve levantar exceção para PDF com tamanho normal."""
        file = self._make_upload_file("documento.pdf")
        _validate_file(file, b"conteudo qualquer")  # não levanta

    def test_txt_valido(self):
        file = self._make_upload_file("notas.txt")
        _validate_file(file, b"texto qualquer")

    def test_tipo_invalido_levanta_erro(self):
        file = self._make_upload_file("script.exe")
        with pytest.raises(HTTPException) as exc:
            _validate_file(file, b"conteudo")
        assert exc.value.status_code == 400
        assert ".exe" in exc.value.detail

    def test_tipo_csv_nao_permitido(self):
        file = self._make_upload_file("dados.csv")
        with pytest.raises(HTTPException) as exc:
            _validate_file(file, b"a,b,c")
        assert exc.value.status_code == 400

    def test_arquivo_muito_grande_levanta_erro(self, oversized_content):
        file = self._make_upload_file("gigante.pdf")
        with pytest.raises(HTTPException) as exc:
            _validate_file(file, oversized_content)
        assert exc.value.status_code == 413  # 413 = Payload Too Large

    def test_arquivo_vazio_levanta_erro(self):
        file = self._make_upload_file("vazio.txt")
        with pytest.raises(HTTPException) as exc:
            _validate_file(file, b"")
        assert exc.value.status_code == 400
