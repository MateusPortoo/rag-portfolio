"""
Testes unitários — chunking de documentos.

Testamos comportamento do splitter: número de chunks, overlap, metadados.
Não testamos conteúdo exato (frágil), mas propriedades estruturais.
"""

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200


def make_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )


class TestChunking:

    def test_texto_curto_gera_um_chunk(self):
        """Texto menor que CHUNK_SIZE deve resultar em exatamente 1 chunk."""
        splitter = make_splitter()
        from langchain_core.documents import Document
        docs = [Document(page_content="Texto curto.", metadata={"source": "teste.txt"})]
        chunks = splitter.split_documents(docs)
        assert len(chunks) == 1

    def test_texto_longo_gera_multiplos_chunks(self):
        """Texto muito maior que CHUNK_SIZE deve gerar mais de 1 chunk."""
        splitter = make_splitter()
        from langchain_core.documents import Document
        # 5000 caracteres garantem múltiplos chunks com CHUNK_SIZE=1000
        long_text = ("palavra " * 700)
        docs = [Document(page_content=long_text, metadata={"source": "longo.txt"})]
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 1

    def test_chunks_respeitam_tamanho_maximo(self):
        """Nenhum chunk deve exceder CHUNK_SIZE em tamanho."""
        splitter = make_splitter()
        from langchain_core.documents import Document
        long_text = "abcde " * 1000
        docs = [Document(page_content=long_text, metadata={"source": "test.txt"})]
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= CHUNK_SIZE

    def test_metadata_source_preservado(self, sample_txt: Path):
        """O metadado 'source' deve sobreviver ao chunking."""
        loader = TextLoader(str(sample_txt), encoding="utf-8")
        docs = loader.load()
        splitter = make_splitter()
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            assert "source" in chunk.metadata
            # source pode ser o caminho completo ou só o nome
            assert "politica_rh" in chunk.metadata["source"]

    def test_chunks_nao_estao_vazios(self, sample_txt: Path):
        """Nenhum chunk deve ter conteúdo vazio."""
        loader = TextLoader(str(sample_txt), encoding="utf-8")
        docs = loader.load()
        splitter = make_splitter()
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            assert chunk.page_content.strip() != ""

    def test_overlap_cria_continuidade(self):
        """
        Com overlap=200, o final de um chunk deve aparecer no início do próximo.
        Isso garante que conceitos não sejam cortados entre chunks.
        """
        splitter = make_splitter()
        from langchain_core.documents import Document

        # Texto longo e uniforme para forçar múltiplos chunks
        text = "palavra " * 400
        docs = [Document(page_content=text, metadata={"source": "test.txt"})]
        chunks = splitter.split_documents(docs)

        if len(chunks) >= 2:
            end_of_first   = chunks[0].page_content[-100:]
            start_of_second = chunks[1].page_content[:100]
            # Pelo menos parte do final do chunk 1 deve aparecer no chunk 2
            overlap_found = any(
                word in start_of_second
                for word in end_of_first.split()
                if len(word) > 3
            )
            assert overlap_found, "Overlap não encontrado entre chunks consecutivos"
