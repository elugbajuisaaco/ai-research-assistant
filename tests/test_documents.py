import pytest
from io import BytesIO

from fastapi.testclient import TestClient
from pydantic import ValidationError
from reportlab.pdfgen import canvas

from app.main import app

from app.models.answer import Citation, AnswerResponse
from app.models.answer_response import AnswerResponse as RAGAnswerResponse
from app.models.chunk import DocumentChunk
from app.models.search_result import SearchResult

from app.services.chunking_service import chunk_text
from app.services.chunk_service import create_chunks
from app.services.document_service import (
    extract_text_from_pdf,
    extract_pages_from_pdf,
)
from app.services.embedding_service import create_embedding
from app.services.indexing_service import index_document
from app.services.llm_service import LLMService
from app.services.rag_service import answer_question
from app.services.retrieval_service import retrieve_relevant_chunks
from app.services.search_service import search_documents
from app.services.similarity_service import cosine_similarity

from app.services.storage_service import (
    save_document,
    get_document,
    delete_document,
)

from app.services.vector_store import InMemoryVectorStore
from app.services.gemini_service import GeminiLLMService
from app.services.rag_dependencies import vector_store


client = TestClient(app)
def test_save_document(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    content = b"test document content"

    document_id = save_document(
        "research.pdf",
        content,
    )

    assert document_id

    saved_files = list(
        (tmp_path / "data").glob("*.pdf")
    )

    assert len(saved_files) == 1
    assert saved_files[0].suffix == ".pdf"
    assert saved_files[0].read_bytes() == content