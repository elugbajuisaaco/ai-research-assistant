from app.services.chunk_service import create_chunks
from app.services.embedding_service import create_embedding
from app.services.vector_store import InMemoryVectorStore


def index_document(
    text: str,
    vector_store: InMemoryVectorStore,
    chunk_size: int = 1000,
    overlap: int = 200,
    document_id: str = "unknown",
    page: int | None = None,
    filename: str = "unknown",
    start_chunk_id: int = 0,
) -> int:
    """
    Split document text into chunks, generate embeddings,
    and store the chunks with their source document metadata.
    """

    if not text.strip():
        return 0

    chunks = create_chunks(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
        document_id=document_id,
        page=page,
        filename=filename,
        start_chunk_id=start_chunk_id,
    )

    for chunk in chunks:
        embedding = create_embedding(chunk.text)

        vector_store.add(
            chunk,
            embedding,
        )

    return len(chunks)