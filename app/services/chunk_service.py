from app.models.chunk import DocumentChunk
from app.services.chunking_service import chunk_text


def create_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    document_id: str = "unknown",
    page: int | None = None,
    filename: str = "unknown",
    start_chunk_id: int = 0,
) -> list[DocumentChunk]:
    """
    Split document text into searchable chunks.

    Each chunk retains its source document ID,
    filename, and PDF page.

    start_chunk_id allows chunk IDs to remain unique
    across multiple pages of the same document.
    """

    if start_chunk_id < 0:
        raise ValueError("start_chunk_id cannot be negative.")

    chunks = chunk_text(
        text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return [
        DocumentChunk(
            chunk_id=start_chunk_id + index,
            text=chunk,
            document_id=document_id,
            filename=filename,
            page=page,
        )
        for index, chunk in enumerate(chunks)
    ]