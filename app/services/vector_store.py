from dataclasses import dataclass

from app.models.chunk import DocumentChunk
from app.models.search_result import SearchResult
from app.services.similarity_service import cosine_similarity


@dataclass
class VectorRecord:
    chunk: DocumentChunk
    embedding: list[float]


class InMemoryVectorStore:
    """
    Simple in-memory vector database.

    Stores document chunks together with their embeddings
    and retrieves the most similar chunks for a query.
    """

    def __init__(self):
        self._records: list[VectorRecord] = []

    def add(
        self,
        chunk: DocumentChunk,
        embedding: list[float],
    ) -> None:
        """Add a document chunk and its embedding to the store."""

        if not embedding:
            raise ValueError("Embedding cannot be empty.")

        self._records.append(
            VectorRecord(
                chunk=chunk,
                embedding=embedding,
            )
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """
        Find the chunks most similar to the query embedding.
        """

        if not query_embedding:
            return []

        if top_k <= 0:
            return []

        results: list[SearchResult] = []

        for record in self._records:
            score = cosine_similarity(
                query_embedding,
                record.embedding,
            )

            results.append(
                SearchResult(
                    chunk=record.chunk,
                    score=float(score),
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """
        Remove all chunks belonging to a document.

        Returns the number of chunks removed.
        """

        original_count = len(self._records)

        self._records = [
            record
            for record in self._records
            if record.chunk.document_id != document_id
        ]

        return original_count - len(self._records)

    def clear(self) -> None:
        """Remove every stored document chunk."""

        self._records.clear()