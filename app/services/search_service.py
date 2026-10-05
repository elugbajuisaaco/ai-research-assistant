from app.models.search_result import SearchResult
from app.services.embedding_service import create_embedding
from app.services.vector_store import InMemoryVectorStore

def search_documents(
    query: str,
    vector_store: InMemoryVectorStore,
    top_k: int = 3,
) -> list[SearchResult]:
    """Converts a textual user query into a vector representation and queries the store."""

    if not query.strip():
        return []

    query_embedding = create_embedding(query)

    results = vector_store.search(
        query_embedding,
        top_k=top_k,
    )

    print("\n" + "=" * 80)
    print(f"SEARCH QUERY: {query}")
    print("=" * 80)

    for index, result in enumerate(results, start=1):
        print(f"\nRESULT {index}")
        print(f"Score: {result.score}")
        print(f"Document: {result.chunk.document_id}")
        print(f"Chunk: {result.chunk.chunk_id}")
        print(f"Page: {result.chunk.page}")
        print(f"Text: {result.chunk.text[:500]}")

    return results