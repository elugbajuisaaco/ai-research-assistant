from app.models.chunk import DocumentChunk
from app.models.search_result import SearchResult
from app.services.embedding_service import create_embedding
from app.services.similarity_service import cosine_similarity


def retrieve_relevant_chunks(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 3,
) -> list[SearchResult]:
    # Handle edge case: empty chunks list or empty query string
    if not chunks or not query.strip():
        return []

    # 1. Embed the query
    query_vector = create_embedding(query)
    
    search_results = []
    
    # 2 & 3. Embed every chunk and calculate its similarity score
    for chunk in chunks:
        chunk_vector = create_embedding(chunk.text)
        score = cosine_similarity(query_vector, chunk_vector)
        
        search_results.append(
            SearchResult(chunk=chunk, score=score)
        )
    
    # 4. Rank the chunks (highest similarity score first)
    search_results.sort(key=lambda result: result.score, reverse=True)
    
    # 5. Return only top_k
    return search_results[:top_k]
