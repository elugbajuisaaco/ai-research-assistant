from app.services.vector_store import InMemoryVectorStore
from app.services.gemini_service import GeminiLLMService


# Shared vector store for the application.
vector_store = InMemoryVectorStore()


# Create the real Gemini service only when requested.
def get_llm() -> GeminiLLMService:
    return GeminiLLMService()


def get_vector_store() -> InMemoryVectorStore:
    return vector_store