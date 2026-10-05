from pydantic import BaseModel
from app.models.chunk import DocumentChunk


class SearchResult(BaseModel):
    chunk: DocumentChunk
    score: float
