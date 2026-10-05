from pydantic import BaseModel
from app.models.chunk import DocumentChunk


class VectorRecord(BaseModel):
    chunk: DocumentChunk
    embedding: list[float]
