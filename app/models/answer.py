from typing import List

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a source citation supporting an answer."""

    citation_number: int = Field(
        ...,
        description="Citation number used in the generated answer.",
    )

    document_id: str = Field(
        ...,
        description="Unique identifier of the source document.",
    )

    filename: str = Field(
        default="unknown",
        description="Original filename of the source document.",
    )

    chunk_id: str = Field(
        ...,
        description="Unique identifier of the source chunk.",
    )

    page: int | None = Field(
        default=None,
        description="PDF page containing the source chunk.",
    )

    score: float = Field(
        ...,
        description="Similarity score of the source chunk.",
    )


class AnswerResponse(BaseModel):
    """Final answer together with supporting source citations."""

    answer: str = Field(
        ...,
        description="Generated answer.",
    )

    citations: List[Citation] = Field(
        default_factory=list,
        description="Sources supporting the answer.",
    )


__all__ = [
    "Citation",
    "AnswerResponse",
]