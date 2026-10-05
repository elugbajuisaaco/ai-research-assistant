from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """
    Represents a searchable chunk of text extracted from a document.
    """

    chunk_id: int = Field(
        ...,
        description="The unique number of the chunk within the document.",
    )

    text: str = Field(
        ...,
        description="The text contained in this chunk.",
    )

    document_id: str = Field(
        default="unknown",
        description="The ID of the source document.",
    )

    filename: str = Field(
        default="unknown",
        description="The filename of the source document.",
    )

    page: int | None = Field(
        default=None,
        description="The PDF page containing this chunk, when available.",
    )