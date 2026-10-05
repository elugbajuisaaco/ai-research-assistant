import json
import uuid
from pathlib import Path


STORAGE_DIR = Path("data")
METADATA_FILE = STORAGE_DIR / "metadata.json"


def _load_metadata() -> dict:
    """Load document metadata from disk."""

    if not METADATA_FILE.exists():
        return {}

    try:
        return json.loads(
            METADATA_FILE.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError:
        return {}


def _save_metadata(metadata: dict) -> None:
    """Save document metadata to disk."""

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=4,
        ),
        encoding="utf-8",
    )


def save_document(
    filename: str,
    content: bytes,
) -> str:
    """
    Save a PDF document using a unique document ID.
    """

    document_id = str(uuid.uuid4())

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        STORAGE_DIR
        / f"{document_id}.pdf"
    )

    file_path.write_bytes(content)

    metadata = _load_metadata()

    metadata[document_id] = {
        "id": document_id,
        "filename": filename,
        "content_type": "application/pdf",
        "status": "stored",
    }

    _save_metadata(metadata)

    return document_id


def get_document(
    document_id: str,
) -> bytes:
    """
    Retrieve a stored PDF document by document ID.
    """

    file_path = (
        STORAGE_DIR
        / f"{document_id}.pdf"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document {document_id} not found."
        )

    return file_path.read_bytes()


def get_document_metadata(
    document_id: str,
) -> dict:
    """
    Retrieve metadata for a document.
    """

    metadata = _load_metadata()

    if document_id not in metadata:
        raise FileNotFoundError(
            f"Document {document_id} not found."
        )

    return metadata[document_id]


def list_documents() -> list[dict]:
    """
    Return metadata for all stored documents.
    """

    metadata = _load_metadata()

    return list(metadata.values())


def update_document_metadata(
    document_id: str,
    **updates,
) -> None:
    """
    Update metadata for an existing document.
    """

    metadata = _load_metadata()

    if document_id not in metadata:
        raise FileNotFoundError(
            f"Document {document_id} not found."
        )

    metadata[document_id].update(updates)

    _save_metadata(metadata)


def delete_document(
    document_id: str,
) -> None:
    """
    Delete a stored PDF document by document ID.
    """

    file_path = (
        STORAGE_DIR
        / f"{document_id}.pdf"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document {document_id} not found."
        )

    file_path.unlink()

    metadata = _load_metadata()

    if document_id in metadata:
        del metadata[document_id]
        _save_metadata(metadata)