from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response

from app.services.document_service import (
    extract_text_from_pdf,
    extract_pages_from_pdf,
)

from app.services.storage_service import (
    save_document,
    get_document,
    delete_document,
    list_documents,
    update_document_metadata,
)

from app.services.indexing_service import index_document
from app.services.rag_dependencies import vector_store


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


# ============================================================
# LIST DOCUMENTS
# ============================================================

@router.get("")
async def get_documents():
    """
    Return all stored PDF documents.
    """

    documents = list_documents()

    return {
        "documents": documents,
        "total": len(documents),
    }


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    """
    Upload, extract, store, and index a PDF document.
    """

    # 1. Validate uploaded file.
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # 2. Read PDF content.
    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    # 3. Extract complete text.
    try:
        text = extract_text_from_pdf(content)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not extract text from PDF: {exc}"
            ),
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "The PDF does not contain extractable text."
            ),
        )

    # 4. Determine filename.
    filename = (
        file.filename
        or "uploaded_document.pdf"
    )

    # 5. Save original PDF.
    document_id = save_document(
        filename,
        content,
    )

    # 6. Extract pages.
    try:
        pages = extract_pages_from_pdf(content)

    except Exception as exc:
        # Remove saved PDF if page extraction fails.
        try:
            delete_document(document_id)
        except FileNotFoundError:
            pass

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not extract PDF pages: {exc}"
            ),
        )

    # 7. Index every page.
    total_chunks = 0
    next_chunk_id = 0

    try:
        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]

            if not page_text.strip():
                continue

            chunks_indexed = index_document(
                text=page_text,
                vector_store=vector_store,
                document_id=document_id,
                page=page_number,
                filename=filename,
                start_chunk_id=next_chunk_id,
            )

            total_chunks += chunks_indexed
            next_chunk_id += chunks_indexed

    except Exception as exc:
        # Remove stored document if indexing fails.
        try:
            delete_document(document_id)
        except FileNotFoundError:
            pass

        # Also remove any chunks that may already
        # have been indexed for this document.
        vector_store.delete_document(
            document_id
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document indexing failed: {exc}"
            ),
        )

    # 8. Return processing information.
    return {
        "id": document_id,
        "filename": filename,
        "content_type": file.content_type,
        "text_length": len(text),
        "pages": len(pages),
        "chunks_indexed": total_chunks,
        "status": "processed",
    }


# ============================================================
# DOWNLOAD DOCUMENT
# ============================================================

@router.get("/{document_id}")
async def download_document(
    document_id: str,
):
    """
    Download a stored PDF document.
    """

    try:
        content = get_document(
            document_id
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f"attachment; filename={document_id}.pdf"
            )
        },
    )


# ============================================================
# UPDATE DOCUMENT METADATA
# ============================================================

@router.patch("/{document_id}/metadata")
async def update_document(
    document_id: str,
    metadata: dict,
):
    """
    Update metadata associated with a document.
    """

    try:
        updated_document = update_document_metadata(
            document_id,
            metadata,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "id": document_id,
        "document": updated_document,
        "status": "updated",
    }


# ============================================================
# DELETE DOCUMENT
# ============================================================

@router.delete("/{document_id}")
async def delete_document_endpoint(
    document_id: str,
):
    """
    Delete a document from storage and remove
    all of its indexed chunks.
    """

    # 1. Delete the PDF from storage.
    try:
        delete_document(
            document_id
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # 2. Remove all indexed chunks.
    deleted_chunks = (
        vector_store.delete_document(
            document_id
        )
    )

    # 3. Return deletion information.
    return {
        "id": document_id,
        "deleted_chunks": deleted_chunks,
        "status": "deleted",
    }