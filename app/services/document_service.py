from io import BytesIO
import re

from pypdf import PdfReader


def _clean_pdf_text(text: str) -> str:
    """
    Clean common PDF text-extraction artifacts.
    """

    if not text:
        return ""

    # Repair words split across a line with a hyphen.
    # Example:
    # machine-
    # learning
    # becomes:
    # machinelearning
    text = re.sub(
        r"-\s*\n\s*",
        "",
        text,
    )

    # Replace line breaks with spaces.
    text = re.sub(
        r"\s*\n\s*",
        " ",
        text,
    )

    # Remove excessive whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # Repair common PDF extraction issue where a space
    # appears immediately after the first character.
    # Example:
    # M achine -> Machine
    text = re.sub(
        r"\b([A-Z])\s+([a-z]{2,})\b",
        r"\1\2",
        text,
    )

    return text.strip()


def extract_text_from_pdf(file_content: bytes) -> str:
    """
    Extract all text from a PDF as one combined string.
    """

    reader = PdfReader(BytesIO(file_content))

    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            cleaned_text = _clean_pdf_text(page_text)

            if cleaned_text:
                text_parts.append(cleaned_text)

    return "\n".join(text_parts)


def extract_pages_from_pdf(file_content: bytes) -> list[dict]:
    """
    Extract PDF text while preserving page numbers.

    Page numbers are 1-based.
    """

    reader = PdfReader(BytesIO(file_content))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        page_text = page.extract_text()

        if page_text:
            cleaned_text = _clean_pdf_text(page_text)

            if cleaned_text:
                pages.append(
                    {
                        "page": page_number,
                        "text": cleaned_text,
                    }
                )

    return pages