import re


def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences while preserving the sentence-ending
    punctuation.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """
    Split text into semantically cleaner chunks.

    The function prefers sentence boundaries instead of cutting
    directly through sentences.

    If an individual sentence is larger than chunk_size, it is
    split into smaller pieces so the maximum chunk size is respected.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    if not text:
        return []

    text = text.strip()

    if not text:
        return []

    sentences = _split_into_sentences(text)

    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_length = 0

    for sentence in sentences:

        sentence_length = len(sentence)

        # If the sentence itself is larger than chunk_size,
        # flush the current chunk first.
        if sentence_length > chunk_size:

            if current_sentences:
                chunks.append(
                    " ".join(current_sentences).strip()
                )

                current_sentences = []
                current_length = 0

            # Split the long sentence into fixed-size pieces.
            start = 0

            while start < sentence_length:
                end = start + chunk_size

                piece = sentence[start:end].strip()

                if piece:
                    chunks.append(piece)

                if end >= sentence_length:
                    break

                start = end - overlap

            continue

        # Add the sentence to the current chunk if it fits.
        proposed_length = (
            current_length
            + sentence_length
            + (1 if current_sentences else 0)
        )

        if proposed_length <= chunk_size:
            current_sentences.append(sentence)

            current_length = proposed_length

        else:
            # Save the current chunk.
            if current_sentences:
                chunks.append(
                    " ".join(current_sentences).strip()
                )

            # Start the next chunk with the current sentence.
            current_sentences = [sentence]
            current_length = sentence_length

    # Add the final chunk.
    if current_sentences:
        chunks.append(
            " ".join(current_sentences).strip()
        )

    return chunks