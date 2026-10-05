import re

from app.models.answer import AnswerResponse, Citation
from app.services.llm_service import LLMService
from app.services.search_service import search_documents
from app.services.vector_store import InMemoryVectorStore


def _validate_citations(
    answer: str,
    number_of_sources: int,
) -> str:
    """
    Remove citation numbers that do not correspond
    to retrieved sources.
    """

    if not answer.strip() or number_of_sources == 0:
        return answer

    def replace_invalid_citation(match):
        citation_number = int(match.group(1))

        if 1 <= citation_number <= number_of_sources:
            return match.group(0)

        return ""

    return re.sub(
        r"\[(\d+)\]",
        replace_invalid_citation,
        answer,
    )


def _build_citation_context(search_results) -> str:
    """
    Build a detailed context block for the LLM.

    Each source contains its citation number,
    document ID, filename, page number, and text.
    """

    context_fragments = []

    for index, result in enumerate(search_results, start=1):
        document_id = str(result.chunk.document_id)
        filename = str(result.chunk.filename)

        page = result.chunk.page

        if page is None:
            page_text = "Unknown"
        else:
            page_text = str(page)

        context_fragments.append(
            f"[{index}]\n"
            f"Document ID: {document_id}\n"
            f"Filename: {filename}\n"
            f"Page: {page_text}\n"
            f"Content:\n"
            f"{result.chunk.text}"
        )

    return "\n\n".join(context_fragments)


def answer_question(
    question: str,
    vector_store: InMemoryVectorStore,
    llm: LLMService,
    top_k: int = 3,
) -> AnswerResponse:
    """
    Search the vector store, build detailed citation context,
    generate an answer, validate citations, and return
    a structured response.
    """

    # 1. Validate the question.
    if not question.strip():
        return AnswerResponse(
            answer="Please provide a valid question.",
            citations=[],
        )

    # 2. Search for relevant document chunks.
    search_results = search_documents(
        question,
        vector_store,
        top_k=top_k,
    )

    # 3. Build detailed citation-aware context.
    compiled_context = _build_citation_context(
        search_results
    )

    # 4. Generate the answer.
    answer_text = llm.generate(
        question,
        compiled_context,
    )

    # 5. Validate citation numbers.
    answer_text = _validate_citations(
        answer_text,
        len(search_results),
    )

    # 6. Convert search results into citations.
    citations = []

    for index, result in enumerate(search_results, start=1):
        citations.append(
            Citation(
                citation_number=index,
                document_id=str(result.chunk.document_id),
                filename=str(result.chunk.filename),
                chunk_id=str(result.chunk.chunk_id),
                page=result.chunk.page,
                score=float(result.score),
            )
        )

    # 7. Return the structured answer.
    return AnswerResponse(
        answer=answer_text,
        citations=citations,
    )