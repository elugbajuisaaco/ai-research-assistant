from fastapi import APIRouter, Depends, HTTPException

from app.models.answer_response import AnswerResponse
from app.services.rag_service import answer_question
from app.services.rag_dependencies import (
    get_vector_store,
    get_llm,
)
from app.services.vector_store import InMemoryVectorStore
from app.services.gemini_service import GeminiLLMService


router = APIRouter(
    prefix="/questions",
    tags=["Questions"],
)


@router.post(
    "/ask",
    response_model=AnswerResponse,
)
async def ask_question(
    question: str,
    vector_store: InMemoryVectorStore = Depends(
        get_vector_store
    ),
    llm: GeminiLLMService = Depends(
        get_llm
    ),
) -> AnswerResponse:

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    return answer_question(
        question=question,
        vector_store=vector_store,
        llm=llm,
        top_k=3,
    )