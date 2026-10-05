from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.questions import router as questions_router


app = FastAPI(
    title="AI Research Assistant",
    description="AI-powered document analysis platform",
    version="0.1.0",
    redirect_slashes=False,
)


app.include_router(documents_router)
app.include_router(questions_router)


@app.get("/")
def root():
    return {
        "message": "AI Research Assistant API is running"
    }