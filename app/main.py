from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Arabic RAG Agent",
    description="مساعد المعرفة — Arabic knowledge assistant powered by WHO sources and Gemini",
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    return {
        "message": "Arabic RAG Agent is running",
        "docs": "/docs",
        "health": "/api/v1/health"
    }