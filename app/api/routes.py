from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest, QueryResponse,
    IngestRequest, IngestResponse,
    HealthResponse
)
from app.core.rag import ArabicRAGPipeline
from app.core.loader import WebLoader
from app.core.embedder import ArabicEmbedder

router = APIRouter()

# Initialize pipeline once at startup
pipeline = ArabicRAGPipeline()
loader = WebLoader()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check API status and collection size"""
    return HealthResponse(
        status="ok",
        collection_count=pipeline.embedder.get_collection_count(),
        model="gemini-2.0-flash"
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Scrape Arabic content from provided URLs,
    embed and store in ChromaDB.
    """
    try:
        documents = loader.load_urls(request.urls)
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No content extracted from provided URLs"
            )
        pipeline.embedder.embed_documents(documents)
        loader.save_to_processed(documents)
        return IngestResponse(
            message="Ingestion complete",
            chunks_indexed=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Answer an Arabic question using RAG pipeline.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    try:
        result = pipeline.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))