from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    urls: list[str] | None = None


class ChunkMetadata(BaseModel):
    source: str
    url: str
    chunk_index: int
    char_count: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


class IngestRequest(BaseModel):
    urls: list[str]


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    collection_count: int
    model: str