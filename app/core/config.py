from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    gemini_api_key: str
    chroma_persist_dir: str = "./data/chroma_db"
    collection_name: str = "arabic_rag"
    top_k: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"

settings = Settings()