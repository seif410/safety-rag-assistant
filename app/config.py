from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_api_key: str
    google_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "safety-docs"
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    embedding_batch_size: int = 50
    embedding_retry_min_seconds: int = 10
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 6


settings = Settings()  # reads .env once, validated
