from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_api_key: str
    google_api_key: str
    cohere_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "safety-docs"
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    cohere_rerank_model: str = "rerank-v4.0-pro"
    chat_model: str = "gemini-2.5-flash"
    chat_model_provider: str = "google_genai"
    embedding_batch_size: int = 50
    embedding_retry_min_seconds: int = 10
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 6
    top_n: int = 4


settings = Settings()  # reads .env once, validated
