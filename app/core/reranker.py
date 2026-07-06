from app.config import settings
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

reranker = CohereRerank(
    model=settings.cohere_rerank_model,
    top_n=settings.top_n,
    cohere_api_key=settings.cohere_api_key,
)


def rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    """Reorder retrieved docs by relevance. Retrieve 6-8, keep top 4."""
    results = reranker.compress_documents(documents=docs, query=query)
    return list(results)
