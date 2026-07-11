from app.config import settings
from logger import log_info, log_success
from cohere.errors import TooManyRequestsError
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import logging

logger = logging.getLogger(__name__)

reranker = CohereRerank(
    model=settings.cohere_rerank_model,
    top_n=settings.top_n,
    cohere_api_key=settings.cohere_api_key,
)


@retry(
    # Cohere trial keys cap at 10 rerank calls/min; a burst (e.g. eval over the
    # whole Q&A set) hits HTTP 429. Back off and retry so the minute window clears.
    retry=retry_if_exception_type(TooManyRequestsError),
    wait=wait_exponential(multiplier=1, min=settings.cohere_retry_min_seconds, max=60),
    stop=stop_after_attempt(settings.cohere_retry_max_attempts),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    """Reorder retrieved docs by relevance. Retrieve 6-8, keep top 4."""
    log_info(f"Reranking {len(docs)} docs for query: {query[:60]}")
    results = reranker.compress_documents(documents=docs, query=query)
    log_success(f"Reranked → kept {len(results)} docs")
    return list(results)
