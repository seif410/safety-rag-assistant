from app.config import settings
from app.core.reranker import rerank_documents
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(url=settings.qdrant_url)


embeddings = NVIDIAEmbeddings(
    model=settings.embedding_model,
    chunk_size=settings.embedding_batch_size,
    retry_min_seconds=settings.embedding_retry_min_seconds,
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=settings.qdrant_collection_name,
    embedding=embeddings,
)


def retrieve_with_filter(
    query: str, doc_type: str | None = None, k: int = settings.retrieval_k
) -> list[Document]:
    """Retrieve relevant documents to help answer user queries about safety regulations."""
    search_kwargs = {"k": k}
    if doc_type:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.doc_type", match=MatchValue(value=doc_type)
                )
            ]
        )
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(query)
    return rerank_documents(query, docs)
