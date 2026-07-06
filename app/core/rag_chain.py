from app.config import settings
from app.core.reranker import rerank_documents
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import ToolMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

client = QdrantClient(url=settings.qdrant_url)
chat_histories: dict[str, InMemoryChatMessageHistory] = {}

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


def list_indexed_sources() -> list[dict]:
    """Aggregate distinct indexed documents by scrolling the Qdrant collection.

    Returns one entry per filename with its doc_type and chunk count.
    Returns an empty list if the collection does not exist yet.
    """
    if not client.collection_exists(settings.qdrant_collection_name):
        return []

    sources: dict[str, dict] = {}
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=settings.qdrant_collection_name,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_offset,
        )
        for point in points:
            metadata = (point.payload or {}).get("metadata", {})
            filename = metadata.get("filename")
            if not filename:
                continue
            entry = sources.setdefault(
                filename,
                {
                    "filename": filename,
                    "doc_type": metadata.get("doc_type"),
                    "chunks": 0,
                },
            )
            entry["chunks"] += 1
        if next_offset is None:
            break

    return sorted(sources.values(), key=lambda s: s["filename"])


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


@tool(response_format="content_and_artifact")
def retrieve_safety_docs(query: str, doc_type: str | None = None):
    """Retrieve relevant safety documentation, regulations, or incident reports.
    Optionally filter by doc_type: 'regulation', or 'incident_report'."""
    docs = retrieve_with_filter(query, doc_type)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('filename','unknown')} (p.{doc.metadata.get('page','')})"
        f"\nType: {doc.metadata.get('doc_type','unknown')}"
        f"\nContent: {doc.page_content}"
        for doc in docs
    )
    return serialized, docs


SYSTEM_PROMPT = (
    "You are a safety compliance assistant for industrial site operators. "
    "You answer questions about OSHA regulations, safety procedures, and incident reports. "
    "Use the retrieve_safety_docs tool to find relevant information before answering. "
    "Always cite sources with document name and page number. "
    "If the user asks about a specific document type (regulation vs incident report), "
    "use the doc_type filter. "
    "If you cannot find the answer in retrieved documents, say so clearly."
)


def _extract_text(content) -> str:
    """Flatten message content to a plain string.

    Some providers return content as a list of blocks (e.g. text + reasoning)
    instead of a bare string; keep only the text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            if isinstance(block, dict) and block.get("type") == "text"
            else block
            if isinstance(block, str)
            else ""
            for block in content
        )
    return str(content)


def run_query(query: str, session_id: str = "default") -> dict:
    """Run RAG pipeline with conversational memory."""
    llm = init_chat_model(
        model=settings.chat_model,
        model_provider=settings.chat_model_provider,
        api_key=settings.google_api_key,
    )
    agent = create_agent(
        llm,
        tools=[retrieve_safety_docs],
        system_prompt=SYSTEM_PROMPT,
    )
    history = get_history(session_id)
    past_messages = history.messages

    messages = list(past_messages) + [{"role": "user", "content": query}]
    response = agent.invoke({"messages": messages})
    answer = _extract_text(response["messages"][-1].content)

    history.add_user_message(query)
    history.add_ai_message(answer)

    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "sources": [
            {
                "filename": doc.metadata.get("filename"),
                "page": doc.metadata.get("page"),
                "doc_type": doc.metadata.get("doc_type"),
            }
            for doc in context_docs
        ],
    }
