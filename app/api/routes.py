from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestRequest,
    IngestResponse,
    SourcesResponse,
)
from app.core.rag_chain import run_query, list_indexed_sources
from app.core.ingestion import ingest_file

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_docs(request: QueryRequest):
    """Ask a question about safety regulations and procedures."""
    result = run_query(query=request.query, session_id=request.session_id or "default")
    return QueryResponse(**result)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest a PDF or Markdown document into the vector store."""
    try:
        result = await ingest_file(request.path, request.doc_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not result["indexed"]:
        raise HTTPException(
            status_code=422, detail=f"No content indexed from {request.path}"
        )
    return IngestResponse(**result)


@router.get("/sources", response_model=SourcesResponse)
async def list_sources():
    """List all indexed document sources."""
    sources = list_indexed_sources()
    return SourcesResponse(sources=sources, count=len(sources))

