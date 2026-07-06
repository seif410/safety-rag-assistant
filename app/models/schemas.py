from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    doc_type_filter: str | None = None


class Source(BaseModel):
    filename: str | None
    page: int | None
    doc_type: str | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestRequest(BaseModel):
    path: str
    doc_type: str = "regulation"  # "regulation" | "incident_report" | "procedure"


class IngestResponse(BaseModel):
    filename: str
    pages: int
    chunks: int
    indexed: bool


class SourceInfo(BaseModel):
    filename: str
    doc_type: str | None
    chunks: int


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]
    count: int
