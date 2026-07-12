"""Pydantic request/response model contracts."""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    Source,
    SourceInfo,
    SourcesResponse,
)


def test_query_request_defaults():
    req = QueryRequest(query="hello")
    assert req.session_id is None
    assert req.doc_type_filter is None


def test_query_request_requires_query():
    with pytest.raises(ValidationError):
        QueryRequest()


def test_ingest_request_default_doc_type():
    assert IngestRequest(path="a.pdf").doc_type == "regulation"


def test_ingest_request_requires_path():
    with pytest.raises(ValidationError):
        IngestRequest(doc_type="regulation")


def test_query_response_roundtrip():
    resp = QueryResponse(
        answer="ok",
        sources=[{"filename": "f.pdf", "page": 3, "doc_type": "regulation"}],
    )
    assert isinstance(resp.sources[0], Source)
    assert resp.sources[0].page == 3


def test_source_page_coerces_numeric_string():
    # Pydantic coerces "5" -> 5 for an int field.
    assert Source(filename="f.pdf", page="5", doc_type="regulation").page == 5


def test_ingest_response_fields():
    r = IngestResponse(filename="f.pdf", pages=2, chunks=7, indexed=True)
    assert (r.pages, r.chunks, r.indexed) == (2, 7, True)


def test_sources_response_nests_source_info():
    resp = SourcesResponse(
        sources=[{"filename": "f.pdf", "doc_type": "regulation", "chunks": 4}],
        count=1,
    )
    assert isinstance(resp.sources[0], SourceInfo)
    assert resp.count == 1
