"""API endpoint behavior via FastAPI TestClient.

The route handlers' dependencies (run_query, ingest_file, list_indexed_sources)
are patched, so these tests check request/response wiring and HTTP status codes,
not the RAG pipeline itself.
"""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api import routes
from app.main import app

client = TestClient(app)


def test_query_returns_answer_and_sources():
    fake = {
        "answer": "Fall protection is required at 6 feet.",
        "sources": [{"filename": "osha.pdf", "page": 12, "doc_type": "regulation"}],
    }
    with patch.object(routes, "run_query", return_value=fake) as rq:
        resp = client.post("/api/v1/query", json={"query": "fall protection height?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == fake["answer"]
    assert body["sources"][0]["filename"] == "osha.pdf"
    rq.assert_called_once()


def test_query_defaults_session_id():
    with patch.object(
        routes, "run_query", return_value={"answer": "ok", "sources": []}
    ) as rq:
        client.post("/api/v1/query", json={"query": "hi"})
    assert rq.call_args.kwargs["session_id"] == "default"


def test_ingest_success():
    summary = {"filename": "r.md", "pages": 1, "chunks": 3, "indexed": True}
    with patch.object(routes, "ingest_file", new=AsyncMock(return_value=summary)):
        resp = client.post(
            "/api/v1/ingest",
            json={"path": "data/incident_reports/r.md", "doc_type": "incident_report"},
        )
    assert resp.status_code == 200
    assert resp.json() == summary


def test_ingest_unsupported_type_returns_400():
    with patch.object(
        routes, "ingest_file", new=AsyncMock(side_effect=ValueError("Unsupported file type: .txt"))
    ):
        resp = client.post("/api/v1/ingest", json={"path": "notes.txt"})
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


def test_ingest_no_content_returns_422():
    summary = {"filename": "empty.md", "pages": 0, "chunks": 0, "indexed": False}
    with patch.object(routes, "ingest_file", new=AsyncMock(return_value=summary)):
        resp = client.post("/api/v1/ingest", json={"path": "empty.md"})
    assert resp.status_code == 422


def test_sources_lists_indexed_documents():
    sources = [{"filename": "a.pdf", "doc_type": "regulation", "chunks": 4}]
    with patch.object(routes, "list_indexed_sources", return_value=sources):
        resp = client.get("/api/v1/sources")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["sources"][0]["chunks"] == 4
