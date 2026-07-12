"""Ingestion pipeline: extraction, chunking, and the ingest_file dispatcher.

Network-touching pieces (embeddings, Qdrant) are mocked in conftest, so these
tests exercise the pure extract/chunk/dispatch logic offline.
"""

import asyncio
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document

from app.core import ingestion
from app.core.ingestion import (
    chunk_documents,
    extract_markdown_with_metadata,
    extract_pdf_with_metadata,
    ingest_file,
)


def test_extract_markdown_returns_single_document(tmp_path):
    md = tmp_path / "incident_report_001.md"
    md.write_text("# Incident\nA worker slipped.", encoding="utf-8")

    docs = extract_markdown_with_metadata(str(md), doc_type="incident_report")

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["filename"] == "incident_report_001.md"
    assert meta["doc_type"] == "incident_report"
    assert meta["source"] == str(md)
    # Markdown docs have no per-page metadata (single blob).
    assert "page" not in meta


def test_extract_markdown_missing_file_returns_empty():
    assert extract_markdown_with_metadata("does/not/exist.md") == []


def test_extract_markdown_empty_file_returns_empty(tmp_path):
    md = tmp_path / "empty.md"
    md.write_text("   \n  ", encoding="utf-8")
    assert extract_markdown_with_metadata(str(md)) == []


def test_extract_pdf_missing_file_returns_empty():
    assert extract_pdf_with_metadata("does/not/exist.pdf") == []


def test_chunk_documents_splits_and_preserves_metadata():
    long_text = "sentence. " * 500  # ~5000 chars, well over chunk_size
    doc = Document(page_content=long_text, metadata={"filename": "big.md"})

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    assert all(c.metadata["filename"] == "big.md" for c in chunks)
    # Every chunk stays within the configured size.
    assert all(len(c.page_content) <= ingestion.settings.chunk_size for c in chunks)


def test_ingest_file_rejects_unsupported_extension():
    try:
        asyncio.run(ingest_file("notes.txt"))
    except ValueError as e:
        assert "Unsupported file type" in str(e)
    else:
        raise AssertionError("expected ValueError for .txt")


def test_ingest_file_no_content_skips_indexing(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")

    with patch.object(ingestion, "index_documents_async", new=AsyncMock()) as idx:
        summary = asyncio.run(ingest_file(str(empty), doc_type="incident_report"))

    assert summary == {
        "filename": "empty.md",
        "pages": 0,
        "chunks": 0,
        "indexed": False,
    }
    idx.assert_not_called()


def test_ingest_file_happy_path_indexes(tmp_path):
    md = tmp_path / "report.md"
    md.write_text("Root cause: missing guardrail. " * 100, encoding="utf-8")

    with patch.object(
        ingestion, "index_documents_async", new=AsyncMock(return_value=True)
    ) as idx:
        summary = asyncio.run(ingest_file(str(md), doc_type="incident_report"))

    assert summary["filename"] == "report.md"
    assert summary["pages"] == 1
    assert summary["chunks"] >= 1
    assert summary["indexed"] is True
    idx.assert_awaited_once()
