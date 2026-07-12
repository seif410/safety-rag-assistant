"""Retrieval layer: filter construction, source aggregation, content flattening."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from qdrant_client.models import Filter

from app.core import rag_chain
from app.core.rag_chain import _extract_text, list_indexed_sources, retrieve_with_filter


def _doc(filename, doc_type="regulation", page=1):
    return Document(
        page_content="x",
        metadata={"filename": filename, "doc_type": doc_type, "page": page},
    )


# --- _extract_text --------------------------------------------------------


def test_extract_text_passes_through_plain_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_joins_text_blocks():
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert _extract_text(content) == "ab"


def test_extract_text_ignores_non_text_blocks_keeps_strings():
    content = [{"type": "reasoning", "text": "skip"}, "kept", {"type": "text", "text": "!"}]
    assert _extract_text(content) == "kept!"


def test_extract_text_stringifies_other_types():
    assert _extract_text(42) == "42"


# --- retrieve_with_filter -------------------------------------------------


def test_retrieve_with_filter_no_doc_type_omits_filter():
    retriever = MagicMock()
    retriever.invoke.return_value = [_doc("a.pdf")]
    with patch.object(rag_chain.vectorstore, "as_retriever", return_value=retriever) as asr, \
         patch.object(rag_chain, "rerank_documents", side_effect=lambda q, d: d):
        retrieve_with_filter("query")

    search_kwargs = asr.call_args.kwargs["search_kwargs"]
    assert "filter" not in search_kwargs
    assert search_kwargs["k"] == rag_chain.settings.retrieval_k


def test_retrieve_with_filter_builds_qdrant_filter():
    retriever = MagicMock()
    retriever.invoke.return_value = []
    with patch.object(rag_chain.vectorstore, "as_retriever", return_value=retriever) as asr, \
         patch.object(rag_chain, "rerank_documents", side_effect=lambda q, d: d):
        retrieve_with_filter("query", doc_type="incident_report")

    search_kwargs = asr.call_args.kwargs["search_kwargs"]
    assert isinstance(search_kwargs["filter"], Filter)


def test_retrieve_with_filter_reranks_results():
    retriever = MagicMock()
    retriever.invoke.return_value = [_doc("a.pdf"), _doc("b.pdf")]
    with patch.object(rag_chain.vectorstore, "as_retriever", return_value=retriever), \
         patch.object(rag_chain, "rerank_documents", return_value=[_doc("b.pdf")]) as rr:
        out = retrieve_with_filter("query")

    rr.assert_called_once()
    assert [d.metadata["filename"] for d in out] == ["b.pdf"]


# --- list_indexed_sources -------------------------------------------------


def test_list_indexed_sources_empty_when_collection_missing():
    with patch.object(rag_chain.client, "collection_exists", return_value=False):
        assert list_indexed_sources() == []


def test_list_indexed_sources_aggregates_chunk_counts():
    def point(filename, doc_type):
        return MagicMock(payload={"metadata": {"filename": filename, "doc_type": doc_type}})

    points = [
        point("a.pdf", "regulation"),
        point("a.pdf", "regulation"),
        point("b.md", "incident_report"),
    ]
    with patch.object(rag_chain.client, "collection_exists", return_value=True), \
         patch.object(rag_chain.client, "scroll", return_value=(points, None)):
        sources = list_indexed_sources()

    # Sorted by filename; counts aggregated per file.
    assert sources == [
        {"filename": "a.pdf", "doc_type": "regulation", "chunks": 2},
        {"filename": "b.md", "doc_type": "incident_report", "chunks": 1},
    ]
