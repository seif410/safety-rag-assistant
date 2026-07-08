import asyncio

import pymupdf

from pathlib import Path
from app.config import settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from logger import Colors, log_error, log_header, log_info, log_success, log_warning

embeddings = NVIDIAEmbeddings(
    model=settings.embedding_model,
    chunk_size=settings.embedding_batch_size,
    retry_min_seconds=settings.embedding_retry_min_seconds,
)

client = QdrantClient(url=settings.qdrant_url)


def ensure_collection() -> None:
    """Create the collection if missing, sized to the embedding model."""
    if client.collection_exists(settings.qdrant_collection_name):
        return
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        collection_name=settings.qdrant_collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    log_success(
        f"Qdrant: Created collection '{settings.qdrant_collection_name}' (dim={dim})"
    )


ensure_collection()
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=settings.qdrant_collection_name,
    embedding=embeddings,
)


def extract_pdf_with_metadata(
    pdf_path: str, doc_type: str = "regulation"
) -> list[Document]:
    """Extract text from PDF with per-page metadata."""
    log_info(f"PDF Extraction: Opening {pdf_path}", Colors.PURPLE)
    try:
        doc = pymupdf.open(pdf_path)
    except FileNotFoundError:
        log_error(f"PDF Extraction: File not found - {pdf_path}")
        return []
    except Exception as e:
        log_error(f"PDF Extraction: Failed to open {pdf_path} - {e}")
        return []

    documents = []
    for page_num, page in enumerate(doc):
        try:
            text = page.get_text()
        except Exception as e:
            log_warning(
                f"PDF Extraction: Failed to read page {page_num + 1} of {pdf_path} - {e}"
            )
            continue
        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path,
                        "page": page_num + 1,
                        "doc_type": doc_type,  # "regulation" | "incident_report"
                        "filename": Path(pdf_path).name,
                    },
                )
            )
    log_success(
        f"PDF Extraction: Extracted {len(documents)} pages from {Path(pdf_path).name}"
    )
    return documents


def extract_markdown_with_metadata(
    md_path: str, doc_type: str = "incident_report"
) -> list[Document]:
    """Extract text from a Markdown file into a single Document with metadata."""
    log_info(f"Markdown Extraction: Opening {md_path}", Colors.PURPLE)
    try:
        text = Path(md_path).read_text(encoding="utf-8")
    except FileNotFoundError:
        log_error(f"Markdown Extraction: File not found - {md_path}")
        return []
    except Exception as e:
        log_error(f"Markdown Extraction: Failed to read {md_path} - {e}")
        return []

    if not text.strip():
        log_warning(f"Markdown Extraction: Empty file - {md_path}")
        return []

    log_success(f"Markdown Extraction: Extracted {Path(md_path).name}")
    return [
        Document(
            page_content=text,
            metadata={
                "source": md_path,
                "doc_type": doc_type,  # "regulation" | "incident_report"
                "filename": Path(md_path).name,
            },
        )
    ]


def chunk_documents(documents: list[Document]) -> list[Document]:
    log_info(
        f"Chunking: Splitting {len(documents)} documents into chunks", Colors.PURPLE
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    log_success(f"Chunking: Created {len(chunks)} chunks")
    return chunks


async def index_documents_async(documents: list[Document], batch_size: int = 50):
    log_header("VECTOR STORAGE PHASE")

    log_info(
        f"VectorStore Indexing: Preparing to add {len(documents)} to vector store",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"VectoreStore Indexing: Split into {len(batches)} of {batch_size} documents each"
    )

    async def add_batch(batch: list[Document], batch_num: int):
        try:
            await vectorstore.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} {len(batch)} documents"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successfull = sum(1 for result in results if result is True)

    if successfull == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successfull}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successfull}/{len(batches)} batches successfully"
        )

    return successfull == len(batches)


async def ingest_file(path: str, doc_type: str = "regulation") -> dict:
    """Extract, chunk, and index a single PDF or Markdown file.

    Returns a summary dict: {filename, pages, chunks, indexed}.
    Raises ValueError for unsupported file types.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        documents = extract_pdf_with_metadata(path, doc_type)
    elif ext in (".md", ".markdown"):
        documents = extract_markdown_with_metadata(path, doc_type)
    else:
        raise ValueError(f"Unsupported file type: {ext or '(none)'}")

    summary = {
        "filename": Path(path).name,
        "pages": len(documents),
        "chunks": 0,
        "indexed": False,
    }
    if not documents:
        log_warning(f"Ingestion: No content extracted from {path}")
        return summary

    chunks = chunk_documents(documents)
    summary["chunks"] = len(chunks)
    summary["indexed"] = await index_documents_async(chunks)
    return summary
