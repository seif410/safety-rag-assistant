"""Shared test setup.

The app builds Qdrant / NVIDIA / Cohere clients — and calls ``ensure_collection()``
— at *import* time (see ``app/core/ingestion.py`` and ``app/core/rag_chain.py``).
Importing the app in a test would therefore hit the network and require real API
keys. This module runs before any test imports ``app`` and:

1. Sets dummy secrets so ``app.config.Settings()`` validates without a real ``.env``.
2. Replaces the client constructors with mocks so import touches no network.
   ``collection_exists`` returns True, which makes ``ensure_collection()`` a no-op.
"""

import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia")
os.environ.setdefault("GOOGLE_API_KEY", "test-google")
os.environ.setdefault("COHERE_API_KEY", "test-cohere")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

# One shared fake Qdrant client so tests can configure it via the imported module
# attribute (ingestion.client and rag_chain.client resolve to this same instance).
_fake_qdrant_client = MagicMock(name="QdrantClient")
_fake_qdrant_client.collection_exists.return_value = True

# Started for the whole test process; no teardown needed — the process is torn
# down when pytest exits.
patch("qdrant_client.QdrantClient", return_value=_fake_qdrant_client).start()
patch(
    "langchain_qdrant.QdrantVectorStore",
    return_value=MagicMock(name="QdrantVectorStore"),
).start()
patch(
    "langchain_nvidia_ai_endpoints.NVIDIAEmbeddings",
    return_value=MagicMock(name="NVIDIAEmbeddings"),
).start()
patch(
    "langchain_cohere.CohereRerank",
    return_value=MagicMock(name="CohereRerank"),
).start()
