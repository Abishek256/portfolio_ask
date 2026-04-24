"""
retrieve.py — Semantic search over the FAISS index.

Loads the pre-built index and metadata once (at import time),
then provides a single retrieve() function for the rest of the system.

Important: BGE models require a query prefix during retrieval.
This is documented in the BAAI/bge-small-en-v1.5 model card.
Skipping this prefix degrades retrieval quality measurably.
We apply it here at query time — document embeddings do NOT use this prefix.
"""

import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

INDEX_PATH = Path("data/index/faiss.index")
METADATA_PATH = Path("data/index/metadata.pkl")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# BGE retrieval prefix — must match what the model was trained to expect
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Load index and model once at module import.
# This avoids reloading on every query — model loading takes ~1-2 seconds.
_index: faiss.IndexFlatIP = None
_metadata: List[Dict[str, str]] = None
_model: SentenceTransformer = None


def _load_resources() -> None:
    """Load FAISS index, metadata, and embedding model into module-level cache."""
    global _index, _metadata, _model

    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `make setup` or `python -m portfolio_ask.ingest` first."
        )

    _index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        _metadata = pickle.load(f)
    _model = SentenceTransformer(EMBEDDING_MODEL)


def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most semantically relevant chunks for a query.

    Args:
        query: The user's natural language question.
        k: Number of chunks to return. Default 4.
           Increase for broader questions, decrease for specific lookups.

    Returns:
        List of dicts, each with:
          - 'text': the chunk content
          - 'source': the source file/identifier
          - 'score': cosine similarity score (higher = more relevant)

    The returned chunks are passed directly into the LLM context.
    Order is preserved (highest score first).
    """
    global _index, _metadata, _model

    # Lazy load on first call
    if _index is None:
        _load_resources()

    # Apply BGE query prefix before embedding
    prefixed_query = BGE_QUERY_PREFIX + query

    query_embedding = _model.encode(
        [prefixed_query],
        normalize_embeddings=True
    )
    query_embedding = np.array(query_embedding, dtype="float32")

    scores, indices = _index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 when fewer than k vectors exist
            continue
        chunk = _metadata[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.

    Each chunk is prefixed with its source for citation tracking.
    The LLM is instructed to cite sources by this exact label.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)