"""
ingest.py — Build the FAISS index from all data sources.

Run this once before using the CLI, or whenever data changes.
Called via: make setup (after deps are installed)

What this does:
  1. Loads portfolio.json, news/*.md, glossary.md
  2. Splits each into paragraph-level chunks
  3. Embeds each chunk with BGE-small-en-v1.5
  4. Stores vectors in a FAISS flat index (exact search, no approximation)
  5. Saves index + metadata to disk at data/index/

Why paragraph-level chunking:
  Each paragraph is a coherent semantic unit. Splitting on double-newlines
  preserves meaning better than fixed-size windows for our structured news data.

Why FAISS FlatL2:
  With ~100 chunks, exact search is instantaneous. No need for approximate
  methods (IVF, HNSW) which are only worthwhile at 100k+ vectors.
"""

import os
import json
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

# Paths
DATA_DIR = Path("data")
NEWS_DIR = DATA_DIR / "news"
INDEX_DIR = DATA_DIR / "index"
INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# Model — loaded once, reused for both ingestion and retrieval
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def chunk_text(text: str, source: str) -> List[Dict[str, str]]:
    """
    Split text into paragraph-level chunks.

    Each chunk is a dict with:
      - 'text': the chunk content
      - 'source': filename/identifier for citation

    Empty paragraphs (whitespace-only) are skipped.
    Paragraphs shorter than 30 characters are merged with the next
    paragraph to avoid creating near-useless tiny chunks.
    """
    raw_paragraphs = text.split("\n\n")
    chunks = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Merge short fragments with the next paragraph
        if len(para) < 30:
            buffer = para + " "
            continue

        full_para = buffer + para
        buffer = ""
        chunks.append({"text": full_para, "source": source})

    # Flush any remaining buffer
    if buffer.strip():
        chunks.append({"text": buffer.strip(), "source": source})

    return chunks


def load_portfolio_chunks() -> List[Dict[str, str]]:
    """
    Convert portfolio.json into retrievable text chunks.
    One chunk per holding — structured as a readable text summary.

    Why text and not raw JSON?
    The embedding model works on natural language. Raw JSON with field
    names like 'avg_cost' embeds poorly compared to a sentence like
    'INFY (Infosys Ltd) is held in the IT sector at an average cost of ₹1380.'
    """
    with open(DATA_DIR / "portfolio.json", "r") as f:
        data = json.load(f)

    chunks = []
    for h in data["holdings"]:
        current_value = h["quantity"] * h["current_price"]
        invested_value = h["quantity"] * h["avg_cost"]
        pnl = current_value - invested_value
        pnl_pct = (pnl / invested_value) * 100

        text = (
            f"{h['ticker']} ({h['name']}) is a {h['type']} holding in the "
            f"{h['sector']} sector, listed on {h['exchange']}. "
            f"The portfolio holds {h['quantity']} units at an average cost of "
            f"₹{h['avg_cost']:,.2f} per unit. Current price is ₹{h['current_price']:,.2f}. "
            f"Invested value: ₹{invested_value:,.2f}. "
            f"Current value: ₹{current_value:,.2f}. "
            f"Unrealised P&L: ₹{pnl:,.2f} ({pnl_pct:.1f}%)."
        )
        chunks.append({"text": text, "source": f"portfolio.json#{h['ticker']}"})

    return chunks


def load_news_chunks() -> List[Dict[str, str]]:
    """Load all news markdown files and chunk by paragraph."""
    chunks = []
    for md_file in sorted(NEWS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        file_chunks = chunk_text(text, source=md_file.name)
        chunks.extend(file_chunks)
    return chunks


def load_glossary_chunks() -> List[Dict[str, str]]:
    """Load glossary.md and chunk by term (each term = one chunk)."""
    text = (DATA_DIR / "glossary.md").read_text(encoding="utf-8")
    return chunk_text(text, source="glossary.md")


def build_index() -> None:
    """
    Main entry point. Loads all data, embeds, builds FAISS index, saves to disk.
    """
    print("Loading data sources...")
    portfolio_chunks = load_portfolio_chunks()
    news_chunks = load_news_chunks()
    glossary_chunks = load_glossary_chunks()

    all_chunks = portfolio_chunks + news_chunks + glossary_chunks
    print(f"  Portfolio chunks : {len(portfolio_chunks)}")
    print(f"  News chunks      : {len(news_chunks)}")
    print(f"  Glossary chunks  : {len(glossary_chunks)}")
    print(f"  Total chunks     : {len(all_chunks)}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]

    # BGE models benefit from a query prefix during retrieval, but for
    # document embeddings we use the text as-is.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True  # Normalise for cosine similarity via dot product
    )

    embeddings = np.array(embeddings, dtype="float32")
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    # FlatIP = exact inner product search (equivalent to cosine similarity
    # when embeddings are normalised, which we did above)
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"Index contains {index.ntotal} vectors")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nIndex saved to   : {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("Ingestion complete.")


if __name__ == "__main__":
    build_index()