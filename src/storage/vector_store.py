"""
storage/vector_store.py
-----------------------
ChromaDB-backed vector store with OpenAI embeddings.

Why ChromaDB over raw numpy:
  - Persistent on disk out of the box (no manual .npy save/load)
  - Metadata filtering built-in (filter by call_id, topic_tag without post-processing)
  - Handles embedding + storage in one call via EmbeddingFunction
  - HNSW index under the hood → scales to millions of chunks
  - No server needed: chromadb.PersistentClient runs embedded

Collection: one global "chunks" collection across all calls.
Embedding:  OpenAI text-embedding-3-small via chromadb's OpenAIEmbeddingFunction.
Fallback:   chromadb's DefaultEmbeddingFunction (all-MiniLM-L6-v2 via ONNX)
            — activates automatically when OPENAI_API_KEY is not set.

External interface (used by Retriever):
  add_chunks(chunk_ids, texts, metadatas)  → int  (number added)
  search(query, top_k, call_id_filter)     → list[{chunk_id, score}]
  total_chunks                             → int
"""

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

VECTOR_DIR      = Path(os.getenv("VECTOR_DIR", "data/vectors"))
COLLECTION_NAME = "chunks"
EMBED_MODEL     = os.getenv("EMBED_MODEL", "text-embedding-3-small")


# ── Embedding function selection ───────────────────────────────────────────────

def _embedding_fn():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        print(f"  [VectorStore] embedding=OpenAI/{EMBED_MODEL}")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=EMBED_MODEL,
        )
    # Fallback: ONNX-backed sentence-transformers, no API key needed
    print("  [VectorStore] No OPENAI_API_KEY — using DefaultEmbeddingFunction (all-MiniLM-L6-v2)")
    return embedding_functions.DefaultEmbeddingFunction()


# ── VectorStore ────────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Metadata stored per chunk (enables server-side filtering):
      - call_id    : "call_001"
      - topic_tag  : "pricing" | "security" | ...
      - start_time : "01:24"
      - end_time   : "02:10"
    """

    def __init__(self):
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self._embed_fn = _embedding_fn()

        # PersistentClient: data survives restarts, lives at VECTOR_DIR
        self._client = chromadb.PersistentClient(path=str(VECTOR_DIR))

        # get_or_create → idempotent on restart
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

        print(f"  [VectorStore] backend=chromadb  "
              f"collection={COLLECTION_NAME}  "
              f"chunks_loaded={self._col.count()}")

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict] = None,
    ) -> int:
        """
        Embed and upsert chunks into ChromaDB.

        Uses upsert (not add) so re-ingesting the same file is safe.
        metadatas: list of dicts with call_id, topic_tag, start_time, end_time.
        Returns number of chunks upserted.
        """
        if not chunk_ids:
            return 0

        metadatas = metadatas or [{} for _ in chunk_ids]

        # ChromaDB upsert: insert if new, update if existing (idempotent)
        self._col.upsert(
            ids=chunk_ids,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunk_ids)

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        call_id_filter: str = None,
        topic_filter: str = None,
    ) -> list[dict]:
        """
        Semantic search. Returns [{chunk_id, score, metadata}].

        call_id_filter and topic_filter use ChromaDB's native where-clause
        filtering — happens server-side before ranking, not post-hoc.
        """
        if self._col.count() == 0:
            return []

        # Build where clause for server-side metadata filtering
        where = _build_where(call_id_filter, topic_filter)

        kwargs = dict(
            query_texts=[query],
            n_results=min(top_k, self._col.count()),
            include=["distances", "metadatas", "documents"],
        )
        if where:
            kwargs["where"] = where

        results = self._col.query(**kwargs)

        hits = []
        for chunk_id, distance, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # ChromaDB cosine returns distance (0=identical, 2=opposite)
            # Convert to similarity score in [0, 1]
            score = 1 - (distance / 2)
            hits.append({"chunk_id": chunk_id, "score": score, "metadata": meta})

        return hits

    def delete_call(self, call_id: str) -> int:
        """Remove all chunks for a given call (useful for re-ingestion)."""
        existing = self._col.get(where={"call_id": call_id})
        ids = existing["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    @property
    def total_chunks(self) -> int:
        return self._col.count()


# ── Where-clause builder ───────────────────────────────────────────────────────

def _build_where(call_id: str = None, topic: str = None) -> dict | None:
    """
    Build ChromaDB $and / single-key where clause from optional filters.
    Returns None if no filters → no where clause sent.
    """
    conditions = []
    if call_id:
        conditions.append({"call_id": {"$eq": call_id}})
    if topic:
        conditions.append({"topic_tag": {"$eq": topic}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}