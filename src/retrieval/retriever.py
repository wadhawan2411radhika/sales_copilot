"""
retrieval/retriever.py
----------------------
Routes queries to the right retrieval strategy:

  1. STRUCTURED  → hits SQLite directly (list calls, action items)
  2. SEMANTIC    → vector search then fetches chunk text from SQLite
  3. FULL-CALL   → fetches all utterances for a given call (for summarisation)

The Retriever is intentionally query-agnostic — the LLM layer decides
which mode to invoke based on intent classification.
"""

from ..storage.db import Database
from ..storage.vector_store import VectorStore


class Retriever:
    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vs = vector_store

    # ── 1. Structured retrieval ────────────────────────────────────────────────

    def list_calls(self) -> list[dict]:
        return self.db.fetchall(
            "SELECT call_id, filename, call_type, ingested_at, num_utterances, num_chunks "
            "FROM calls ORDER BY ingested_at ASC"
        )

    def get_last_call(self) -> dict | None:
        return self.db.fetchone(
            "SELECT * FROM calls ORDER BY ingested_at DESC LIMIT 1"
        )

    def get_call_by_id(self, call_id: str) -> dict | None:
        return self.db.fetchone(
            "SELECT * FROM calls WHERE call_id = ?", (call_id,)
        )

    def get_action_items(self, call_id: str = None) -> list[dict]:
        if call_id:
            return self.db.fetchall(
                "SELECT * FROM action_items WHERE call_id = ? ORDER BY id", (call_id,)
            )
        return self.db.fetchall("SELECT * FROM action_items ORDER BY call_id, id")

    # ── 2. Full-call retrieval (for summarisation) ────────────────────────────

    def get_full_transcript(self, call_id: str) -> list[dict]:
        """Returns all utterances for a call, ordered by time."""
        return self.db.fetchall(
            "SELECT speaker, role, side, timestamp, text "
            "FROM utterances WHERE call_id = ? ORDER BY timestamp_secs",
            (call_id,),
        )

    def get_call_chunks(self, call_id: str) -> list[dict]:
        return self.db.fetchall(
            "SELECT * FROM chunks WHERE call_id = ? ORDER BY chunk_index",
            (call_id,),
        )

    # ── 3. Semantic retrieval ──────────────────────────────────────────────────

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        call_id_filter: str = None,
        topic_filter: str = None,
    ) -> list[dict]:
        """
        Returns top_k chunks with full text + metadata.
        call_id_filter and topic_filter are pushed to ChromaDB's where clause
        — filtered server-side before ranking, not post-hoc.
        """
        hits = self.vs.search(
            query,
            top_k=top_k,
            call_id_filter=call_id_filter,
            topic_filter=topic_filter,
        )

        results = []
        for hit in hits:
            chunk = self.db.fetchone(
                "SELECT * FROM chunks WHERE chunk_id = ?", (hit["chunk_id"],)
            )
            if not chunk:
                continue
            chunk["score"] = hit["score"]
            results.append(chunk)

        return results

    def semantic_search_with_utterances(
        self,
        query: str,
        top_k: int = 5,
        call_id_filter: str = None,
        topic_filter: str = None,
    ) -> list[dict]:
        """
        Same as semantic_search but enriches each chunk with its
        individual utterances (for citation rendering).
        """
        chunks = self.semantic_search(query, top_k, call_id_filter, topic_filter)

        for chunk in chunks:
            utterances = self.db.fetchall(
                """SELECT speaker, role, side, timestamp, text
                   FROM utterances
                   WHERE call_id = ?
                     AND timestamp_secs BETWEEN
                       (SELECT timestamp_secs FROM utterances
                        WHERE call_id = ? AND timestamp = ? LIMIT 1)
                       AND
                       (SELECT timestamp_secs FROM utterances
                        WHERE call_id = ? AND timestamp = ? LIMIT 1)
                   ORDER BY timestamp_secs""",
                (chunk["call_id"],
                 chunk["call_id"], chunk["start_time"],
                 chunk["call_id"], chunk["end_time"]),
            )
            chunk["utterances"] = utterances

        return chunks

    # ── 4. Topic-filtered retrieval (for "negative pricing" queries) ───────────

    def get_chunks_by_topic(self, topic: str, side: str = None) -> list[dict]:
        """
        Fetch all chunks tagged with a topic.
        Optionally filter to only prospect/vendor utterances within them.
        """
        chunks = self.db.fetchall(
            "SELECT * FROM chunks WHERE topic_tag = ? ORDER BY call_id, chunk_index",
            (topic,),
        )
        return chunks