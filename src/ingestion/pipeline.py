"""
ingestion/pipeline.py
---------------------
Orchestrates the full ingestion flow:
  parse → chunk → store utterances → embed + index chunks → extract metadata

Idempotent: re-ingesting the same file is a no-op (file_hash dedup).
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .parser import parse_transcript, ParsedCall
from .chunker import chunk_utterances
from ..storage.db import Database
from ..storage.vector_store import VectorStore


# ── Action item extraction (keyword heuristic) ─────────────────────────────────

ACTION_VERBS = re.compile(
    r"\b(will send|to follow|by eod|by july|schedule|share|loop|capture|"
    r"attach|send over|include|prepare|commit|provide|draft|update|fix|resend)\b",
    re.IGNORECASE,
)

def _extract_action_items(call_id: str, utterances) -> list[dict]:
    items = []
    for utt in utterances:
        if ACTION_VERBS.search(utt.text):
            items.append({
                "call_id": call_id,
                "owner": utt.speaker,
                "side": utt.side,
                "description": utt.text[:300],
                "deadline": None,
            })
    return items


# ── Call type inference (can also be enriched by LLM later) ───────────────────

CALL_TYPE_SIGNALS = {
    "demo":        ["demo", "show", "dashboard", "feature", "copilot"],
    "pricing":     ["price", "sku", "discount", "₹", "tcv", "quote"],
    "objection":   ["security", "legal", "gdpr", "dpa", "soc 2", "concern"],
    "negotiation": ["negotiation", "docusign", "sign", "closed", "final"],
}

def _infer_call_type_from_content(text: str, filename: str) -> str:
    lower_name = filename.lower()
    for k in CALL_TYPE_SIGNALS:
        if k in lower_name:
            return k
    lower_text = text[:500].lower()
    for call_type, signals in CALL_TYPE_SIGNALS.items():
        if any(s in lower_text for s in signals):
            return call_type
    return "unknown"


# ── Main pipeline function ─────────────────────────────────────────────────────

def ingest_transcript(
    path: str | Path,
    db: Database,
    vector_store: VectorStore,
    call_id_override: str = None,
) -> dict:
    """
    Ingest a single transcript file.
    Returns a summary dict of what was ingested.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")

    # ── 1. Dedup check ─────────────────────────────────────────────────────────
    raw_bytes = path.read_bytes()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()

    existing = db.fetchone("SELECT call_id FROM calls WHERE file_hash = ?", (file_hash,))
    if existing:
        return {
            "status": "skipped",
            "reason": "already_ingested",
            "call_id": existing["call_id"],
            "filename": path.name,
        }

    # ── 2. Parse ───────────────────────────────────────────────────────────────
    parsed: ParsedCall = parse_transcript(path)

    if not parsed.utterances:
        return {"status": "error", "reason": "no_utterances_parsed", "filename": path.name}

    # ── 3. Assign call_id ──────────────────────────────────────────────────────
    if call_id_override:
        call_id = call_id_override
    else:
        # Auto-assign: call_001, call_002, ...
        count = db.fetchone("SELECT COUNT(*) as n FROM calls")["n"]
        call_id = f"call_{(count + 1):03d}"

    # ── 4. Infer call type ─────────────────────────────────────────────────────
    full_text = " ".join(u.text for u in parsed.utterances)
    call_type = _infer_call_type_from_content(full_text, path.name)

    # ── 5. Chunk ───────────────────────────────────────────────────────────────
    chunks = chunk_utterances(parsed.utterances, call_id)

    # ── 6. Write to SQLite ─────────────────────────────────────────────────────
    db.execute(
        """INSERT INTO calls (call_id, filename, call_type, ingested_at, 
                              num_utterances, num_chunks, file_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (call_id, path.name, call_type,
         datetime.now(timezone.utc).isoformat(),
         len(parsed.utterances), len(chunks), file_hash),
    )

    db.executemany(
        """INSERT INTO utterances 
           (call_id, speaker, role, side, timestamp, timestamp_secs, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (call_id, u.speaker, u.role, u.side, u.timestamp, u.timestamp_secs, u.text)
            for u in parsed.utterances
        ],
    )

    db.executemany(
        """INSERT INTO chunks 
           (chunk_id, call_id, chunk_index, text, speaker_turns, start_time, end_time, topic_tag)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (c.chunk_id, c.call_id, c.chunk_index, c.text,
             c.speaker_turns, c.start_time, c.end_time, c.topic_tag)
            for c in chunks
        ],
    )

    # ── 7. Extract & store action items ───────────────────────────────────────
    action_items = _extract_action_items(call_id, parsed.utterances)
    if action_items:
        db.executemany(
            """INSERT INTO action_items (call_id, owner, side, description, deadline)
               VALUES (?, ?, ?, ?, ?)""",
            [(a["call_id"], a["owner"], a["side"], a["description"], a["deadline"])
             for a in action_items],
        )

    db.commit()

    # ── 8. Embed + index chunks ────────────────────────────────────────────────
    chunk_ids   = [c.chunk_id for c in chunks]
    chunk_texts = [c.text for c in chunks]
    metadatas   = [
        {"call_id": c.call_id, "topic_tag": c.topic_tag,
         "start_time": c.start_time, "end_time": c.end_time}
        for c in chunks
    ]
    indexed_count = vector_store.add_chunks(chunk_ids, chunk_texts, metadatas)

    return {
        "status": "success",
        "call_id": call_id,
        "filename": path.name,
        "call_type": call_type,
        "utterances": len(parsed.utterances),
        "chunks": len(chunks),
        "chunks_indexed": indexed_count,
        "action_items_found": len(action_items),
    }


def ingest_directory(
    directory: str | Path,
    db: Database,
    vector_store: VectorStore,
    pattern: str = "*.txt",
) -> list[dict]:
    """Ingest all matching transcripts in a directory, sorted by filename."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    return [ingest_transcript(f, db, vector_store) for f in files]