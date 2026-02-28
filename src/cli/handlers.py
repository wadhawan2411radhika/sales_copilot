"""
cli/handlers.py
---------------
One handler function per Intent. Each handler:
  1. Calls the retriever (structured or semantic)
  2. Calls the LLM (if needed)
  3. Returns a formatted string for the CLI to print
"""

from pathlib import Path

from ..retrieval.retriever import Retriever
from ..ingestion.pipeline import ingest_transcript
from ..storage.vector_store import VectorStore
from ..llm import client as llm
from .router import ParsedIntent


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_calls_table(calls: list[dict]) -> str:
    if not calls:
        return "No calls ingested yet. Use: ingest a new call from <path>"
    lines = [f"{'Call ID':<12} {'Type':<14} {'Utterances':<12} {'File'}"]
    lines.append("-" * 60)
    for c in calls:
        lines.append(
            f"{c['call_id']:<12} {(c['call_type'] or '?'):<14} "
            f"{c['num_utterances']:<12} {c['filename']}"
        )
    return "\n".join(lines)


def _transcript_to_text(utterances: list[dict]) -> str:
    lines = []
    for u in utterances:
        role = f" ({u['role']})" if u.get("role") else ""
        lines.append(f"[{u['timestamp']}] {u['speaker']}{role}: {u['text']}")
    return "\n".join(lines)


def _resolve_call_id(parsed: ParsedIntent, retriever: Retriever) -> tuple[str | None, str | None]:
    """Resolve a call reference to a call_id. Returns (call_id, error_message)."""
    if parsed.call_id:
        call = retriever.get_call_by_id(parsed.call_id)
        if not call:
            return None, f"Call '{parsed.call_id}' not found."
        return parsed.call_id, None

    if parsed.call_ref == "last":
        call = retriever.get_last_call()
        if not call:
            return None, "No calls found."
        return call["call_id"], None

    if parsed.call_ref:
        # Try to match by type
        calls = retriever.list_calls()
        for c in calls:
            if parsed.call_ref in (c.get("call_type") or ""):
                return c["call_id"], None

    return None, None   # no specific call, will search across all


# ── Handlers ───────────────────────────────────────────────────────────────────

def handle_list_calls(parsed: ParsedIntent, retriever: Retriever, **_) -> str:
    calls = retriever.list_calls()
    return _fmt_calls_table(calls)


def handle_summarise(parsed: ParsedIntent, retriever: Retriever, **_) -> str:
    call_id, error = _resolve_call_id(parsed, retriever)
    if error:
        return f"❌ {error}"
    if not call_id:
        return "❌ Please specify which call to summarise, e.g. 'summarise call_003' or 'summarise the last call'."

    call_meta = retriever.get_call_by_id(call_id)
    utterances = retriever.get_full_transcript(call_id)
    if not utterances:
        return f"❌ No utterances found for {call_id}."

    print(f"  [Summarising {call_id} ({call_meta['call_type']} call, "
          f"{len(utterances)} utterances)...]")

    transcript_text = _transcript_to_text(utterances)
    return llm.summarise_call(call_id, transcript_text)


def handle_qa(parsed: ParsedIntent, retriever: Retriever, **_) -> str:
    call_id, error = _resolve_call_id(parsed, retriever)
    if error:
        return f"❌ {error}"

    print(f"  [Searching {'call ' + call_id if call_id else 'all calls'}...]")

    chunks = retriever.semantic_search(
        query=parsed.raw_query,
        top_k=5,
        call_id_filter=call_id,
    )
    if not chunks:
        return "No relevant transcript segments found."

    return llm.answer_with_citations(parsed.raw_query, chunks)


def handle_negative_filter(parsed: ParsedIntent, retriever: Retriever, **_) -> str:
    query = parsed.raw_query
    topic = parsed.topic or "pricing"

    print(f"  [Searching for negative sentiment on topic: '{topic}' across all calls...]")

    # Combine: topic-filtered chunks + semantic search
    topic_chunks = retriever.get_chunks_by_topic("pricing")
    semantic_chunks = retriever.semantic_search(query, top_k=8)

    # Merge and deduplicate by chunk_id
    seen = set()
    all_chunks = []
    for c in (topic_chunks + semantic_chunks):
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            all_chunks.append(c)

    if not all_chunks:
        return "No relevant segments found."

    return llm.filter_negative_sentiment(query, all_chunks)


def handle_action_items(parsed: ParsedIntent, retriever: Retriever, **_) -> str:
    call_id, error = _resolve_call_id(parsed, retriever)
    if error:
        return f"❌ {error}"

    items = retriever.get_action_items(call_id)
    if not items:
        scope = f"call {call_id}" if call_id else "any call"
        return f"No action items found for {scope}."

    lines = []
    current_call = None
    for item in items:
        if item["call_id"] != current_call:
            current_call = item["call_id"]
            lines.append(f"\n📋 {current_call}")
            lines.append("-" * 40)
        side_icon = "🏢" if item["side"] == "vendor" else "👤"
        lines.append(f"  {side_icon} [{item['owner']}] {item['description'][:120]}")

    return "\n".join(lines)


def handle_ingest(
    parsed: ParsedIntent,
    retriever: Retriever,
    db,
    vector_store: VectorStore,
    **_,
) -> str:
    if not parsed.ingest_path:
        return (
            "❌ Please provide a file path. Example:\n"
            "   ingest a new call from ./data/transcripts/5_followup.txt"
        )

    path = Path(parsed.ingest_path)
    print(f"  [Ingesting {path}...]")
    result = ingest_transcript(path, db, vector_store)
    return llm.summarise_ingestion(result)


def handle_help(**_) -> str:
    return """
╔══════════════════════════════════════════════════════╗
║          Sales Call Copilot — Command Reference      ║
╠══════════════════════════════════════════════════════╣
║  list my call ids                                    ║
║  summarise the last call                             ║
║  summarise call_003                                  ║
║  what objections did the prospect raise?             ║
║  give me all negative comments about pricing         ║
║  what are the action items from call_002?            ║
║  ingest a new call from ./path/to/transcript.txt     ║
║  help                                                ║
║  exit / quit                                         ║
╚══════════════════════════════════════════════════════╝
"""


def handle_unknown(parsed: ParsedIntent, **_) -> str:
    return (
        f"❓ I didn't understand: \"{parsed.raw_query}\"\n"
        "   Try 'help' to see available commands."
    )