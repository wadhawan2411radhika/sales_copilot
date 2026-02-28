"""
cli/router.py
-------------
Classifies user input into one of these intents and dispatches:

  LIST_CALLS       → "list my call ids", "show calls"
  SUMMARISE        → "summarise call 3", "summarise the last call"
  QA               → "what objections did they raise?"
  NEGATIVE_FILTER  → "negative comments when pricing was mentioned"
  INGEST           → "ingest a new call from ./path"
  ACTION_ITEMS     → "what are the action items from call 2?"
  HELP             → "help", "?"
  UNKNOWN          → fallback

Pattern matching first (fast, no LLM cost).
Falls back to semantic classification only if needed (future).
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class Intent(Enum):
    LIST_CALLS      = auto()
    SUMMARISE       = auto()
    QA              = auto()
    NEGATIVE_FILTER = auto()
    INGEST          = auto()
    ACTION_ITEMS    = auto()
    HELP            = auto()
    UNKNOWN         = auto()


@dataclass
class ParsedIntent:
    intent: Intent
    call_id: str | None = None       # e.g. "call_003"
    call_ref: str | None = None      # e.g. "last", "3", "the pricing call"
    ingest_path: str | None = None
    topic: str | None = None
    raw_query: str = ""


# ── Patterns ───────────────────────────────────────────────────────────────────

_LIST_RE = re.compile(
    r"\b(list|show|display)\b.*(call|transcript|recording)", re.I
)
_SUMMARISE_RE = re.compile(
    r"\b(summaris[e|ed]?|summarize|summary|recap|overview)\b", re.I
)
_NEGATIVE_RE = re.compile(
    r"\b(negative|bad|concern|objection|complaint|problem|issue|reject|refuse)\b", re.I
)
_INGEST_RE = re.compile(
    r"\b(ingest|upload|add|load|import)\b.*\b(call|transcript|from)\b", re.I
)
_ACTION_RE = re.compile(
    r"\b(action items?|todo|task|commit|next steps?|follow.?up)\b", re.I
)
_HELP_RE = re.compile(r"^\s*(help|\?|commands|what can you do)\s*$", re.I)

# Patterns for extracting call references
_LAST_RE = re.compile(r"\b(last|latest|most recent|final)\b", re.I)
_CALL_NUM_RE = re.compile(r"\bcall[_\s#-]?(\d+)\b", re.I)
_CALL_ID_RE = re.compile(r"\b(call_\d{3})\b", re.I)
_CALL_TYPE_RE = re.compile(
    r"\b(demo|pricing|objection|negotiation|security|legal)\b", re.I
)

# Path extraction for ingest
_PATH_RE = re.compile(r"(?:from\s+)([./~\w\-]+\.txt)", re.I)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_call_ref(text: str) -> tuple[str | None, str | None]:
    """Returns (call_id_or_none, human_ref_or_none)"""
    m = _CALL_ID_RE.search(text)
    if m:
        return m.group(1), m.group(1)
    m = _LAST_RE.search(text)
    if m:
        return None, "last"
    m = _CALL_NUM_RE.search(text)
    if m:
        return f"call_{int(m.group(1)):03d}", m.group(1)
    m = _CALL_TYPE_RE.search(text)
    if m:
        return None, m.group(1).lower()
    return None, None


# ── Main router ────────────────────────────────────────────────────────────────

def route(query: str) -> ParsedIntent:
    q = query.strip()

    if _HELP_RE.match(q):
        return ParsedIntent(Intent.HELP, raw_query=q)

    if _LIST_RE.search(q):
        return ParsedIntent(Intent.LIST_CALLS, raw_query=q)

    if _INGEST_RE.search(q):
        path_match = _PATH_RE.search(q)
        ingest_path = path_match.group(1) if path_match else None
        return ParsedIntent(Intent.INGEST, ingest_path=ingest_path, raw_query=q)

    if _ACTION_RE.search(q):
        call_id, call_ref = _extract_call_ref(q)
        return ParsedIntent(Intent.ACTION_ITEMS, call_id=call_id,
                            call_ref=call_ref, raw_query=q)

    if _SUMMARISE_RE.search(q):
        call_id, call_ref = _extract_call_ref(q)
        return ParsedIntent(Intent.SUMMARISE, call_id=call_id,
                            call_ref=call_ref, raw_query=q)

    if _NEGATIVE_RE.search(q):
        # Extract topic (what topic were they negative about?)
        topic_m = re.search(
            r"\b(pric|discount|security|feature|legal|competi|sla|contract)\w*", q, re.I
        )
        topic = topic_m.group(0).lower() if topic_m else None
        return ParsedIntent(Intent.NEGATIVE_FILTER, topic=topic, raw_query=q)

    # Default: general Q&A
    call_id, call_ref = _extract_call_ref(q)
    return ParsedIntent(Intent.QA, call_id=call_id, call_ref=call_ref, raw_query=q)