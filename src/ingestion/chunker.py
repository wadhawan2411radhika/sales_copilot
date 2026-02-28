"""
ingestion/chunker.py
--------------------
Groups utterances into retrieval chunks.

Strategy: Speaker-turn aware windowing.
- We do NOT chunk purely by token count (that splits Q&A pairs).
- Instead: group N consecutive utterances into a chunk, with overlap.
- Each chunk preserves its speaker context.

Why this beats naive 256-token splits:
  A prospect's objection + the AE's rebuttal must stay together.
  Splitting them means retrieving half an exchange.
"""

import json
from dataclasses import dataclass, field
from .parser import Utterance

# ── Keyword → topic mapping (simple, no LLM needed at index time) ───────────

TOPIC_KEYWORDS = {
    "pricing":    ["price", "pricing", "discount", "₹", "cost", "sku", "seat", "tier",
                   "list", "quote", "tcv", "arR", "invoice", "payment", "budget"],
    "security":   ["soc", "iso", "encrypt", "kms", "gdpr", "dpdpa", "pii", "redact",
                   "audit", "pen test", "pentest", "byok", "cmk", "residency", "vpc"],
    "legal":      ["msa", "governing law", "liability", "indemnity", "arbitration",
                   "clawback", "clause", "dpa", "mfn", "opt-out", "termination"],
    "objection":  ["concern", "worried", "competitor", "compare", "instead", "however",
                   "but ", "can you match", "why is", "challenge", "not sure"],
    "feature":    ["diarization", "summary", "copilot", "dashboard", "slack", "crm",
                   "salesforce", "onboarding", "coaching", "multilingual", "hindi",
                   "action item", "next step", "battle-card"],
    "action_item":["action item", "will send", "to follow", "by eod", "by july",
                   "schedule", "loop legal", "let's capture", "send over"],
    "sla":        ["sla", "uptime", "credit", "99.9", "penalty", "slip", "guarantee",
                   "deadline", "ga ", "early access", "ea "],
}


@dataclass
class Chunk:
    chunk_id: str          # "call_001::0"
    call_id: str
    chunk_index: int
    utterances: list[Utterance]
    topic_tag: str = "general"

    @property
    def text(self) -> str:
        lines = []
        for u in self.utterances:
            role_str = f" ({u.role})" if u.role else ""
            lines.append(f"[{u.timestamp}] {u.speaker}{role_str}: {u.text}")
        return "\n".join(lines)

    @property
    def speaker_turns(self) -> str:
        seen = []
        for u in self.utterances:
            if not seen or seen[-1] != u.speaker:
                seen.append(u.speaker)
        return json.dumps(seen)

    @property
    def start_time(self) -> str:
        return self.utterances[0].timestamp if self.utterances else ""

    @property
    def end_time(self) -> str:
        return self.utterances[-1].timestamp if self.utterances else ""


# ── Topic inference ────────────────────────────────────────────────────────────

def _infer_topic(text: str) -> str:
    lower = text.lower()
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[topic] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Main chunker ───────────────────────────────────────────────────────────────

def chunk_utterances(
    utterances: list[Utterance],
    call_id: str,
    window: int = 6,      # utterances per chunk
    overlap: int = 2,     # overlap between consecutive chunks
) -> list[Chunk]:
    """
    Sliding window chunker with overlap.
    window=6, overlap=2 → each chunk is ~6 utterances,
    and consecutive chunks share 2 utterances for context continuity.
    """
    if not utterances:
        return []

    chunks = []
    step = window - overlap
    idx = 0

    while idx < len(utterances):
        window_utts = utterances[idx: idx + window]
        chunk_index = len(chunks)
        chunk_id = f"{call_id}::{chunk_index}"
        combined_text = " ".join(u.text for u in window_utts)
        topic = _infer_topic(combined_text)

        chunks.append(Chunk(
            chunk_id=chunk_id,
            call_id=call_id,
            chunk_index=chunk_index,
            utterances=window_utts,
            topic_tag=topic,
        ))
        idx += step

    return chunks