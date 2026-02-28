"""
ingestion/parser.py
-------------------
Parses raw transcript .txt files into structured Utterance objects.

Format assumed:
  [MM:SS] Speaker (Role):  Text...
  [MM:SS] Speaker:  Text...          ← role optional

Also handles stage directions like:
  *screen share: ROI.xlsx*
  Audio plays: "..."
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class Utterance:
    speaker: str
    role: str            # e.g. "RevOps Director", "AE", "SE", "CISO"
    side: str            # "vendor" | "prospect" | "unknown"
    timestamp: str       # "01:28"
    timestamp_secs: int  # 88
    text: str


@dataclass
class ParsedCall:
    filename: str
    call_type: str       # inferred from filename or content
    utterances: list[Utterance] = field(default_factory=list)


# ── Constants ──────────────────────────────────────────────────────────────────

# Known vendor-side speaker names/roles (extend as needed)
VENDOR_INDICATORS = {
    "ae", "se", "ciso", "maya", "luis", "elena", "jordan",
    "asha", "pricing strategist", "sales engineer"
}

CALL_TYPE_MAP = {
    "demo": "demo",
    "pricing": "pricing",
    "objection": "objection",
    "negotiation": "negotiation",
}

# Regex: [MM:SS] or [HH:MM:SS]
UTTERANCE_RE = re.compile(
    r"^\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s+"   # timestamp group
    r"([^\(:\n]+?)"                            # speaker name
    r"(?:\s*\(([^)]+)\))?"                    # optional (Role)
    r"\s*:\s*(.+)$",                           # colon + text
    re.DOTALL,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_secs(ts: str) -> int:
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def _infer_side(speaker: str, role: str) -> str:
    combined = (speaker + " " + (role or "")).lower()
    if any(v in combined for v in VENDOR_INDICATORS):
        return "vendor"
    # "Prospect" label explicitly used in these transcripts
    if "prospect" in combined:
        return "prospect"
    return "unknown"


def _clean_text(text: str) -> str:
    """Strip stage directions like *plays clip* from text."""
    text = re.sub(r"\*[^*]+\*", "", text)   # *action*
    text = text.strip()
    return text


def _infer_call_type(filename: str) -> str:
    lower = filename.lower()
    for key, val in CALL_TYPE_MAP.items():
        if key in lower:
            return val
    return "unknown"


# ── Main parser ────────────────────────────────────────────────────────────────

def parse_transcript(path: str | Path) -> ParsedCall:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    call_type = _infer_call_type(path.name)
    parsed = ParsedCall(filename=path.name, call_type=call_type)

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        m = UTTERANCE_RE.match(line)
        if not m:
            # Could be a continuation line or stage direction — skip for now
            continue

        ts, speaker, role, text = m.group(1), m.group(2).strip(), m.group(3), m.group(4)
        role = role.strip() if role else ""
        text = _clean_text(text)

        if not text:
            continue

        utt = Utterance(
            speaker=speaker,
            role=role,
            side=_infer_side(speaker, role),
            timestamp=ts,
            timestamp_secs=_to_secs(ts),
            text=text,
        )
        parsed.utterances.append(utt)

    return parsed