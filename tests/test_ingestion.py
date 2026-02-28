"""
tests/test_ingestion.py
-----------------------
Tests for parsing and chunking logic (no LLM or API calls needed).
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.parser import parse_transcript, _to_secs, _infer_side
from src.ingestion.chunker import chunk_utterances, _infer_topic


# ── Sample transcript for testing ─────────────────────────────────────────────

SAMPLE_TRANSCRIPT = """[00:00] AE (Jordan):  Good morning, Priya!  How's the quarter?
[00:05] Prospect (Priya – RevOps Director):  Busy—pipeline is healthy.
[01:28] SE (Luis):  We embed with OpenAI's text-embedding-3-small, store in FAISS.
[02:30] Prospect (Priya):  How accurate are the speaker diarization segments?
[04:41] AE:  List price is ₹1 800 per user per month, billed annually.
[05:03] Prospect:  I'll need that in writing.
"""


# ── Parser tests ───────────────────────────────────────────────────────────────

def test_parse_utterance_count():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix="_demo_call.txt", delete=False) as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp = f.name
    try:
        parsed = parse_transcript(tmp)
        assert len(parsed.utterances) == 6
    finally:
        os.unlink(tmp)


def test_parse_speaker_sides():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix="_demo_call.txt", delete=False) as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp = f.name
    try:
        parsed = parse_transcript(tmp)
        sides = {u.speaker: u.side for u in parsed.utterances}
        assert sides.get("AE") == "vendor"
        # Prospect label → prospect
        prospect_utts = [u for u in parsed.utterances if "prospect" in u.side or "Prospect" in u.speaker]
        assert len(prospect_utts) >= 2
    finally:
        os.unlink(tmp)


def test_parse_call_type():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix="1_demo_call.txt", delete=False) as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp = f.name
    try:
        parsed = parse_transcript(tmp)
        assert parsed.call_type == "demo"
    finally:
        os.unlink(tmp)


def test_timestamp_to_secs():
    assert _to_secs("01:28") == 88
    assert _to_secs("00:00") == 0
    assert _to_secs("10:05") == 605


def test_infer_side():
    assert _infer_side("AE", "Jordan") == "vendor"
    assert _infer_side("SE", "Luis") == "vendor"
    assert _infer_side("Prospect", "Priya – RevOps Director") == "prospect"


# ── Chunker tests ──────────────────────────────────────────────────────────────

def test_chunk_count():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix="_demo_call.txt", delete=False) as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp = f.name
    try:
        parsed = parse_transcript(tmp)
        chunks = chunk_utterances(parsed.utterances, "call_001", window=4, overlap=1)
        assert len(chunks) >= 1
        # All chunks have chunk_id
        assert all("::" in c.chunk_id for c in chunks)
    finally:
        os.unlink(tmp)


def test_chunk_overlap():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix="_demo_call.txt", delete=False) as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp = f.name
    try:
        parsed = parse_transcript(tmp)
        chunks = chunk_utterances(parsed.utterances, "call_001", window=4, overlap=2)
        if len(chunks) > 1:
            # Overlapping chunks share utterances
            last_of_first = chunks[0].utterances[-2:]
            first_of_second = chunks[1].utterances[:2]
            assert last_of_first[0].timestamp == first_of_second[0].timestamp
    finally:
        os.unlink(tmp)


def test_topic_inference():
    assert _infer_topic("price ₹1800 per user discount SKU") == "pricing"
    assert _infer_topic("SOC 2 Type II encryption GDPR DPDPA") == "security"
    assert _infer_topic("governing law liability indemnity arbitration") == "legal"
    assert _infer_topic("diarization summary Copilot Slack CRM") == "feature"


def test_chunk_text_contains_timestamps():
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix="_demo_call.txt", delete=False) as f:
        f.write(SAMPLE_TRANSCRIPT)
        tmp = f.name
    try:
        parsed = parse_transcript(tmp)
        chunks = chunk_utterances(parsed.utterances, "call_001")
        for chunk in chunks:
            assert "[" in chunk.text   # timestamps present in chunk text
    finally:
        os.unlink(tmp)