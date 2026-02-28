"""
storage/schema.py
-----------------
SQLite schema for structured metadata.
Keeps two concerns separate:
  - calls          → call-level metadata
  - utterances     → every timestamped line (raw)
  - chunks         → retrieval units (window of utterances)
  - action_items   → explicitly committed todos per call
"""

CREATE_CALLS = """
CREATE TABLE IF NOT EXISTS calls (
    call_id     TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    call_type   TEXT,                  -- demo | pricing | objection | negotiation
    ingested_at TEXT NOT NULL,         -- ISO-8601
    num_utterances INTEGER DEFAULT 0,
    num_chunks     INTEGER DEFAULT 0,
    file_hash   TEXT UNIQUE            -- SHA-256 for idempotent re-ingestion
);
"""

CREATE_UTTERANCES = """
CREATE TABLE IF NOT EXISTS utterances (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id     TEXT NOT NULL,
    speaker     TEXT NOT NULL,
    role        TEXT,                  -- AE | SE | CISO | Prospect | etc.
    side        TEXT,                  -- vendor | prospect
    timestamp   TEXT NOT NULL,         -- e.g. "01:28"
    timestamp_secs INTEGER,            -- for ordering
    text        TEXT NOT NULL,
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);
"""

CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,      -- call_id::chunk_idx
    call_id     TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,         -- concatenated utterances in this chunk
    speaker_turns TEXT,                -- JSON list of speakers in this chunk
    start_time  TEXT,
    end_time    TEXT,
    topic_tag   TEXT,                  -- pricing | security | objection | feature | ...
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);
"""

CREATE_ACTION_ITEMS = """
CREATE TABLE IF NOT EXISTS action_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id     TEXT NOT NULL,
    owner       TEXT,                  -- Jordan | Elena | Priya | etc.
    side        TEXT,                  -- vendor | prospect
    description TEXT NOT NULL,
    deadline    TEXT,                  -- if mentioned
    FOREIGN KEY (call_id) REFERENCES calls(call_id)
);
"""

ALL_TABLES = [CREATE_CALLS, CREATE_UTTERANCES, CREATE_CHUNKS, CREATE_ACTION_ITEMS]