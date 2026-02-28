# 📞 Sales Call Copilot

A CLI chatbot that helps sales teams query, summarise, and extract insights from call transcripts using RAG (Retrieval-Augmented Generation).

---

## Quick Start

```bash
# 1. Clone and enter project
git clone <your-repo-url>
cd sales-copilot

# 2. Copy env file and add your OpenAI key
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-...

# 3. Install dependencies
pip install -r requirements.txt

# 4. Drop transcript .txt files into data/transcripts/

# 5. Run (first run auto-ingests all transcripts)
python cli.py

# Or single-shot mode:
python cli.py "list my call ids"
python cli.py "summarise the last call"
```

---

## Example Commands

```
list my call ids
summarise the last call
summarise call_003
what objections did the prospect raise in the pricing call?
give me all negative comments when pricing was mentioned
what are the action items from call_002?
ingest a new call from ./data/transcripts/5_followup.txt
help
```

---

## Architecture

```
sales-copilot/
├── cli.py                      # Entry point (REPL + single-shot)
├── src/
│   ├── ingestion/
│   │   ├── parser.py           # Raw .txt → structured Utterance objects
│   │   ├── chunker.py          # Utterances → retrieval chunks (sliding window)
│   │   └── pipeline.py         # Orchestrates: parse → chunk → embed → store
│   ├── storage/
│   │   ├── schema.py           # SQLite DDL (calls, utterances, chunks, action_items)
│   │   ├── db.py               # SQLite wrapper
│   │   └── vector_store.py     # Pluggable vector index (numpy cosine similarity)
│   ├── retrieval/
│   │   └── retriever.py        # Routes queries: structured | semantic | full-call
│   ├── llm/
│   │   └── client.py           # LLM wrapper + all prompt templates
│   └── cli/
│       ├── router.py           # Regex intent classifier
│       └── handlers.py         # One handler per intent
├── tests/
│   └── test_ingestion.py       # Parser + chunker unit tests (no API needed)
├── data/
│   ├── transcripts/            # Raw .txt transcript files
│   ├── sales_copilot.db        # SQLite database (auto-created)
│   └── vectors/                # Vector index (auto-created)
└── requirements.txt
```

---

## Design Decisions

### Storage: Dual-layer (SQLite + Vector Index)

| Layer | What | Why |
|-------|------|-----|
| SQLite | Calls, utterances, chunks, action items | Structured queries (`list calls`, `action items`), fast metadata filtering, zero infra |
| Vector index (numpy) | TF-IDF/embedding matrix + chunk_id map | Semantic similarity search across all calls |

**Why not a single vector DB?**  
Queries like `list my call ids` or `action items from call_002` are pure SQL — putting them through a vector search is wasteful and less accurate.

### Chunking: Speaker-turn sliding window

- Window of 6 utterances, overlap of 2
- Keeps Q&A exchanges intact (a prospect objection + vendor rebuttal in the same chunk)
- Topic tag inferred at chunk level via keyword scoring (no LLM needed at ingest time)

### Vector Backend: Auto-selected

Priority: `OpenAI embeddings` → `sentence-transformers` → `TF-IDF (sklearn)`

The TF-IDF fallback works offline with no API key — useful for testing and cost-free development. Switch to OpenAI for production-quality semantic matching.

### Intent Routing: Regex-first

Pattern matching handles all 4 required intents without LLM calls:
- `LIST_CALLS` → structured SQL
- `SUMMARISE` → full-transcript LLM call (map-reduce for long calls)
- `NEGATIVE_FILTER` → hybrid: topic-filter SQL + semantic search + LLM sentiment filter
- `INGEST` → ingestion pipeline

Fallback: general Q&A via semantic search + citation-aware LLM answer.

### Idempotent Ingestion

Every file is SHA-256 hashed before ingestion. Re-running on the same file is a no-op — safe to restart.

---

## Retrieval Strategy per Query Type

| Query | Strategy |
|-------|----------|
| `list my call ids` | SQL: `SELECT * FROM calls` |
| `summarise the last call` | Full transcript → LLM summary |
| `negative pricing comments` | Topic filter (SQL) + semantic search → LLM sentiment extraction |
| `ingest from <path>` | Pipeline: parse → chunk → embed → upsert |
| `what objections were raised?` | Semantic search → LLM answer with citations |
| `action items from call_002` | SQL: `SELECT * FROM action_items WHERE call_id = ?` |

---

## Assumptions

1. **Transcript format**: `[MM:SS] Speaker (Role): text` — standard for Zoom/Gong-style exports.
2. **Speaker sides**: Vendor speakers identified by name/role keywords (`AE`, `SE`, `CISO`, `Maya`, etc.). All others treated as prospect or unknown.
3. **Call ordering**: Files sorted alphabetically for `call_id` assignment — prefix with `1_`, `2_` etc. to control order.
4. **LLM model**: Defaults to `gpt-4o-mini` (cheap, fast). Set `CHAT_MODEL=gpt-4o` in `.env` for higher quality.
5. **Action items**: Extracted via keyword heuristics at ingest time. Accuracy improves with LLM extraction (future enhancement).
6. **Multi-language**: Parser handles Hindi-English mixed transcripts as plain text. Semantic search quality on non-English segments improves with multilingual embedding models.

---

## Running Tests

```bash
# No pytest required — runs with plain Python
python3 -c "
import sys; sys.path.insert(0, '.')
from tests.test_ingestion import *
for t in [test_parse_utterance_count, test_timestamp_to_secs,
          test_infer_side, test_chunk_count, test_topic_inference]:
    t(); print(f'✅ {t.__name__}')
"
```

---

## Future Enhancements (Incremental Roadmap)

| Phase | Enhancement |
|-------|-------------|
| v0.2 | LLM-based action item extraction (replace keyword heuristic) |
| v0.2 | Sentence-level sentiment tagging at ingest time |
| v0.3 | Cross-call "deal timeline" view |
| v0.3 | Named entity extraction: prices, dates, competitor names |
| v0.4 | LLM intent classification fallback for complex queries |
| v0.4 | Streaming responses for long summaries |
| v1.0 | REST API + web UI |
| v1.0 | FAISS or ChromaDB swap-in for production scale |