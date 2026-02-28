#!/usr/bin/env python3
"""
cli.py
------
Entry point for the Sales Call Copilot CLI.

Usage:
  python cli.py                    # interactive REPL
  python cli.py "list my call ids" # single command mode
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from src.storage.db import Database
from src.storage.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.ingestion.pipeline import ingest_directory
from src.cli.router import route, Intent
from src.cli.handlers import (
    handle_list_calls, handle_summarise, handle_qa,
    handle_negative_filter, handle_action_items,
    handle_ingest, handle_help, handle_unknown,
)


BANNER = """
╔══════════════════════════════════════════════════════╗
║       📞 Sales Call Copilot  v0.1                    ║
║       Type 'help' for commands · 'exit' to quit      ║
╚══════════════════════════════════════════════════════╝
"""

DISPATCH = {
    Intent.LIST_CALLS:      handle_list_calls,
    Intent.SUMMARISE:       handle_summarise,
    Intent.QA:              handle_qa,
    Intent.NEGATIVE_FILTER: handle_negative_filter,
    Intent.ACTION_ITEMS:    handle_action_items,
    Intent.INGEST:          handle_ingest,
    Intent.HELP:            handle_help,
    Intent.UNKNOWN:         handle_unknown,
}


def run(db: Database, vector_store: VectorStore, query: str = None):
    retriever = Retriever(db, vector_store)
    ctx = dict(retriever=retriever, db=db, vector_store=vector_store)

    def process(q: str):
        parsed = route(q)
        handler = DISPATCH.get(parsed.intent, handle_unknown)
        result = handler(parsed=parsed, **ctx)
        print("\n" + result + "\n")

    if query:
        process(query)
        return

    print(BANNER)
    while True:
        try:
            q = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        process(q)


def bootstrap(db: Database, vector_store: VectorStore):
    """
    Ingest any transcript in data/transcripts/ not yet in the DB.
    Runs on every startup — safe because ingestion is idempotent (SHA-256 dedup).
    This means adding new .txt files to data/transcripts/ and restarting
    the CLI is all you need to ingest them.
    """
    transcript_dir = Path("data/transcripts")
    if not transcript_dir.exists():
        return

    files = sorted(transcript_dir.glob("*.txt"))
    if not files:
        return

    # Check which filenames are already in the DB
    already = {r["filename"] for r in db.fetchall("SELECT filename FROM calls")}
    pending = [f for f in files if f.name not in already]

    if not pending:
        return  # all files already indexed, nothing to do

    print(f"🔄 Ingesting {len(pending)} new transcript(s) from data/transcripts/ ...")
    results = ingest_directory(transcript_dir, db, vector_store)
    for r in results:
        if r["status"] == "success":
            print(f"  ✅ {r['filename']} → {r['call_id']}")
        elif r["status"] == "skipped":
            pass  # already indexed — silent
        else:
            print(f"  ⚠️  {r.get('filename','?')}: {r.get('reason','?')}")

    newly = sum(1 for r in results if r["status"] == "success")
    if newly:
        print(f"  Done. {newly} call(s) indexed.\n")


def main():
    db = Database()
    db.connect()

    vector_store = VectorStore()
    bootstrap(db, vector_store)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    run(db, vector_store, query)
    db.close()


if __name__ == "__main__":
    main()