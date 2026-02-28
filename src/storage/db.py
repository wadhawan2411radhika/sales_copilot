"""
storage/db.py
-------------
Thin wrapper around SQLite.
All other modules import this — never open sqlite3 directly.
"""

import sqlite3
import os
from pathlib import Path
from .schema import ALL_TABLES

DB_PATH = Path(os.getenv("DB_PATH", "data/sales_copilot.db"))


class Database:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None

    def connect(self):
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row       # dict-like rows
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        return self

    def _init_schema(self):
        for ddl in ALL_TABLES:
            self._conn.execute(ddl)
        self._conn.commit()

    def execute(self, sql: str, params=()):
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, params_seq):
        return self._conn.executemany(sql, params_seq)

    def commit(self):
        self._conn.commit()

    def fetchall(self, sql: str, params=()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def fetchone(self, sql: str, params=()) -> dict | None:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    def close(self):
        if self._conn:
            self._conn.close()

    # Context manager support
    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.close()