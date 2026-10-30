"""SQLite persistence with an FTS5 (BM25) index over transcript chunks."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .chunking import Chunk
from .schemas import MeetingAnalysis
from .transcript import Segment

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    duration REAL NOT NULL,
    status TEXT NOT NULL,          -- processing | ready | failed
    error TEXT,
    analysis TEXT                  -- MeetingAnalysis JSON
);
CREATE TABLE IF NOT EXISTS segments (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    PRIMARY KEY (meeting_id, idx)
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""

STOPWORDS = frozenset(
    """a an and are as at be but by did do does for from had has have how i if in is it its
    me my of on or our so that the their them there they this to was we were what when where
    which who why will with you your about can could should would any all""".split()
)


@dataclass
class SearchHit:
    meeting_id: str
    meeting_title: str
    chunk_index: int
    start: float
    end: float
    text: str
    score: float


def fts_query(question: str) -> str:
    """Turn free text into a safe FTS5 OR-query of its content words."""
    terms = [t for t in re.findall(r"[\w']+", question.lower()) if t not in STOPWORDS]
    terms = [t.strip("'") for t in terms if len(t.strip("'")) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


class Store:
    def __init__(self, path: Path | str):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    # -- meetings -----------------------------------------------------------

    def create_meeting(
        self, title: str, source_name: str, segments: list[Segment], chunks: list[Chunk]
    ) -> str:
        meeting_id = uuid.uuid4().hex[:12]
        duration = max((s.end for s in segments), default=0.0)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO meetings VALUES (?, ?, ?, ?, ?, 'processing', NULL, NULL)",
                (meeting_id, title, source_name, datetime.now(UTC).isoformat(), duration),
            )
            self._conn.executemany(
                "INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (meeting_id, i, s.speaker, s.start, s.end, s.text)
                    for i, s in enumerate(segments)
                ],
            )
            self._conn.executemany(
                "INSERT INTO chunks (meeting_id, idx, start, end, text) VALUES (?, ?, ?, ?, ?)",
                [(meeting_id, c.index, c.start, c.end, c.text) for c in chunks],
            )
        return meeting_id

    def set_analysis(self, meeting_id: str, analysis: MeetingAnalysis) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE meetings SET status='ready', error=NULL, analysis=?, title=? WHERE id=?",
                (analysis.model_dump_json(), analysis.title, meeting_id),
            )

    def set_status(self, meeting_id: str, status: str, error: str | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE meetings SET status=?, error=? WHERE id=?", (status, error, meeting_id)
            )

    def list_meetings(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, title, source_name, created_at, duration, status, error, analysis "
            "FROM meetings ORDER BY created_at DESC"
        ).fetchall()
        return [self._meeting_dict(r) for r in rows]

    def get_meeting(self, meeting_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone()
        return self._meeting_dict(row) if row else None

    def get_segments(self, meeting_id: str) -> list[Segment]:
        rows = self._conn.execute(
            "SELECT speaker, start, end, text FROM segments WHERE meeting_id=? ORDER BY idx",
            (meeting_id,),
        ).fetchall()
        return [Segment(r["speaker"], r["start"], r["end"], r["text"]) for r in rows]

    def delete_meeting(self, meeting_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
        return cur.rowcount > 0

    @staticmethod
    def _meeting_dict(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["analysis"] = json.loads(data["analysis"]) if data["analysis"] else None
        return data

    # -- retrieval ----------------------------------------------------------

    def search(self, question: str, k: int = 8, meeting_id: str | None = None) -> list[SearchHit]:
        query = fts_query(question)
        if not query:
            return []
        sql = (
            "SELECT c.meeting_id, m.title, c.idx, c.start, c.end, c.text, "
            "bm25(chunks_fts) AS score FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "JOIN meetings m ON m.id = c.meeting_id "
            "WHERE chunks_fts MATCH ?"
        )
        params: list = [query]
        if meeting_id:
            sql += " AND c.meeting_id = ?"
            params.append(meeting_id)
        sql += " ORDER BY score LIMIT ?"
        params.append(k)
        rows = self._conn.execute(sql, params).fetchall()
        # bm25() is lower-is-better; flip the sign so callers can sort descending.
        return [SearchHit(r[0], r[1], r[2], r[3], r[4], r[5], -r[6]) for r in rows]
