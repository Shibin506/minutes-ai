""" "What changed?": follow decisions across meetings and flag the ones that moved or flipped.

The model sees every meeting's decisions and key points in date order, each with an id
like `M3-D1` (meeting 3, decision 1). It groups them into threads (one per topic) and must cite
those ids, never free text, so every entry links back to a real moment in a real transcript.
Without a model (demo mode), threads come from word overlap, and a thread counts as changed when
its dates or numbers differ between meetings.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from .retrieval import tokens
from .store import Store

Status = Literal["changed", "reversed", "consistent"]


class ThreadOut(BaseModel):
    topic: str = Field(description="Short topic name, e.g. 'Launch date'.")
    status: Status = Field(
        description="changed: a value moved; reversed: undone; consistent: confirmed again."
    )
    summary: str = Field(
        description="One sentence: what it was, what it became, and why if stated."
    )
    item_ids: list[str] = Field(
        description="Ids of the items in this thread, oldest first, e.g. ['M1-D2', 'M3-D1']."
    )


class ChangesOut(BaseModel):
    threads: list[ThreadOut]


SYSTEM = """\
You track how a team's decisions evolve across a series of meetings, given in date order.

Each item has an id like M2-D1 (meeting 2, decision 1) or M4-K1 (meeting 4, key point 1).
Group items that are about the same subject (a date, a price,
a scope choice, a metric, an owner) into threads. Only keep threads that span at least two
different meetings. One thread per subject: if several items track the same feature or number
over time, they belong in a single thread, never two threads that overlap.

For each thread:
- status "changed" when a value moved (a date slipped, a price or target changed),
  "reversed" when an earlier decision was undone or brought back, "consistent" when later
  meetings confirm it unchanged.
- summary: one plain sentence saying what it was, what it became, and the reason if stated.
- item_ids: the ids of the items in the thread, oldest first. Use only ids from the input.
  Always include the most recent item on the subject, so each thread ends with where things
  stand now, and the summary must describe that latest state.

Prefer the threads a manager would care about: dates, scope, prices, owners, risks.
List changed and reversed threads before consistent ones."""


def meeting_date(meeting: dict) -> date:
    """Date from a YYYY-MM-DD prefix on the file name, else the upload date."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", meeting.get("source_name", ""))
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    return date.fromisoformat(meeting["created_at"][:10])


def _items(store: Store) -> tuple[list[dict], dict[str, dict]]:
    meetings = sorted(
        (m for m in store.list_meetings() if m["status"] == "ready" and m["analysis"]),
        key=lambda m: (meeting_date(m), m["created_at"]),
    )
    index: dict[str, dict] = {}
    for mi, m in enumerate(meetings, start=1):
        segments = store.get_segments(m["id"])

        def at(seg: int, segments=segments) -> float:
            return segments[seg].start if 0 <= seg < len(segments) else 0.0

        for di, d in enumerate(m["analysis"]["decisions"], start=1):
            index[f"M{mi}-D{di}"] = {
                "meeting": m,
                "kind": "decision",
                "text": d["decision"],
                "start": at(d["segment"]),
            }
        for ki, point in enumerate(m["analysis"]["key_points"], start=1):
            index[f"M{mi}-K{ki}"] = {"meeting": m, "kind": "point", "text": point, "start": None}
    return meetings, index


def _render(meetings: list[dict], index: dict[str, dict]) -> str:
    out = []
    for mi, m in enumerate(meetings, start=1):
        out.append(f"## M{mi}: {m['title']} ({meeting_date(m).isoformat()})")
        out += [
            f"{iid} [{it['kind']}] {it['text']}"
            for iid, it in index.items()
            if iid.startswith(f"M{mi}-")
        ]
        out.append("")
    return "\n".join(out)


def _entry(iid: str, it: dict) -> dict:
    m = it["meeting"]
    return {
        "id": iid,
        "meeting_id": m["id"],
        "meeting_title": m["title"],
        "date": meeting_date(m).isoformat(),
        "kind": it["kind"],
        "statement": it["text"],
        "start": it["start"],
    }


_VALUE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}|\b\d+(?:[.,]\d+)?\s*(?:%|percent|cents|dollars|weeks?)|\$\d+(?:\.\d+)?|\bq[1-4]\b",
    re.I,
)


def _heuristic(meetings: list[dict], index: dict[str, dict]) -> list[dict]:
    """Group decisions by shared words across meetings; changed if their dates/numbers differ."""
    decisions = [
        (iid, it, set(tokens(it["text"]))) for iid, it in index.items() if it["kind"] == "decision"
    ]
    used: set[str] = set()
    threads = []
    for iid, it, words in decisions:
        if iid in used or not words:
            continue
        group = [(iid, it)]
        for jid, jt, jwords in decisions:
            if (
                jid == iid
                or jid in used
                or jt["meeting"]["id"] in {g[1]["meeting"]["id"] for g in group}
            ):
                continue
            if len(words & jwords) / len(words | jwords) >= 0.2:
                group.append((jid, jt))
        if len({g[1]["meeting"]["id"] for g in group}) < 2:
            continue
        used.update(g[0] for g in group)
        values = [frozenset(v.lower() for v in _VALUE.findall(g[1]["text"])) for g in group]
        changed = len({v for v in values if v}) > 1
        topic = (
            " ".join(w.capitalize() for w in sorted(words & set(tokens(group[1][1]["text"])))[:3])
            or "Decision"
        )
        threads.append(
            {
                "topic": topic,
                "status": "changed" if changed else "consistent",
                "summary": f"Came up in {len(group)} meetings"
                + ("; the stated values differ." if changed else "."),
                "history": [_entry(g[0], g[1]) for g in group],
            }
        )
    return sorted(threads, key=lambda t: t["status"] != "changed")


def what_changed(store: Store, llm) -> dict:
    meetings, index = _items(store)
    if len(meetings) < 2:
        return {"threads": [], "meetings": len(meetings), "method": "none"}
    if getattr(llm, "demo", True):
        return {
            "threads": _heuristic(meetings, index),
            "meetings": len(meetings),
            "method": "heuristic",
        }
    # Full reasoning: grouping a whole project's decisions is the hard part, and the result is
    # cached until the meetings change.
    out = llm.complete_json(SYSTEM, _render(meetings, index), ChangesOut, max_tokens=4500)
    threads = []
    for t in out.threads:
        ids = [i for i in dict.fromkeys(t.item_ids) if i in index]  # drop anything made up
        history = sorted(
            (_entry(i, index[i]) for i in ids), key=lambda e: (e["date"], e["start"] or 0.0)
        )
        if len({e["meeting_id"] for e in history}) < 2:
            continue
        threads.append(
            {"topic": t.topic, "status": t.status, "summary": t.summary, "history": history}
        )
    order = {"changed": 0, "reversed": 1, "consistent": 2}
    threads.sort(key=lambda t: order[t["status"]])
    return {"threads": threads, "meetings": len(meetings), "method": llm.name}
