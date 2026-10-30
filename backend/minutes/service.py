"""The ingest pipeline: transcript -> segments -> chunks -> index -> analysis."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .chunking import chunk_segments
from .llm import LLM
from .schemas import MeetingAnalysis
from .store import Store
from .transcript import Segment, format_timestamp, merge_turns, parse_transcript

log = logging.getLogger(__name__)


def title_from_filename(filename: str) -> str:
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}[_ -]*", "", Path(filename).name.rsplit(".", 1)[0])
    stem = stem.replace("_", " ").replace("-", " ").strip()
    return stem[:1].upper() + stem[1:] if stem else "Untitled meeting"


class MeetingService:
    def __init__(self, store: Store, llm: LLM, chunk_words: int = 160):
        self.store = store
        self.llm = llm
        self.chunk_words = chunk_words

    def ingest_text(self, content: str, filename: str, title: str | None = None) -> str:
        return self.ingest_segments(parse_transcript(content, filename), filename, title)

    def ingest_segments(self, segments: list[Segment], filename: str, title: str | None) -> str:
        segments = merge_turns(segments)
        chunks = chunk_segments(segments, max_words=self.chunk_words)
        return self.store.create_meeting(
            title or title_from_filename(filename), filename, segments, chunks
        )

    def analyze(self, meeting_id: str) -> None:
        """Run the model over a stored meeting. Safe to call from a background task."""
        meeting = self.store.get_meeting(meeting_id)
        if meeting is None:
            return
        self.store.set_status(meeting_id, "processing")
        try:
            analysis = self.llm.analyze(meeting["title"], self.store.get_segments(meeting_id))
        except Exception as exc:  # surfaced to the UI; the transcript stays searchable
            log.exception("analysis failed for %s", meeting_id)
            self.store.set_status(meeting_id, "failed", str(exc)[:500])
            return
        self.store.set_analysis(meeting_id, analysis)


def to_markdown(meeting: dict, segments: list[Segment]) -> str:
    """Render meeting notes for pasting into a doc, wiki or ticket."""

    def at(i: int) -> str:
        return format_timestamp(segments[i].start) if 0 <= i < len(segments) else "--:--"

    lines = [f"# {meeting['title']}", ""]
    if not meeting.get("analysis"):
        lines.append("_Analysis not available yet._")
        return "\n".join(lines) + "\n"
    a = MeetingAnalysis.model_validate(meeting["analysis"])
    lines += ["## Summary", "", a.summary, ""]
    if a.key_points:
        lines += ["## Key points", "", *[f"- {p}" for p in a.key_points], ""]
    if a.decisions:
        lines += [
            "## Decisions",
            "",
            *[f"- {d.decision} ({at(d.segment)})" for d in a.decisions],
            "",
        ]
    if a.action_items:
        lines += ["## Action items", ""]
        for item in a.action_items:
            due = f", due {item.due}" if item.due else ""
            lines.append(f"- [ ] **{item.owner}**: {item.task}{due} ({at(item.segment)})")
        lines.append("")
    if a.open_questions:
        lines += ["## Open questions", "", *[f"- {q}" for q in a.open_questions], ""]
    return "\n".join(lines)
