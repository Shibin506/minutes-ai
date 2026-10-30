"""Prompt text and the helpers that render transcripts and sources into it."""

from __future__ import annotations

from .store import SearchHit
from .transcript import Segment, format_timestamp

ANALYSIS_SYSTEM = """\
You turn meeting transcripts into notes that a teammate who missed the meeting can act on.

The transcript is given as numbered segments: `[#<segment> <time>] <speaker>: <text>`.
When a field asks for a segment number, use the number of the segment where the
item is actually said or agreed - never guess one.

Guidelines:
- Action items are concrete commitments with a clear next step. Use the name of the
  person who took it on as owner; if nobody did, use "Unassigned". Copy deadlines as
  said ("Friday", "end of Q3"); leave `due` empty if none was given.
- Decisions are things the group settled, not proposals that were left open.
- Open questions are things raised but not resolved by the end of the meeting.
- Topics cover the meeting in order, without gaps or overlaps.
- Be specific: names, numbers and product terms from the transcript beat generic wording.
- Do not invent anything that is not in the transcript.
"""

ANSWER_SYSTEM = """\
You answer questions about a team's past meetings using only the transcript excerpts provided.

Each excerpt is wrapped in <source id="S#" meeting="..." time="...">. Cite every claim with
the id(s) of the excerpt(s) that support it, in square brackets, e.g. "We moved launch to
June [S2]." Combine information across meetings when the question needs it, and say which
meeting said what when they disagree; later meetings supersede earlier ones.

If the excerpts don't contain the answer, say so plainly and suggest what to search for
instead. Keep answers short: a direct answer first, then supporting detail as bullets
if needed. Do not use headings.
"""


def render_transcript(segments: list[Segment]) -> str:
    return "\n".join(
        f"[#{i} {format_timestamp(s.start)}] {s.speaker}: {s.text}" for i, s in enumerate(segments)
    )


def analysis_request(title_hint: str, segments: list[Segment]) -> str:
    return (
        f"Meeting file: {title_hint}\n\n<transcript>\n{render_transcript(segments)}\n</transcript>"
    )


def answer_request(question: str, hits: list[SearchHit]) -> str:
    sources = "\n\n".join(
        f'<source id="S{i}" meeting="{h.meeting_title}" '
        f'time="{format_timestamp(h.start)}-{format_timestamp(h.end)}">\n{h.text}\n</source>'
        for i, h in enumerate(hits, start=1)
    )
    return f"{sources}\n\nQuestion: {question}"
