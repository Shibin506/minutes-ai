"""Split a transcript into overlapping, speaker-aware retrieval chunks."""

from __future__ import annotations

from dataclasses import dataclass

from .transcript import Segment


@dataclass
class Chunk:
    index: int
    first_segment: int
    last_segment: int
    start: float
    end: float
    text: str


def render_segment(seg: Segment) -> str:
    return f"{seg.speaker}: {seg.text}"


def chunk_segments(segments: list[Segment], max_words: int = 160, overlap: int = 1) -> list[Chunk]:
    """Greedily pack whole turns into chunks of ~max_words.

    Chunks never split a turn, so every chunk has a clean start/end timestamp to
    cite. The last `overlap` turns of a chunk are repeated at the start of the
    next one so a question-and-answer exchange is never cut in half.
    """
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    chunks: list[Chunk] = []
    i = 0
    while i < len(segments):
        j = i
        words = 0
        while j < len(segments):
            n = len(segments[j].text.split())
            if j > i and words + n > max_words:
                break
            words += n
            j += 1
        window = segments[i:j]
        chunks.append(
            Chunk(
                index=len(chunks),
                first_segment=i,
                last_segment=j - 1,
                start=window[0].start,
                end=window[-1].end,
                text="\n".join(render_segment(s) for s in window),
            )
        )
        if j >= len(segments):
            break
        i = max(j - overlap, i + 1)
    return chunks
