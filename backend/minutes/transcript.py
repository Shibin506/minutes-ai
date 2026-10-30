"""Parse meeting transcripts from common formats into timed, speaker-labelled segments.

Supported inputs:
  * WebVTT (.vtt) - Zoom, Teams and Google Meet exports, with `<v Name>` or `Name:` speakers
  * SubRip (.srt)
  * Plain text (.txt) - `[hh:mm:ss] Name: text`, `Name: text`, or bare lines
  * JSON (.json) - a list of {speaker, start, end, text}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

WORDS_PER_SECOND = 2.5  # ~150 wpm, used when a transcript has no timestamps


class TranscriptError(ValueError):
    """The uploaded file could not be read as a transcript."""


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


_CUE_TIME = re.compile(
    r"(?P<start>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*(?P<end>(?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})"
)
_VOICE_TAG = re.compile(r"^<v(?:\.[^ >]+)?\s+([^>]+)>(.*?)(?:</v>)?$")
_SPEAKER_PREFIX = re.compile(r"^(?P<speaker>[A-Z][\w .'\-]{0,40}?):\s+(?P<text>.+)$")
_TXT_LINE = re.compile(r"^\[?(?P<ts>(?:\d+:)?\d{1,2}:\d{2}(?:[.,]\d+)?)\]?\s*[-–]?\s*(?P<rest>.*)$")
_TAGS = re.compile(r"</?[^>]+>")


def parse_timestamp(value: str) -> float:
    """'01:02:03.500' / '02:03,5' / '2:03' -> seconds."""
    parts = value.replace(",", ".").split(":")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _split_speaker(text: str, default: str) -> tuple[str, str]:
    voice = _VOICE_TAG.match(text)
    if voice:
        return voice.group(1).strip(), _TAGS.sub("", voice.group(2)).strip()
    text = _TAGS.sub("", text).strip()
    prefixed = _SPEAKER_PREFIX.match(text)
    if prefixed:
        return prefixed.group("speaker").strip(), prefixed.group("text").strip()
    return default, text


def _parse_cues(content: str) -> list[Segment]:
    """Shared VTT/SRT parser: every block with a `-->` line is a cue."""
    segments: list[Segment] = []
    for block in re.split(r"\n\s*\n", content):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        for i, line in enumerate(lines):
            match = _CUE_TIME.search(line)
            if not match:
                continue
            body = " ".join(lines[i + 1 :])
            if body:
                speaker, text = _split_speaker(body, "Speaker")
                if text:
                    segments.append(
                        Segment(
                            speaker,
                            parse_timestamp(match.group("start")),
                            parse_timestamp(match.group("end")),
                            text,
                        )
                    )
            break
    return segments


def _parse_text(content: str) -> list[Segment]:
    segments: list[Segment] = []
    clock = 0.0
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        start: float | None = None
        ts = _TXT_LINE.match(line)
        if ts and ts.group("rest"):
            start = parse_timestamp(ts.group("ts"))
            line = ts.group("rest").strip()
        speaker, text = _split_speaker(line, segments[-1].speaker if segments else "Speaker")
        if not text:
            continue
        start = clock if start is None else start
        duration = max(1.0, len(text.split()) / WORDS_PER_SECOND)
        if segments and segments[-1].end > start:
            segments[-1].end = max(segments[-1].start, start)
        segments.append(Segment(speaker, start, start + duration, text))
        clock = start + duration
    return segments


def _parse_json(content: str) -> list[Segment]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise TranscriptError(f"Invalid JSON: {exc.msg}") from exc
    if isinstance(data, dict):
        data = data.get("segments", [])
    if not isinstance(data, list):
        raise TranscriptError("JSON transcript must be a list of segments.")
    segments = []
    for item in data:
        if not isinstance(item, dict) or not str(item.get("text", "")).strip():
            continue
        start = float(item.get("start", 0))
        end = float(item.get("end", start))
        segments.append(
            Segment(
                str(item.get("speaker") or "Speaker"), start, max(end, start), item["text"].strip()
            )
        )
    return segments


def merge_turns(
    segments: list[Segment], max_gap: float = 2.0, max_words: int = 120
) -> list[Segment]:
    """Join consecutive cues from the same speaker into a single turn.

    Caption exports split one sentence across many short cues; turns are what
    people (and the model) actually reason about.
    """
    merged: list[Segment] = []
    for seg in segments:
        prev = merged[-1] if merged else None
        if (
            prev
            and prev.speaker == seg.speaker
            and seg.start - prev.end <= max_gap
            and len(prev.text.split()) + len(seg.text.split()) <= max_words
        ):
            prev.text = f"{prev.text} {seg.text}"
            prev.end = max(prev.end, seg.end)
        else:
            merged.append(Segment(seg.speaker, seg.start, seg.end, seg.text))
    return merged


def parse_transcript(content: str, filename: str = "") -> list[Segment]:
    content = content.lstrip("﻿").replace("\r\n", "\n")
    name = filename.lower()
    if name.endswith(".json") or re.match(r"\s*(\{|\[\s*\{)", content):
        segments = _parse_json(content)
    elif name.endswith((".vtt", ".srt")) or content.startswith("WEBVTT") or "-->" in content:
        segments = _parse_cues(content)
    else:
        segments = _parse_text(content)
    segments = merge_turns(sorted(segments, key=lambda s: s.start))
    if not segments:
        raise TranscriptError("No speech found in the transcript.")
    return segments
