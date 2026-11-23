"""Speech-to-text for audio uploads and live meetings.

With a Groq key, audio goes to Groq's hosted Whisper (fast, nothing to install, works on
serverless hosts). Otherwise it falls back to local faster-whisper
(`pip install minutes-ai[audio]`). There is no speaker diarization, so every turn is
labelled with one speaker name; upload a VTT export from Zoom/Meet/Teams to keep names.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import httpx

from .transcript import Segment, TranscriptError

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".flac")
GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"


def is_audio(filename: str) -> bool:
    return filename.lower().endswith(AUDIO_EXTENSIONS)


def transcribe_groq(
    data: bytes,
    filename: str,
    api_key: str,
    speaker: str = "Speaker",
    offset: float = 0.0,
    http: httpx.Client | None = None,
    prompt: str = "",
) -> list[Segment]:
    """Transcribe with Groq Whisper; `offset` shifts timestamps (for live chunks)."""
    client = http or httpx.Client(timeout=120)
    form = {"model": GROQ_WHISPER_MODEL, "response_format": "verbose_json", "temperature": "0"}
    if prompt:  # the previous words, so sentences carry across chunk boundaries
        form["prompt"] = prompt[-400:]
    r = client.post(
        GROQ_TRANSCRIBE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        data=form,
        files={"file": (filename, data)},
    )
    if r.status_code >= 400:
        raise TranscriptError(f"Transcription failed (HTTP {r.status_code}): {r.text[:200]}")
    body = r.json()
    segments = body.get("segments") or [
        {"start": 0.0, "end": body.get("duration", 0.0), "text": body.get("text", "")}
    ]
    return [
        Segment(speaker, offset + float(s["start"]), offset + float(s["end"]), s["text"].strip())
        for s in segments
        if s.get("text", "").strip() and s.get("no_speech_prob", 0) < 0.8
    ]


@lru_cache(maxsize=1)
def _model(size: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptError(
            "Audio upload needs a Groq key (GROQ_API_KEY) or the optional dependency: "
            "pip install 'minutes-ai[audio]'. Or upload a transcript (.vtt, .srt, .txt, .json)."
        ) from exc
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(path: Path, model_size: str = "base", groq_api_key: str = "") -> list[Segment]:
    if groq_api_key:
        return transcribe_groq(path.read_bytes(), path.name, groq_api_key)
    segments, _info = _model(model_size).transcribe(str(path), vad_filter=True)
    return [Segment("Speaker", s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
