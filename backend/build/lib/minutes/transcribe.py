"""Optional local speech-to-text for audio uploads (`pip install minutes-ai[audio]`).

Uses faster-whisper on CPU. There is no speaker diarization, so every turn is
labelled "Speaker"; upload a VTT export from Zoom/Meet/Teams to keep names.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .transcript import Segment, TranscriptError

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".mp4", ".webm", ".ogg", ".flac")


def is_audio(filename: str) -> bool:
    return filename.lower().endswith(AUDIO_EXTENSIONS)


@lru_cache(maxsize=1)
def _model(size: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptError(
            "Audio upload needs the optional dependency: pip install 'minutes-ai[audio]'. "
            "Or upload a transcript (.vtt, .srt, .txt, .json)."
        ) from exc
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(path: Path, model_size: str = "base") -> list[Segment]:
    segments, _info = _model(model_size).transcribe(str(path), vad_filter=True)
    return [Segment("Speaker", s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
