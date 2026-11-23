"""Live meetings: transcribe audio in short chunks while the meeting runs, refresh the notes as
it goes, and save it as a normal meeting at the end.

The browser records ~10-second chunks and posts each one with its start offset; Groq Whisper
returns timed text in about a second. The browser can also send text it transcribed itself
(the Web Speech API), so live mode works without a Groq key too.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .llm import LLM
from .service import MeetingService
from .transcribe import transcribe_groq
from .transcript import Segment, TranscriptError

MAX_CHUNK_BYTES = 10 * 1024 * 1024


class LiveSegment(BaseModel):
    speaker: str = Field(default="Speaker", max_length=80)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(max_length=4000)


class LiveTranscript(BaseModel):
    title: str = Field(default="Live meeting", max_length=200)
    segments: list[LiveSegment] = Field(max_length=4000)


def _segments(body: LiveTranscript) -> list[Segment]:
    segs = [Segment(s.speaker, s.start, max(s.end, s.start), s.text.strip()) for s in body.segments]
    segs = [s for s in segs if s.text]
    if not segs:
        raise HTTPException(422, "Nothing has been transcribed yet")
    return segs


def live_router(service: MeetingService, llm: LLM, settings) -> APIRouter:
    router = APIRouter(prefix="/api/live")

    @router.post("/transcribe")
    async def transcribe_chunk(
        file: UploadFile = File(...),
        offset: float = Form(0.0),
        speaker: str = Form("Speaker"),
        prompt: str = Form(""),
    ) -> list[dict]:
        if not settings.groq_api_key:
            raise HTTPException(501, "Server-side live transcription needs a Groq key")
        data = await file.read(MAX_CHUNK_BYTES + 1)
        if len(data) > MAX_CHUNK_BYTES:
            raise HTTPException(413, "Audio chunk is too large")
        if len(data) < 1000:
            return []  # silence or an empty chunk
        try:
            segs = transcribe_groq(
                data,
                file.filename or "chunk.webm",
                settings.groq_api_key,
                speaker=speaker[:80] or "Speaker",
                offset=max(offset, 0.0),
                prompt=prompt,
            )
        except TranscriptError as exc:
            raise HTTPException(502, str(exc)) from exc
        return [s.__dict__ for s in segs]

    @router.post("/notes")
    def notes(body: LiveTranscript) -> dict:
        """Notes for the meeting so far (refreshed every so often while it runs)."""
        segs = _segments(body)
        try:
            return service.llm.analyze(body.title, segs).model_dump()
        except Exception as exc:  # noqa: BLE001 - the next refresh tries again
            raise HTTPException(502, f"Could not update notes: {str(exc)[:200]}") from exc

    @router.post("/save", status_code=201)
    def save(body: LiveTranscript, background: BackgroundTasks) -> dict:
        """End the live meeting: store it like an upload and analyze the full transcript."""
        meeting_id = service.ingest_segments(_segments(body), "live-meeting.txt", body.title)
        if settings.sync_analysis:
            service.analyze(meeting_id)
        else:
            background.add_task(service.analyze, meeting_id)
        return service.store.get_meeting(meeting_id)

    return router
