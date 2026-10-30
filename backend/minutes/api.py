"""HTTP API (FastAPI). Run with: uvicorn minutes.api:create_app --factory --reload"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .llm import LLM, build_llm
from .service import MeetingService, to_markdown
from .store import Store
from .transcribe import is_audio, transcribe
from .transcript import TranscriptError

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
SAMPLES_DIR = Path(
    os.getenv("MINUTES_SAMPLES_DIR", Path(__file__).resolve().parents[2] / "samples")
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    meeting_id: str | None = None


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.db_path)
    llm = llm or build_llm(settings.model, settings.demo_mode)
    service = MeetingService(store, llm, settings.chunk_words)

    app = FastAPI(title="Minutes", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = service

    def require(meeting_id: str) -> dict:
        meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise HTTPException(404, "Meeting not found")
        return meeting

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__, "model": llm.name}

    @app.get("/api/meetings")
    def list_meetings() -> list[dict]:
        return store.list_meetings()

    @app.post("/api/meetings", status_code=201)
    async def upload(
        background: BackgroundTasks,
        file: UploadFile = File(...),
        title: str | None = Form(None),
    ) -> dict:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File is larger than 200 MB")
        filename = file.filename or "meeting.txt"
        try:
            if is_audio(filename):
                with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    segments = transcribe(Path(tmp.name))
                meeting_id = service.ingest_segments(segments, filename, title)
            else:
                text = data.decode("utf-8", errors="replace")
                meeting_id = service.ingest_text(text, filename, title)
        except TranscriptError as exc:
            raise HTTPException(422, str(exc)) from exc
        background.add_task(service.analyze, meeting_id)
        return require(meeting_id)

    @app.post("/api/samples", status_code=201)
    def load_samples(background: BackgroundTasks) -> list[dict]:
        """Load the bundled example meetings so the app can be tried in one click."""
        loaded = []
        for path in sorted(SAMPLES_DIR.glob("*.vtt")):
            meeting_id = service.ingest_text(path.read_text(), path.name)
            background.add_task(service.analyze, meeting_id)
            loaded.append(require(meeting_id))
        return loaded

    @app.get("/api/meetings/{meeting_id}")
    def get_meeting(meeting_id: str) -> dict:
        meeting = require(meeting_id)
        meeting["segments"] = [s.__dict__ for s in store.get_segments(meeting_id)]
        return meeting

    @app.delete("/api/meetings/{meeting_id}", status_code=204)
    def delete_meeting(meeting_id: str) -> None:
        if not store.delete_meeting(meeting_id):
            raise HTTPException(404, "Meeting not found")

    @app.post("/api/meetings/{meeting_id}/analyze", status_code=202)
    def reanalyze(meeting_id: str, background: BackgroundTasks) -> dict:
        require(meeting_id)
        store.set_status(meeting_id, "processing")
        background.add_task(service.analyze, meeting_id)
        return {"status": "processing"}

    @app.get("/api/meetings/{meeting_id}/notes.md", response_class=PlainTextResponse)
    def export_notes(meeting_id: str) -> str:
        return to_markdown(require(meeting_id), store.get_segments(meeting_id))

    @app.get("/api/action-items")
    def action_items() -> list[dict]:
        """Every action item across all meetings, newest meeting first."""
        items = []
        for meeting in store.list_meetings():
            analysis = meeting["analysis"] or {}
            segments = store.get_segments(meeting["id"]) if analysis else []
            for item in analysis.get("action_items", []):
                seg = item["segment"]
                start = segments[seg].start if 0 <= seg < len(segments) else 0.0
                items.append(
                    {
                        **item,
                        "start": start,
                        "meeting_id": meeting["id"],
                        "meeting": meeting["title"],
                    }
                )
        return items

    @app.get("/api/search")
    def search(q: str, meeting_id: str | None = None, k: int = 10) -> list[dict]:
        return [hit.__dict__ for hit in store.search(q, k=min(k, 50), meeting_id=meeting_id)]

    @app.post("/api/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        """Retrieval-augmented Q&A over meetings, streamed as server-sent events.

        Events: `sources` (the cited excerpts), `delta` (answer text), `done`, `error`.
        """
        if req.meeting_id:
            require(req.meeting_id)
        hits = store.search(req.question, k=settings.retrieval_k, meeting_id=req.meeting_id)

        def sse(event: str, data: object) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def events() -> Iterator[str]:
            yield sse("sources", [hit.__dict__ for hit in hits])
            try:
                for text in llm.stream_answer(req.question, hits):
                    yield sse("delta", text)
            except Exception as exc:  # the stream is already open; report in-band
                yield sse("error", str(exc)[:300])
            yield sse("done", None)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    return app
