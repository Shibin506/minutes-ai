"""HTTP API (FastAPI). Run with: uvicorn minutes.api:create_app --factory --reload"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .changes import what_changed
from .config import Settings
from .exports import action_items_ics, follow_up
from .live import live_router
from .llm import LLM, build_llm, build_reranker
from .retrieval import LSAEmbedder, Retriever
from .schemas import MeetingAnalysis
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


def seed_samples(service: MeetingService, settings: Settings, background=None) -> list[str]:
    """Load the sample meetings. Saved analyses (real model output, see
    scripts/precompute_samples.py) are used when present, so no model call is needed."""
    store = service.store
    ids = []
    for path in sorted(SAMPLES_DIR.glob("*.vtt")):
        meeting_id = service.ingest_text(path.read_text(), path.name)
        saved = SAMPLES_DIR / "analyses" / f"{path.stem}.json"
        if saved.exists():
            store.set_analysis(meeting_id, MeetingAnalysis.model_validate_json(saved.read_text()))
        elif settings.sync_analysis or background is None:
            service.analyze(meeting_id)
        else:
            background.add_task(service.analyze, meeting_id)
        ids.append(meeting_id)
    return ids


def create_app(settings: Settings | None = None, llm: LLM | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.db_path)
    llm = llm or build_llm(settings)
    service = MeetingService(store, llm, settings.chunk_words)
    retriever = Retriever(
        store, LSAEmbedder(), llm=build_reranker(settings, llm), mode=settings.retrieval
    )
    if settings.seed_samples and not store.list_meetings():
        seed_samples(service, settings)
    # "What changed" is one model call over every meeting; cache it until the meetings change.
    changes_cache: dict[str, dict] = {}

    app = FastAPI(title="Minutes", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.service = service
    app.state.retriever = retriever

    def analyze_now_or_later(background: BackgroundTasks, meeting_id: str) -> None:
        # Serverless hosts stop work once the response is sent, so analyze inside the request.
        if settings.sync_analysis:
            service.analyze(meeting_id)
        else:
            background.add_task(service.analyze, meeting_id)

    def require(meeting_id: str) -> dict:
        meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise HTTPException(404, "Meeting not found")
        return meeting

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "model": llm.name,
            "demo": bool(getattr(llm, "demo", False)),
            # Reranking needs a model: without one (demo mode) it is plain hybrid.
            "retrieval": settings.retrieval if retriever.llm else settings.retrieval.split("+")[0],
            "live": bool(settings.groq_api_key),
            "hosted_demo": settings.seed_samples,
        }

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
                    segments = transcribe(Path(tmp.name), groq_api_key=settings.groq_api_key)
                meeting_id = service.ingest_segments(segments, filename, title)
            else:
                text = data.decode("utf-8", errors="replace")
                meeting_id = service.ingest_text(text, filename, title)
        except TranscriptError as exc:
            raise HTTPException(422, str(exc)) from exc
        analyze_now_or_later(background, meeting_id)
        return require(meeting_id)

    @app.post("/api/samples", status_code=201)
    def load_samples(background: BackgroundTasks) -> list[dict]:
        """Load the bundled example meetings so the app can be tried in one click."""
        return [require(i) for i in seed_samples(service, settings, background)]

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
        analyze_now_or_later(background, meeting_id)
        return {"status": require(meeting_id)["status"]}

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
        hits = retriever.search(q, k=min(k, 50), meeting_id=meeting_id)
        return [hit.__dict__ for hit in hits]

    @app.get("/api/changes")
    def changes() -> dict:
        """Decisions tracked across meetings: what moved, what flipped, what held."""
        ready = [m for m in store.list_meetings() if m["status"] == "ready"]
        key = ",".join(
            sorted(f"{m['id']}:{hash(json.dumps(m['analysis'], sort_keys=True))}" for m in ready)
        )
        if key not in changes_cache:
            saved = SAMPLES_DIR / "analyses" / "_changes.json"
            sample_names = sorted(p.name for p in SAMPLES_DIR.glob("*.vtt"))
            if saved.exists() and sorted(m["source_name"] for m in ready) == sample_names:
                changes_cache[key] = remap_saved_changes(json.loads(saved.read_text()), ready)
            else:
                try:
                    changes_cache[key] = what_changed(store, llm)
                except Exception as exc:  # noqa: BLE001 - shown in the UI
                    raise HTTPException(
                        502, f"Could not compare meetings: {str(exc)[:200]}"
                    ) from exc
        return changes_cache[key]

    @app.get("/api/meetings/{meeting_id}/action-items.ics")
    def meeting_ics(meeting_id: str) -> Response:
        text, _, _ = action_items_ics([require(meeting_id)])
        return Response(
            text,
            media_type="text/calendar",
            headers={
                "Content-Disposition": f'attachment; filename="action-items-{meeting_id}.ics"'
            },
        )

    @app.get("/api/action-items.ics")
    def all_ics() -> Response:
        text, _, _ = action_items_ics(store.list_meetings())
        return Response(
            text,
            media_type="text/calendar",
            headers={"Content-Disposition": 'attachment; filename="action-items.ics"'},
        )

    @app.post("/api/meetings/{meeting_id}/follow-up")
    def follow_up_email(meeting_id: str) -> dict:
        meeting = require(meeting_id)
        if not meeting.get("analysis"):
            raise HTTPException(409, "The meeting has not been analyzed yet")
        try:
            return follow_up(meeting, llm).model_dump()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(502, f"Could not draft the email: {str(exc)[:200]}") from exc

    app.include_router(live_router(service, llm, settings))

    @app.post("/api/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        """Retrieval-augmented Q&A over meetings, streamed as server-sent events.

        Events: `sources` (the cited excerpts), `delta` (answer text), `done`, `error`.
        """
        if req.meeting_id:
            require(req.meeting_id)
        hits = retriever.search(req.question, k=settings.retrieval_k, meeting_id=req.meeting_id)

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


def remap_saved_changes(saved: dict, ready: list[dict]) -> dict:
    """Saved sample "what changed" output refers to meetings by file name; point it at this
    database's meeting ids."""
    by_source = {m["source_name"]: m for m in ready}
    for thread in saved["threads"]:
        for entry in thread["history"]:
            m = by_source[entry.pop("source_name")]
            entry["meeting_id"] = m["id"]
            entry["meeting_title"] = m["title"]
    return saved
