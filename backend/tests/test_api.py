import json

import pytest
from fastapi.testclient import TestClient

from minutes.api import create_app
from minutes.config import Settings
from minutes.llm import DemoLLM


class BrokenLLM(DemoLLM):
    def analyze(self, title_hint, segments):
        raise RuntimeError("model unavailable")


def make_client(tmp_path, llm=None) -> TestClient:
    settings = Settings(
        db_path=tmp_path / "test.db",
        model="claude-opus-5-5",
        demo_mode=True,
        chunk_words=120,
        retrieval_k=5,
        cors_origins=["*"],
    )
    return TestClient(create_app(settings, llm or DemoLLM()))


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


def parse_sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def upload(client, content: str, name: str = "standup.txt"):
    return client.post("/api/meetings", files={"file": (name, content.encode(), "text/plain")})


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_upload_analyzes_in_background(client):
    res = upload(client, "[00:01] Ann: I'll fix the login bug by Friday.\n[00:05] Ben: Thanks.")
    assert res.status_code == 201
    meeting_id = res.json()["id"]

    meeting = client.get(f"/api/meetings/{meeting_id}").json()
    assert meeting["status"] == "ready"
    assert meeting["title"] == "Standup"
    assert meeting["segments"][0] == {
        "speaker": "Ann",
        "start": 1.0,
        "end": 3.8,  # 7 words at ~150 wpm
        "text": "I'll fix the login bug by Friday.",
    }
    [item] = meeting["analysis"]["action_items"]
    assert item["owner"] == "Ann" and item["due"] == "Friday"

    notes = client.get(f"/api/meetings/{meeting_id}/notes.md").text
    assert "- [ ] **Ann**: I'll fix the login bug by Friday., due Friday (00:01)" in notes


def test_custom_title(client):
    res = client.post(
        "/api/meetings",
        files={"file": ("a.txt", b"Ann: hello world", "text/plain")},
        data={"title": "Board meeting"},
    )
    assert res.json()["title"] == "Board meeting"


def test_bad_transcript_is_422(client):
    assert upload(client, "   \n\n").status_code == 422


def test_failed_analysis_keeps_transcript_searchable(tmp_path):
    client = make_client(tmp_path, BrokenLLM())
    meeting_id = upload(client, "Ann: the kiwi migration is blocked").json()["id"]
    meeting = client.get(f"/api/meetings/{meeting_id}").json()
    assert meeting["status"] == "failed" and "model unavailable" in meeting["error"]
    assert client.get("/api/search", params={"q": "kiwi"}).json()


def test_samples_action_items_and_ask(client):
    loaded = client.post("/api/samples").json()
    assert len(loaded) == 3
    assert {m["status"] for m in client.get("/api/meetings").json()} == {"ready"}

    items = client.get("/api/action-items").json()
    assert any(i["owner"] == "Aisha" and "dedupe" in i["task"] for i in items)
    assert all({"meeting", "meeting_id", "start"} <= set(i) for i in items)

    res = client.post("/api/ask", json={"question": "Why was the launch moved?"})
    assert res.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(res.text)
    assert events[0][0] == "sources" and events[0][1]
    assert events[-1] == ("done", None)
    answer = "".join(data for kind, data in events if kind == "delta")
    assert "[S1]" in answer


def test_ask_scoped_to_meeting(client):
    loaded = client.post("/api/samples").json()
    design = next(m for m in loaded if "design" in m["title"].lower())
    res = client.post("/api/ask", json={"question": "Plaid", "meeting_id": design["id"]})
    sources = parse_sse(res.text)[0][1]
    assert sources and {s["meeting_id"] for s in sources} == {design["id"]}


def test_missing_meeting_is_404(client):
    assert client.get("/api/meetings/nope").status_code == 404
    assert client.delete("/api/meetings/nope").status_code == 404
    assert client.post("/api/ask", json={"question": "hi", "meeting_id": "nope"}).status_code == 404


def test_delete(client):
    meeting_id = upload(client, "Ann: hello").json()["id"]
    assert client.delete(f"/api/meetings/{meeting_id}").status_code == 204
    assert client.get("/api/meetings").json() == []


def test_reanalyze(tmp_path):
    client = make_client(tmp_path, BrokenLLM())
    meeting_id = upload(client, "Ann: hello").json()["id"]
    client.app.state.service.llm = DemoLLM()
    assert client.post(f"/api/meetings/{meeting_id}/analyze").status_code == 202
    assert client.get(f"/api/meetings/{meeting_id}").json()["status"] == "ready"
