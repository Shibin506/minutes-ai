"""GeminiLLM against a mocked OpenAI-compatible endpoint: no network, no API key."""

import json

import httpx
import pytest
import respx

from minutes.config import Settings
from minutes.llm import GEMINI_BASE_URL, AnalysisError, ClaudeLLM, DemoLLM, GeminiLLM, build_llm
from minutes.store import SearchHit
from minutes.transcript import Segment

URL = f"{GEMINI_BASE_URL}/chat/completions"
SEGMENTS = [Segment("Ann", 0, 5, "I'll send the deck by Friday."), Segment("Ben", 6, 9, "Great.")]
ANALYSIS = {
    "title": "Sync",
    "summary": "Ann will send the deck.",
    "key_points": ["Deck due Friday"],
    "decisions": [{"decision": "Ship it", "segment": 1}],
    "action_items": [{"task": "Send the deck", "owner": "Ann", "due": "Friday", "segment": 7}],
    "open_questions": [],
    "topics": [{"name": "Deck", "first_segment": 0, "last_segment": 1}],
}


@respx.mock
def test_analyze_sends_a_strict_schema_and_clamps_segments():
    route = respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(ANALYSIS)}}]
            },
        )
    )
    result = GeminiLLM(api_key="k", reasoning_effort="low").analyze("Sync", SEGMENTS)
    assert result.action_items[0].owner == "Ann"
    assert result.action_items[0].segment == 1  # clamped into the transcript
    req = route.calls[0].request
    assert req.headers["authorization"] == "Bearer k"
    body = json.loads(req.content)
    fmt = body["response_format"]["json_schema"]
    assert fmt["strict"] is True and "$ref" not in json.dumps(fmt["schema"])
    # "title" is both a JSON Schema annotation (dropped) and a real field here (kept)
    assert "title" in fmt["schema"]["properties"] and "title" in fmt["schema"]["required"]
    assert "title" not in fmt["schema"]["properties"]["summary"]
    assert body["reasoning_effort"] == "low"
    assert body["messages"][0]["role"] == "system"


@respx.mock
def test_analyze_errors_are_reported():
    respx.post(URL).mock(return_value=httpx.Response(429, json={"error": "quota"}))
    with pytest.raises(AnalysisError, match="429"):
        GeminiLLM().analyze("Sync", SEGMENTS)


@respx.mock
def test_stream_answer_yields_text_deltas():
    chunks = [
        {"choices": [{"delta": {"content": "The launch moved "}}]},
        {"choices": [{"delta": {"content": "to May [S1]."}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    route = respx.post(URL).mock(return_value=httpx.Response(200, text=sse))
    hits = [SearchHit("m1", "Launch", 0, 0, 10, "Ann: we move the launch to May.", 1.0)]
    assert "".join(GeminiLLM().stream_answer("Why?", hits)) == "The launch moved to May [S1]."
    assert json.loads(route.calls[0].request.content)["stream"] is True


def test_build_llm_picks_the_provider(tmp_path):
    base = dict(
        db_path=tmp_path / "x.db",
        model="claude-opus-5-5",
        chunk_words=160,
        retrieval_k=8,
        cors_origins=[],
    )
    assert isinstance(build_llm(Settings(demo_mode=True, provider="gemini", **base)), DemoLLM)
    assert isinstance(build_llm(Settings(demo_mode=False, provider="gemini", **base)), GeminiLLM)
    assert isinstance(build_llm(Settings(demo_mode=False, **base)), ClaudeLLM)


def test_settings_read_the_gemini_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MINUTES_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    s = Settings.from_env()
    assert s.provider == "gemini" and not s.demo_mode and s.gemini_api_key == "k"
    monkeypatch.delenv("GEMINI_API_KEY")
    assert Settings.from_env().demo_mode
