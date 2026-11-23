"""GroqLLM and Groq Whisper against mocked endpoints: no network, no API key."""

import json

import httpx
import pytest
import respx

from minutes.config import Settings
from minutes.llm import GROQ_BASE_URL, AnalysisError, GroqLLM, build_llm
from minutes.transcribe import GROQ_TRANSCRIBE_URL, transcribe_groq
from minutes.transcript import Segment

URL = f"{GROQ_BASE_URL}/chat/completions"
SEGMENTS = [Segment("Ann", 0, 5, "I'll send the deck by Friday."), Segment("Ben", 6, 9, "Great.")]
ANALYSIS = {
    "title": "Sync",
    "summary": "Ann will send the deck.",
    "key_points": ["Deck due Friday"],
    "decisions": [],
    "action_items": [{"task": "Send the deck", "owner": "Ann", "due": "Friday", "segment": 0}],
    "open_questions": [],
    "topics": [{"name": "Deck", "first_segment": 0, "last_segment": 1}],
}


def ok(content: str) -> httpx.Response:
    return httpx.Response(
        200, json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]}
    )


@respx.mock
def test_analysis_uses_a_strict_closed_schema():
    route = respx.post(URL).mock(return_value=ok(json.dumps(ANALYSIS)))
    result = GroqLLM(api_key="k", retry_delays=()).analyze("Sync", SEGMENTS)
    assert result.action_items[0].owner == "Ann"
    body = json.loads(route.calls[0].request.content)
    schema = body["response_format"]["json_schema"]["schema"]
    assert body["response_format"]["json_schema"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    item = schema["properties"]["action_items"]["items"]
    assert item["additionalProperties"] is False  # nested objects are closed too
    assert body["reasoning_effort"] == "medium"


@respx.mock
def test_switches_to_a_current_model_when_the_configured_one_is_retired():
    calls = []

    def chat(request):
        model = json.loads(request.content)["model"]
        calls.append(model)
        if model == "llama-3.3-70b-versatile":
            return httpx.Response(
                400, json={"error": {"code": "model_decommissioned", "message": "retired"}}
            )
        return ok("Hello")

    respx.post(URL).mock(side_effect=chat)
    respx.get(f"{GROQ_BASE_URL}/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "meta-llama/llama-prompt-guard-2-22m"},
                    {"id": "openai/gpt-oss-120b"},
                ]
            },
        )
    )
    llm = GroqLLM(model="llama-3.3-70b-versatile", api_key="k", retry_delays=())
    assert llm.complete("sys", "hi") == "Hello"
    assert calls == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    assert llm.model == "openai/gpt-oss-120b"


@respx.mock
def test_whisper_segments_are_offset_and_silence_is_dropped():
    respx.post(GROQ_TRANSCRIBE_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": " We ship Friday. ", "no_speech_prob": 0.01},
                    {"start": 2.0, "end": 4.0, "text": " Thank you.", "no_speech_prob": 0.95},
                ]
            },
        )
    )
    segs = transcribe_groq(b"audio", "a.webm", "k", speaker="Room", offset=30)
    assert [(s.start, s.end, s.speaker, s.text) for s in segs] == [
        (30.0, 32.0, "Room", "We ship Friday.")
    ]


def test_provider_is_picked_from_whichever_key_is_set(monkeypatch):
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "Groq",
        "MINUTES_PROVIDER",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("Groq", "g")
    s = Settings.from_env()
    assert (s.provider, s.demo_mode, s.groq_api_key) == ("groq", False, "g")
    assert isinstance(build_llm(s), GroqLLM)
    monkeypatch.delenv("Groq")
    assert Settings.from_env().demo_mode is True


@respx.mock
def test_other_400s_do_not_switch_models():
    respx.post(URL).mock(
        return_value=httpx.Response(
            400,
            json={"error": {"code": "json_validate_failed", "message": "model output was invalid"}},
        )
    )
    llm = GroqLLM(model="openai/gpt-oss-20b", api_key="k", retry_delays=())
    with pytest.raises(AnalysisError):
        llm.complete("sys", "hi")
    assert llm.model == "openai/gpt-oss-20b"


@respx.mock
def test_daily_quota_falls_back_to_another_model():
    seen = []

    def chat(request):
        model = json.loads(request.content)["model"]
        seen.append(model)
        if model == "openai/gpt-oss-120b":
            return httpx.Response(
                429,
                headers={"retry-after": "420"},
                json={"error": {"message": "Rate limit reached on tokens per day (TPD)"}},
            )
        return ok("Fine")

    respx.post(URL).mock(side_effect=chat)
    llm = GroqLLM(api_key="k", retry_delays=())
    assert llm.complete("sys", "hi") == "Fine"
    assert seen == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


@respx.mock
def test_streamed_answers_fall_back_on_daily_quota():
    def chat(request):
        model = json.loads(request.content)["model"]
        if model == "openai/gpt-oss-120b":
            return httpx.Response(
                429, headers={"retry-after": "600"}, json={"error": {"message": "tokens per day"}}
            )
        delta = json.dumps({"choices": [{"delta": {"content": "Moved to March 15 [S1]."}}]})
        sse = f"data: {delta}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    respx.post(URL).mock(side_effect=chat)
    out = "".join(GroqLLM(api_key="k", retry_delays=()).stream_answer("When?", []))
    assert out == "Moved to March 15 [S1]."


def test_whisper_silence_hallucination_dropped():
    from minutes.transcribe import transcribe_groq

    def handler(request):
        return httpx.Response(
            200,
            json={
                "segments": [
                    {"start": 0.0, "end": 8.0, "text": " Thank you.", "no_speech_prob": 0.4}
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    assert transcribe_groq(b"x", "c.webm", "k", http=http) == []
