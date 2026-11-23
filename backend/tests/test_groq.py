"""GroqLLM and Groq Whisper against mocked endpoints: no network, no API key."""

import json

import httpx
import respx

from minutes.config import Settings
from minutes.llm import GROQ_BASE_URL, GroqLLM, build_llm
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
            return httpx.Response(400, json={"error": {"message": "model has been decommissioned"}})
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
