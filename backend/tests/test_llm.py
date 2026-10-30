"""ClaudeLLM is tested against a fake client: no network, no API key."""

from types import SimpleNamespace

import pytest

from minutes.llm import FALLBACK_BETA, AnalysisError, ClaudeLLM, DemoLLM, clamp_segments
from minutes.schemas import ActionItem, Decision, MeetingAnalysis, Topic
from minutes.store import SearchHit
from minutes.transcript import Segment, parse_transcript

SEGMENTS = [Segment("Ann", 0, 5, "I'll send the deck by Friday."), Segment("Ben", 6, 9, "Great.")]


def analysis(**overrides) -> MeetingAnalysis:
    data = dict(
        title="Sync",
        summary="s",
        key_points=[],
        decisions=[Decision(decision="d", segment=99)],
        action_items=[ActionItem(task="t", owner="Ann", due="", segment=-3)],
        open_questions=[],
        topics=[Topic(name="x", first_segment=1, last_segment=0)],
    )
    data.update(overrides)
    return MeetingAnalysis(**data)


class FakeStream:
    def __init__(self, chunks, stop_reason="end_turn"):
        self.text_stream = iter(chunks)
        self._stop = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop)


class FakeMessages:
    def __init__(self, parse_result=None, stream=None):
        self.calls = []
        self._parse_result = parse_result
        self._stream = stream

    def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        return self._parse_result

    def stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        return self._stream


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_clamp_segments_keeps_references_in_range():
    fixed = clamp_segments(analysis(), n_segments=2)
    assert fixed.decisions[0].segment == 1
    assert fixed.action_items[0].segment == 0
    assert fixed.topics[0].last_segment >= fixed.topics[0].first_segment


def test_analyze_sends_structured_request_with_fallback():
    messages = FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=analysis()))
    llm = ClaudeLLM("claude-opus-5-5", client=fake_client(messages))
    result = llm.analyze("Weekly sync", SEGMENTS)

    kind, kwargs = messages.calls[0]
    assert kind == "parse"
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_format"] is MeetingAnalysis
    assert kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert "[#0 00:00] Ann: I'll send the deck" in kwargs["messages"][0]["content"]
    assert result.decisions[0].segment == 1  # clamped


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(stop_reason="refusal", parsed_output=None),
        SimpleNamespace(stop_reason="max_tokens", parsed_output=None),
    ],
)
def test_analyze_raises_on_refusal_or_truncation(response):
    llm = ClaudeLLM(client=fake_client(FakeMessages(response)))
    with pytest.raises(AnalysisError):
        llm.analyze("x", SEGMENTS)


def test_stream_answer_yields_text_and_cites_sources():
    hit = SearchHit("m1", "Planning", 0, 0, 30, "Ann: launch is June", 1.0)
    messages = FakeMessages(stream=FakeStream(["Launch ", "is June [S1]."]))
    llm = ClaudeLLM(client=fake_client(messages))
    assert "".join(llm.stream_answer("When is launch?", [hit])) == "Launch is June [S1]."
    prompt = messages.calls[0][1]["messages"][0]["content"]
    assert '<source id="S1" meeting="Planning" time="00:00-00:30">' in prompt


def test_stream_answer_reports_refusal():
    messages = FakeMessages(stream=FakeStream([], stop_reason="refusal"))
    text = "".join(ClaudeLLM(client=fake_client(messages)).stream_answer("q", []))
    assert "declined" in text


def test_demo_llm_extracts_commitments_and_decisions(sample_paths):
    path = next(p for p in sample_paths if "go-no-go" in p.name)
    segments = parse_transcript(path.read_text(), path.name)
    result = DemoLLM().analyze("Go/no-go", segments)
    owners = {item.owner for item in result.action_items}
    assert {"Aisha", "Jordan", "Marcus"} <= owners
    assert any("October 29th" in d.decision for d in result.decisions)
    assert all(0 <= item.segment < len(segments) for item in result.action_items)


def test_demo_llm_answer_without_hits():
    assert "couldn't find" in "".join(DemoLLM().stream_answer("anything", []))
