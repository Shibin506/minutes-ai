"""Language-model backends.

`ClaudeLLM` is the default implementation and `GeminiLLM` an alternative (it has a
free tier). `DemoLLM` is a small rule-based stand-in so the app (and CI) runs end
to end without an API key.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from typing import Protocol

import anthropic
import httpx

from . import prompts
from .schemas import ActionItem, Decision, MeetingAnalysis, Topic
from .store import STOPWORDS, SearchHit
from .transcript import Segment

# Server-side refusal fallback: if a safety classifier declines, the API re-runs
# the request on Anthropic's recommended fallback model instead of failing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnalysisError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def analyze(self, title_hint: str, segments: list[Segment]) -> MeetingAnalysis: ...

    def stream_answer(self, question: str, hits: list[SearchHit]) -> Iterator[str]: ...


def clamp_segments(analysis: MeetingAnalysis, n_segments: int) -> MeetingAnalysis:
    """Keep every segment reference inside the transcript."""
    last = max(n_segments - 1, 0)

    def fix(i: int) -> int:
        return min(max(i, 0), last)

    for item in analysis.action_items:
        item.segment = fix(item.segment)
    for decision in analysis.decisions:
        decision.segment = fix(decision.segment)
    for topic in analysis.topics:
        topic.first_segment = fix(topic.first_segment)
        topic.last_segment = max(topic.first_segment, fix(topic.last_segment))
    return analysis


class ClaudeLLM:
    def __init__(self, model: str = "claude-opus-5-5", client: anthropic.Anthropic | None = None):
        self.name = model
        self.model = model
        self.client = client or anthropic.Anthropic()

    def analyze(self, title_hint: str, segments: list[Segment]) -> MeetingAnalysis:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=prompts.ANALYSIS_SYSTEM,
            messages=[{"role": "user", "content": prompts.analysis_request(title_hint, segments)}],
            output_format=MeetingAnalysis,
        )
        if response.stop_reason == "refusal":
            raise AnalysisError("The model declined to analyze this transcript.")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise AnalysisError("The analysis was cut off before it finished.")
        return clamp_segments(response.parsed_output, len(segments))

    def stream_answer(self, question: str, hits: list[SearchHit]) -> Iterator[str]:
        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=4000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            system=prompts.ANSWER_SYSTEM,
            messages=[{"role": "user", "content": prompts.answer_request(question, hits)}],
        ) as stream:
            yield from stream.text_stream
            if stream.get_final_message().stop_reason == "refusal":
                yield "\n\n_The model declined to answer this question._"


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _inline_refs(schema: dict) -> dict:
    """Resolve local $refs: Gemini's structured output does not accept $defs."""
    defs = schema.get("$defs", {})

    def walk(node, is_properties=False):
        if isinstance(node, dict):
            if "$ref" in node and not is_properties:
                return walk(defs[node["$ref"].split("/")[-1]])
            if is_properties:  # keys here are field names (a field may be called "title")
                return {k: walk(v) for k, v in node.items()}
            return {
                k: walk(v, is_properties=(k == "properties"))
                for k, v in node.items()
                if k not in ("$defs", "title")
            }
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)


class GeminiLLM:
    """Gemini through its OpenAI-compatible endpoint: a strict JSON schema for the
    analysis and server-sent events for streamed answers."""

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        api_key: str = "",
        reasoning_effort: str = "",
        base_url: str = GEMINI_BASE_URL,
        http: httpx.Client | None = None,
    ):
        self.name = model
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(timeout=180)
        self.schema = _inline_refs(MeetingAnalysis.model_json_schema())

    def _body(self, system: str, user: str, **extra) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key or 'unset'}"}

    def analyze(self, title_hint: str, segments: list[Segment]) -> MeetingAnalysis:
        body = self._body(
            prompts.ANALYSIS_SYSTEM,
            prompts.analysis_request(title_hint, segments),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "meeting_analysis", "strict": True, "schema": self.schema},
            },
        )
        r = self.http.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=body)
        if r.status_code >= 400:
            raise AnalysisError(f"Gemini returned HTTP {r.status_code}: {r.text[:300]}")
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") not in ("stop", None):
            raise AnalysisError(f"The analysis stopped early ({choice.get('finish_reason')}).")
        analysis = MeetingAnalysis.model_validate_json(choice["message"]["content"])
        return clamp_segments(analysis, len(segments))

    def stream_answer(self, question: str, hits: list[SearchHit]) -> Iterator[str]:
        body = self._body(
            prompts.ANSWER_SYSTEM, prompts.answer_request(question, hits), stream=True
        )
        with self.http.stream(
            "POST", f"{self.base_url}/chat/completions", headers=self._headers(), json=body
        ) as r:
            if r.status_code >= 400:
                r.read()
                raise RuntimeError(f"Gemini returned HTTP {r.status_code}: {r.text[:300]}")
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[6:]).get("choices", []):
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        yield text


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_COMMITMENT = re.compile(
    r"\b(i'll|i will|we'll|we will|i can take|i'm going to|action item|let me)\b", re.I
)
_DECISION = re.compile(
    r"\b(we decided|decision is|let's go with|we're going with|agreed|we'll go with|final call)\b",
    re.I,
)
_DUE = re.compile(
    r"\b(by|before|on|until)\s+((?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|"
    r"tomorrow|end of (?:day|week|month|sprint|q[1-4])|eod|eow|\w+ \d{1,2}(?:st|nd|rd|th)?))\b",
    re.I,
)


def _sentences(segments: list[Segment]) -> Iterator[tuple[int, Segment, str]]:
    for i, seg in enumerate(segments):
        for sentence in _SENTENCE.split(seg.text):
            if sentence.strip():
                yield i, seg, sentence.strip()


def _keywords(text: str, n: int) -> list[str]:
    words = [
        w for w in re.findall(r"\b[a-z][a-z\-]{3,}\b(?!')", text.lower()) if w not in STOPWORDS
    ]
    return [w for w, _ in Counter(words).most_common(n)]


class DemoLLM:
    """Keyword heuristics. Good enough to demo the UI; not a substitute for the model."""

    name = "demo (no API key)"

    def analyze(self, title_hint: str, segments: list[Segment]) -> MeetingAnalysis:
        speakers = list(dict.fromkeys(s.speaker for s in segments))
        full_text = " ".join(s.text for s in segments)
        keywords = _keywords(full_text, 5)

        actions, decisions, questions, long_sentences = [], [], [], []
        for i, seg, sentence in _sentences(segments):
            if _COMMITMENT.search(sentence) and not sentence.endswith("?"):
                due = _DUE.search(sentence)
                actions.append(
                    ActionItem(
                        task=sentence,
                        owner=seg.speaker,
                        due=due.group(2) if due else "",
                        segment=i,
                    )
                )
            if _DECISION.search(sentence):
                decisions.append(Decision(decision=sentence, segment=i))
            if sentence.endswith("?") and len(sentence.split()) > 5:
                questions.append(sentence)
            long_sentences.append((len(sentence.split()), sentence))

        third = max(len(segments) // 3, 1)
        topics = [
            Topic(
                name=", ".join(_keywords(" ".join(s.text for s in segments[a : a + third]), 2))
                or "Discussion",
                first_segment=a,
                last_segment=min(a + third, len(segments)) - 1,
            )
            for a in range(0, len(segments), third)
        ][:3]

        return MeetingAnalysis(
            title=title_hint,
            summary=(
                f"{len(speakers)} participants ({', '.join(speakers[:5])}) discussed "
                f"{', '.join(keywords) or 'several topics'}. Running in demo mode: set "
                "ANTHROPIC_API_KEY (or GEMINI_API_KEY) for a real summary."
            ),
            key_points=[s for _, s in sorted(long_sentences, reverse=True)[:4]],
            decisions=decisions[:6],
            action_items=actions[:10],
            open_questions=questions[-4:],
            topics=topics,
        )

    def stream_answer(self, question: str, hits: list[SearchHit]) -> Iterator[str]:
        if not hits:
            yield "I couldn't find anything about that in your meetings."
            return
        terms = set(re.findall(r"\w+", question.lower())) - STOPWORDS
        yield "Demo mode (no API key) - most relevant lines from your meetings:\n\n"
        for i, hit in enumerate(hits[:3], start=1):
            lines = hit.text.splitlines()
            best = max(lines, key=lambda line: len(terms & set(re.findall(r"\w+", line.lower()))))
            yield f"- {best} [S{i}]\n"


def build_llm(settings) -> LLM:
    if settings.demo_mode:
        return DemoLLM()
    if settings.provider == "gemini":
        return GeminiLLM(
            settings.gemini_model, settings.gemini_api_key, settings.gemini_reasoning_effort
        )
    return ClaudeLLM(settings.model)
