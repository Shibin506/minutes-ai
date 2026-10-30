"""Language-model backends.

`ClaudeLLM` is the real implementation. `DemoLLM` is a small rule-based stand-in
so the app (and CI) runs end to end without an API key.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from typing import Protocol

import anthropic

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
                "ANTHROPIC_API_KEY for a real summary."
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


def build_llm(model: str, demo_mode: bool) -> LLM:
    return DemoLLM() if demo_mode else ClaudeLLM(model)
