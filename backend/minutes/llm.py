"""Language-model backends.

`ClaudeLLM` is the default implementation and `GeminiLLM` an alternative (it has a
free tier). `DemoLLM` is a small rule-based stand-in so the app (and CI) runs end
to end without an API key.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Iterator
from typing import Protocol, TypeVar

import anthropic
import httpx
from pydantic import BaseModel

from . import prompts
from .schemas import ActionItem, Decision, MeetingAnalysis, Topic
from .store import STOPWORDS, SearchHit
from .transcript import Segment

# Server-side refusal fallback: if a safety classifier declines, the API re-runs
# the request on Anthropic's recommended fallback model instead of failing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnalysisError(RuntimeError):
    pass


T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    name: str
    #: True for the rule-based stand-in; features then use their own heuristics.
    demo: bool

    def analyze(self, title_hint: str, segments: list[Segment]) -> MeetingAnalysis: ...

    def stream_answer(self, question: str, hits: list[SearchHit]) -> Iterator[str]: ...

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str: ...

    def complete_json(
        self, system: str, user: str, schema: type[T], light: bool = False, max_tokens: int = 3000
    ) -> T: ...


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

    demo = False

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    def complete_json(
        self, system: str, user: str, schema: type[T], light: bool = False, max_tokens: int = 3000
    ) -> T:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=8000,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if response.parsed_output is None:
            raise AnalysisError("The model returned no structured output.")
        return response.parsed_output


# ---------------------------------------------------------------------------
# OpenAI-compatible providers: Gemini and Groq
# ---------------------------------------------------------------------------

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Providers answer 5xx "high demand" in short bursts; a few quick retries ride them out.
RETRY_STATUSES = {429, 500, 502, 503}
GEMINI_RETRY_STATUSES = RETRY_STATUSES
RETRY_DELAYS = (2.0, 5.0, 10.0)
GEMINI_RETRY_DELAYS = RETRY_DELAYS


#: Some models write citations as 【S1】 and use narrow no-break spaces and hyphens; the UI
#: links "[S1]" and people search for "March 15". Per character, so it is safe on a stream.
PLAIN_TEXT = str.maketrans({"【": "[", "】": "]", "\u202f": " ", "\u00a0": " ", "\u2011": "-"})


def _wait(r: httpx.Response, default: float) -> float:
    """How long to back off: the provider's retry-after when it gives one (rate limits)."""
    try:
        return min(float(r.headers.get("retry-after", default)), 60.0)
    except ValueError:
        return default


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


def _strict(schema: dict) -> dict:
    """Strict JSON-schema mode (Groq, OpenAI) wants every object closed and fully required."""
    if isinstance(schema, dict):
        out = {k: _strict(v) for k, v in schema.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(schema, list):
        return [_strict(v) for v in schema]
    return schema


def json_schema_for(model: type[BaseModel], strict: bool) -> dict:
    schema = _inline_refs(model.model_json_schema())
    return _strict(schema) if strict else schema


class OpenAICompatLLM:
    """A chat-completions API: strict JSON schema for structured output, server-sent events for
    streamed answers, retries on overload, and an optional fallback to a current model."""

    demo = False
    base_url_default = GEMINI_BASE_URL
    strict_schema = False

    def __init__(
        self,
        model: str,
        api_key: str = "",
        reasoning_effort: str = "",
        base_url: str | None = None,
        http: httpx.Client | None = None,
        retry_delays: tuple[float, ...] = RETRY_DELAYS,
    ):
        self.name = model
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.base_url = (base_url or self.base_url_default).rstrip("/")
        self.http = http or httpx.Client(timeout=180)
        self.retry_delays = retry_delays
        self.schema = json_schema_for(MeetingAnalysis, self.strict_schema)

    # -- plumbing -----------------------------------------------------------

    def _body(self, system: str, user: str, effort: str | None = None, **extra) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **extra,
        }
        effort = effort if effort is not None else self.reasoning_effort
        if effort:
            body["reasoning_effort"] = effort
        return body

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key or 'unset'}"}

    def _post(self, body: dict) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"
        r = self.http.post(url, headers=self._headers(), json=body)
        for delay in self.retry_delays:
            if r.status_code not in RETRY_STATUSES or self._quota_fallbacks(r):
                break  # a daily limit won't clear by waiting: go straight to another model
            time.sleep(_wait(r, delay))
            r = self.http.post(url, headers=self._headers(), json=body)
        if r.status_code in (400, 404) and self._fallback_model(r):
            body = {**body, "model": self.model}
            r = self.http.post(url, headers=self._headers(), json=body)
        # Out of quota for a long time (a daily limit): try the provider's other models.
        for model in self._quota_fallbacks(r):
            r = self.http.post(url, headers=self._headers(), json=self._for_model(body, model))
            if r.status_code != 429:
                break
        return r

    def _for_model(self, body: dict, model: str) -> dict:
        """The request body for `model` (a fallback may need different settings)."""
        return {**body, "model": model}

    def _fallback_model(self, r: httpx.Response) -> bool:
        """Override to switch to a current model when the configured one is retired."""
        return False

    #: Other models to try when this one is out of quota (e.g. a free tier's daily token limit).
    quota_fallbacks: tuple[str, ...] = ()

    def _quota_fallbacks(self, r: httpx.Response) -> list[str]:
        if r.status_code != 429 or _wait(r, 0) < 30 and "per day" not in r.text.lower():
            return []
        return [m for m in self.quota_fallbacks if m != self.model]

    def _content(self, r: httpx.Response, what: str) -> str:
        if r.status_code >= 400:
            raise AnalysisError(f"{self.provider} returned HTTP {r.status_code}: {r.text[:300]}")
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") not in ("stop", None):
            raise AnalysisError(f"The {what} stopped early ({choice.get('finish_reason')}).")
        return choice["message"].get("content") or ""

    @property
    def provider(self) -> str:
        return type(self).__name__.removesuffix("LLM")

    def _json(
        self,
        system: str,
        user: str,
        model: type[T],
        schema: dict,
        name: str,
        effort=None,
        max_tokens: int = 3000,
    ) -> T:
        body = self._body(
            system,
            user,
            effort=effort,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        )
        return model.model_validate_json(self._content(self._post(body), name.replace("_", " ")))

    # -- the LLM protocol ---------------------------------------------------

    def analyze(self, title_hint: str, segments: list[Segment]) -> MeetingAnalysis:
        args = (
            prompts.ANALYSIS_SYSTEM,
            prompts.analysis_request(title_hint, segments),
            MeetingAnalysis,
            self.schema,
            "meeting_analysis",
        )
        # Free tiers count the reserved output against a per-minute token budget, so reserve
        # what an analysis needs, not the model's maximum.
        try:
            analysis = self._json(*args, max_tokens=6000)
        except AnalysisError as exc:
            if "length" not in str(exc):
                raise
            # Long reasoning ran out of room: try once more, thinking less.
            analysis = self._json(*args, effort=self.light_effort or "low", max_tokens=6000)
        return clamp_segments(analysis, len(segments))

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        body = self._body(system, user, effort=self.light_effort, max_tokens=max_tokens)
        return self._content(self._post(body), "reply").strip()

    def complete_json(
        self, system: str, user: str, schema: type[T], light: bool = False, max_tokens: int = 3000
    ) -> T:
        return self._json(
            system,
            user,
            schema,
            json_schema_for(schema, self.strict_schema),
            schema.__name__,
            effort=self.light_effort if light else None,
            max_tokens=max_tokens,
        )

    #: Reasoning effort for short replies (answers, reranking, judging); None = provider default.
    light_effort: str | None = None

    def stream_answer(self, question: str, hits: list[SearchHit]) -> Iterator[str]:
        body = self._body(
            prompts.ANSWER_SYSTEM,
            prompts.answer_request(question, hits),
            effort=self.light_effort,
            stream=True,
        )
        models = [self.model]
        waits = list(self.retry_delays[:2])  # short per-minute limits: wait, then retry
        attempt = -1
        while attempt + 1 < len(models):
            attempt += 1
            model = models[attempt]
            with self.http.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._for_model(body, model),
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    if attempt == 0 and (more := self._quota_fallbacks(r)):
                        models += more  # out of daily quota: answer with another model
                        continue
                    if r.status_code in RETRY_STATUSES and waits:
                        time.sleep(_wait(r, waits.pop(0)))
                        attempt -= 1  # same model again
                        continue
                    if r.status_code == 429 and attempt < len(models) - 1:
                        continue
                    raise RuntimeError(
                        f"{self.provider} returned HTTP {r.status_code}: {r.text[:300]}"
                    )
                for line in r.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    for choice in json.loads(line[6:]).get("choices", []):
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield text.translate(PLAIN_TEXT)
                return


class GeminiLLM(OpenAICompatLLM):
    """Gemini through its OpenAI-compatible endpoint (it has a free tier)."""

    base_url_default = GEMINI_BASE_URL

    def __init__(self, model: str = "gemini-3.5-flash", api_key: str = "", **kw):
        super().__init__(model, api_key, **kw)


GROQ_PREFERRED = (r"gpt-oss-120b", r"qwen3", r"gpt-oss-20b", r"llama")


class GroqLLM(OpenAICompatLLM):
    """Groq: open-weight models served very fast. Strict JSON schema for the analysis; if the
    configured model has been retired, it switches to a current one from Groq's model list,
    and if it is out of daily quota, a request falls back to another model with its own quota."""

    base_url_default = GROQ_BASE_URL
    quota_fallbacks = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
    strict_schema = True
    light_effort = "low"

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: str = "",
        reasoning_effort: str = "medium",
        quota_fallbacks: tuple[str, ...] | None = None,
        **kw,
    ):
        super().__init__(model, api_key, reasoning_effort, **kw)
        if quota_fallbacks is not None:
            self.quota_fallbacks = quota_fallbacks

    def _for_model(self, body: dict, model: str) -> dict:
        body = super()._for_model(body, model)
        # Qwen reasons at length even on "low"; for a short call, "none" is what was meant.
        if "qwen" in model and body.get("reasoning_effort") == "low":
            body["reasoning_effort"] = "none"
        return body

    def _fallback_model(self, r: httpx.Response) -> bool:
        try:
            err = r.json().get("error", {})
        except ValueError:
            return False
        code = str(err.get("code", "")).lower()
        message = str(err.get("message", "")).lower()
        retired = code in {"model_decommissioned", "model_not_found"} or any(
            k in message for k in ("decommissioned", "does not exist", "model not found")
        )
        if not retired:
            return False
        try:
            listing = self.http.get(f"{self.base_url}/models", headers=self._headers())
            ids = [m["id"] for m in listing.json().get("data", [])]
        except Exception:  # noqa: BLE001 - keep the original error
            return False
        for pattern in GROQ_PREFERRED:
            for model_id in ids:
                if (
                    re.search(pattern, model_id)
                    and model_id != self.model
                    and "guard" not in model_id
                ):
                    self.model = self.name = model_id
                    return True
        return False


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
    demo = True

    def complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        raise NotImplementedError("demo mode has no language model")

    def complete_json(
        self, system: str, user: str, schema: type[T], light: bool = False, max_tokens: int = 3000
    ) -> T:
        raise NotImplementedError("demo mode has no language model")

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


def build_reranker(settings, llm: LLM) -> LLM | None:
    """A small, fast model for reranking (its own rate-limit bucket on Groq); else the main one."""
    if getattr(llm, "demo", False):
        return None
    if settings.provider == "groq":
        # Reranking is an optimization: fail fast (one short retry, a 15 s timeout) rather
        # than hold up an answer waiting out a rate limit.
        return GroqLLM(
            settings.groq_rerank_model,
            settings.groq_api_key,
            reasoning_effort="low",
            http=httpx.Client(timeout=15),
            retry_delays=(2.0,),
            # No fallback model: on the retrieval eval a small non-reasoning model (Qwen 3.8
            # 27B) re-ranked worse than no re-ranking at all, so keep the hybrid order instead.
            quota_fallbacks=(),
        )
    return llm


def build_llm(settings) -> LLM:
    if settings.demo_mode:
        return DemoLLM()
    if settings.provider == "gemini":
        return GeminiLLM(
            settings.gemini_model,
            settings.gemini_api_key,
            reasoning_effort=settings.gemini_reasoning_effort,
        )
    if settings.provider == "groq":
        return GroqLLM(settings.groq_model, settings.groq_api_key)
    return ClaudeLLM(settings.model)
