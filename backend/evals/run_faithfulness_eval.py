"""Faithfulness eval: is every cited sentence in an answer actually backed by the passages it cites?

For each eval question: retrieve (hybrid), answer with citations, split the answer into claims,
then a judge checks each cited claim against exactly the passages it cites. The judge is a
different model from the one answering (no self-grading).

Reported per run:
  supported / partial / unsupported  share of cited claims
  uncited                            claims with no [S#] at all
  answered with the fact             the expected evidence appears in the answer (a correctness
                                     proxy: right fact, not just a faithful non-answer)

    python -m evals.run_faithfulness_eval --demo              # offline: rule-based answers + judge
    python -m evals.run_faithfulness_eval --limit 30   # Groq answers, a second Groq model judges
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from minutes.llm import DemoLLM
from minutes.retrieval import LSAEmbedder, Retriever, tokens

from .run_retrieval_eval import HERE, SETS, build

CITE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into claims with the source numbers each cites: a bullet is one claim,
    prose is split into sentences (a citation stays with the sentence it follows)."""
    parts: list[str] = []
    for line in answer.splitlines():
        line = line.strip()
        if re.match(r"^[-*•]\s", line):
            parts.append(line)
        else:
            parts += re.split(r"(?<=[.!?])\s+(?!\[S\d)|(?<=\])\s+(?=[A-Z])", line)
    out = []
    for part in parts:
        text = part.strip(" -*•\t")
        if len(text) < 12 or text.endswith(":"):
            continue
        ids = [int(x.strip()[1:]) for group in CITE.findall(text) for x in group.split(",")]
        out.append((CITE.sub("", text).strip(), ids))
    return out


class Verdict(BaseModel):
    claim: int
    verdict: Literal["supported", "partial", "unsupported"]


class Judgement(BaseModel):
    verdicts: list[Verdict]


JUDGE_SYSTEM = """\
You check whether claims are supported by the meeting-transcript passages they cite.
For each numbered claim you get the exact passages it cites. Answer per claim:
  supported   - every part of the claim is stated in or directly implied by those passages
  partial     - the main point is supported but a detail (a number, date, name) is not
  unsupported - the passages do not say this
Judge only against the given passages, not general knowledge."""


@dataclass
class Totals:
    questions: int = 0
    claims: int = 0
    cited: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    with_fact: int = 0
    examples: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        c = max(self.cited, 1)
        return {
            "questions": self.questions,
            "claims": self.claims,
            "cited_claims": self.cited,
            "supported": self.supported / c,
            "partial": self.partial / c,
            "unsupported": self.unsupported / c,
            "uncited": (self.claims - self.cited) / max(self.claims, 1),
            "answered_with_fact": self.with_fact / max(self.questions, 1),
        }


def judge_heuristic(claim: str, sources: list[str]) -> str:
    words = set(tokens(claim))
    seen = set(tokens(" ".join(sources)))
    overlap = len(words & seen) / max(len(words), 1)
    return "supported" if overlap >= 0.6 else "partial" if overlap >= 0.3 else "unsupported"


def run(answerer, judge, limit: int | None = None, k: int = 8) -> Totals:
    store, ids = build()
    retriever = Retriever(store, LSAEmbedder(), llm=answerer)
    cases = [c for name in SETS for c in json.loads((HERE / SETS[name]).read_text())]
    cases = cases[:limit] if limit else cases
    t = Totals()
    for case in cases:
        hits = retriever.search(case["question"], k=k)
        answer = "".join(answerer.stream_answer(case["question"], hits))
        t.questions += 1
        t.with_fact += case["evidence"].lower() in answer.lower()
        found = claims(answer)
        cited = [(i, c, ids_) for i, (c, ids_) in enumerate(found, start=1) if ids_]
        t.claims += len(found)
        t.cited += len(cited)
        if not cited:
            continue
        if judge is None:
            verdicts = {
                i: judge_heuristic(c, [hits[n - 1].text for n in ids_ if 0 < n <= len(hits)])
                for i, c, ids_ in cited
            }
        else:
            prompt = "\n\n".join(
                f"Claim {i}: {c}\n"
                + "\n".join(
                    f"<passage>{hits[n - 1].text}</passage>" for n in ids_ if 0 < n <= len(hits)
                )
                for i, c, ids_ in cited
            )
            got = judge.complete_json(JUDGE_SYSTEM, prompt, Judgement).verdicts
            verdicts = {v.claim: v.verdict for v in got}
        for i, c, _ in cited:
            v = verdicts.get(i, "unsupported")
            setattr(t, v, getattr(t, v) + 1)
            if v != "supported" and len(t.examples) < 8:
                t.examples.append({"question": case["question"], "claim": c, "verdict": v})
    return t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--demo", action="store_true", help="offline: rule-based answerer and judge"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.demo:
        answerer, judge, label = DemoLLM(), None, "demo answerer, word-overlap judge"
    else:
        from minutes.llm import GroqLLM

        key = os.environ["GROQ_API_KEY"]
        answerer = GroqLLM(api_key=key)
        judge = GroqLLM(model="qwen/qwen3.8-27b", api_key=key, reasoning_effort="")
        label = f"answers: {answerer.model}, judge: {judge.model}"
    t = run(answerer, judge, args.limit)
    result = {"setup": label, **t.as_dict(), "non_supported_examples": t.examples}
    print(json.dumps(result, indent=1))
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
