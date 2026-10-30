"""Retrieval eval: does search put the evidence for each question near the top?

A hit at rank r means the r-th chunk comes from the expected meeting *and*
contains the evidence phrase. Runs offline in under a second:

    python -m evals.run_retrieval_eval [--chunk-words 160] [--k 5]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from minutes.service import MeetingService
from minutes.store import Store

HERE = Path(__file__).resolve().parent
SAMPLES = HERE.parents[1] / "samples"


@dataclass
class Report:
    n: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    misses: list[str]


class _NoLLM:
    name = "none"


def run(chunk_words: int = 160, k: int = 5) -> Report:
    store = Store(":memory:")
    service = MeetingService(store, _NoLLM(), chunk_words=chunk_words)
    ids = {
        p.stem: service.ingest_text(p.read_text(), p.name) for p in sorted(SAMPLES.glob("*.vtt"))
    }
    cases = json.loads((HERE / "cases.json").read_text())

    ranks: list[int | None] = []
    misses = []
    for case in cases:
        meeting_id = next(mid for stem, mid in ids.items() if stem.endswith(case["meeting"]))
        hits = store.search(case["question"], k=k)
        rank = next(
            (
                i
                for i, h in enumerate(hits, start=1)
                if h.meeting_id == meeting_id and case["evidence"].lower() in h.text.lower()
            ),
            None,
        )
        ranks.append(rank)
        if rank is None:
            misses.append(case["question"])

    n = len(cases)
    return Report(
        n=n,
        hit_at_1=sum(r == 1 for r in ranks) / n,
        hit_at_3=sum(r is not None and r <= 3 for r in ranks) / n,
        mrr=sum(1 / r for r in ranks if r) / n,
        misses=misses,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = run(args.chunk_words, args.k)
    print(f"cases   {report.n}")
    print(f"hit@1   {report.hit_at_1:.2f}")
    print(f"hit@3   {report.hit_at_3:.2f}")
    print(f"MRR     {report.mrr:.2f}")
    for question in report.misses:
        print(f"miss    {question}")


if __name__ == "__main__":
    main()
