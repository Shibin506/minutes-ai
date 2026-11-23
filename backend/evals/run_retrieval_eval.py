"""Retrieval eval: does search put the evidence for each question near the top?

A hit at rank r means the r-th chunk comes from the expected meeting *and* contains the
evidence phrase. Two question sets over the ten sample meetings:

* `cases.json`: questions that reuse the transcript's wording (keyword-friendly);
* `cases_paraphrase.json`: the same facts asked in other words ("bank-data vendor" for Plaid,
  "go-live" for launch), which is where pure keyword search struggles.

    python -m evals.run_retrieval_eval                      # BM25 vs hybrid (offline, CI)
    python -m evals.run_retrieval_eval --gemini --rerank    # + hosted embeddings, LLM reranking
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from minutes.retrieval import APIEmbedder, LSAEmbedder, Retriever
from minutes.service import MeetingService
from minutes.store import Store

HERE = Path(__file__).resolve().parent
SAMPLES = HERE.parents[1] / "samples"
SETS = {"keyword": "cases.json", "paraphrase": "cases_paraphrase.json"}


@dataclass
class Report:
    n: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    misses: list[str] = field(default_factory=list)


class _NoLLM:
    name = "none"
    demo = True


def build(chunk_words: int = 160) -> tuple[Store, dict[str, str]]:
    store = Store(":memory:")
    service = MeetingService(store, _NoLLM(), chunk_words=chunk_words)
    ids = {
        p.stem: service.ingest_text(p.read_text(), p.name) for p in sorted(SAMPLES.glob("*.vtt"))
    }
    return store, ids


def score(
    retriever: Retriever, ids: dict[str, str], cases: list[dict], mode: str, k: int = 5
) -> Report:
    ranks: list[int | None] = []
    misses = []
    for case in cases:
        meeting_id = next(mid for stem, mid in ids.items() if stem.endswith(case["meeting"]))
        hits = retriever.search(case["question"], k=k, mode=mode)
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


def run(
    chunk_words: int = 160,
    k: int = 5,
    case_set: str = "keyword",
    mode: str = "bm25",
    embedder=None,
    llm=None,
) -> Report:
    store, ids = build(chunk_words)
    cases = json.loads((HERE / SETS[case_set]).read_text())
    return score(Retriever(store, embedder or LSAEmbedder(), llm=llm), ids, cases, mode, k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--gemini", action="store_true", help="also try Gemini embeddings (GEMINI_API_KEY)"
    )
    parser.add_argument(
        "--rerank", action="store_true", help="also try LLM reranking (GROQ_API_KEY)"
    )
    parser.add_argument("--json", type=Path, help="write the results table as JSON")
    args = parser.parse_args()

    configs: list[tuple[str, str, object, object]] = [
        ("BM25", "bm25", None, None),
        ("Hybrid: BM25 + LSA", "hybrid", None, None),
    ]
    if args.gemini:
        gem = APIEmbedder(
            "gemini-embedding-001",
            os.environ["GEMINI_API_KEY"],
            "https://generativelanguage.googleapis.com/v1beta/openai",
        )
        configs.append(("Hybrid: BM25 + Gemini embeddings", "hybrid", gem, None))
    if args.rerank:
        from minutes.llm import GroqLLM

        groq = GroqLLM("openai/gpt-oss-20b", os.environ["GROQ_API_KEY"], reasoning_effort="low")
        groq.quota_fallbacks = ()  # measure this model: wait out rate limits, don't switch
        configs.append(("Hybrid (LSA) + Groq rerank", "hybrid+rerank", None, groq))
        if args.gemini:
            configs.append(("Hybrid (Gemini) + Groq rerank", "hybrid+rerank", gem, groq))

    rows = []
    print(f"{'method':38s} {'set':11s} {'n':>3s} {'hit@1':>6s} {'hit@3':>6s} {'MRR':>5s}")
    store, ids = build(args.chunk_words)
    all_cases = {name: json.loads((HERE / f).read_text()) for name, f in SETS.items()}
    if args.gemini:  # one batched call for every question, instead of one call each
        gem.embed([c["question"] for cases in all_cases.values() for c in cases])
    for label, mode, embedder, llm in configs:
        retriever = Retriever(store, embedder or LSAEmbedder(), llm=llm)
        for case_set in SETS:
            print(f"  running {label} / {case_set}...", flush=True)
            r = score(retriever, ids, all_cases[case_set], mode, args.k)
            rows.append(
                {
                    "method": label,
                    "set": case_set,
                    "n": r.n,
                    "hit@1": r.hit_at_1,
                    "hit@3": r.hit_at_3,
                    "mrr": r.mrr,
                    "misses": r.misses,
                }
            )
            cols = f"{r.n:3d} {r.hit_at_1:6.2f} {r.hit_at_3:6.2f} {r.mrr:5.2f}"
            print(f"{label:38s} {case_set:11s} {cols}")
            if retriever.rerank_errors:
                print(
                    f"  ! {retriever.rerank_errors} rerank calls failed (fell back to hybrid order)"
                )
                retriever.rerank_errors = 0
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
