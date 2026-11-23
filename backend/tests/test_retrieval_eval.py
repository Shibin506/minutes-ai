"""Offline retrieval quality floors (no API keys): CI fails if search gets worse.

Measured on the ten sample meetings; see the README for the full table, including hosted
embeddings and LLM reranking.
"""

import pytest

from evals.run_retrieval_eval import run

FLOORS = {
    ("keyword", "bm25"): 0.85,
    ("keyword", "hybrid"): 0.95,
    ("paraphrase", "bm25"): 0.60,
    ("paraphrase", "hybrid"): 0.50,
}


@pytest.mark.parametrize(("case_set", "mode"), list(FLOORS))
def test_retrieval_quality_does_not_regress(case_set, mode):
    report = run(chunk_words=160, k=5, case_set=case_set, mode=mode)
    assert report.n >= 20
    assert report.hit_at_3 >= FLOORS[(case_set, mode)], report.misses
