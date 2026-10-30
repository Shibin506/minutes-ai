from evals.run_retrieval_eval import run


def test_retrieval_quality_does_not_regress():
    report = run(chunk_words=160, k=5)
    assert report.n >= 20
    assert report.hit_at_3 == 1.0, report.misses
    assert report.hit_at_1 >= 0.7
