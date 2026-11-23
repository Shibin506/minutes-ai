import numpy as np

from minutes.retrieval import LSAEmbedder, Retriever, rrf, tokens
from minutes.service import MeetingService
from minutes.store import SearchHit, Store


def hit(mid, idx):
    return SearchHit(mid, mid, idx, 0, 1, "", 0.0)


def test_rrf_rewards_agreement_between_rankings():
    a = [hit("m", 1), hit("m", 2), hit("m", 3)]
    b = [hit("m", 3), hit("m", 1), hit("m", 4)]
    fused = [(h.meeting_id, h.chunk_index) for h in rrf([a, b])]
    assert fused[:2] == [("m", 1), ("m", 3)]  # in both lists, near the top
    assert set(fused) == {("m", 1), ("m", 2), ("m", 3), ("m", 4)}


def test_lsa_links_words_that_share_context():
    docs = [
        "launch date moved because the sync bug delays launch",
        "go-live date moved because the sync bug delays go-live",
        "pricing for the premium tier is five dollars",
        "premium tier pricing compared with competitors",
    ]
    lsa = LSAEmbedder(dims=2)
    lsa.fit(docs)
    v = lsa.embed(docs + ["when is go-live"])
    q = v[-1]
    assert np.argmax(v[:4] @ q) in (0, 1)  # the launch passages, even the one without "go-live"
    assert (v[0] @ q) > (v[2] @ q)


def test_tokens_stem_and_drop_stopwords():
    assert tokens("The launches were delayed by pricing") == ["launch", "delay", "pric"]


def test_hybrid_search_refreshes_when_meetings_change():
    store = Store(":memory:")

    class NoLLM:
        name, demo = "none", True

    svc = MeetingService(store, NoLLM())
    r = Retriever(store)
    svc.ingest_text(
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:05.000\n<v Ann>We picked Plaid for bank sync.</v>\n",
        "a.vtt",
    )
    assert r.search("bank sync vendor", k=3)[0].text.startswith("Ann")
    svc.ingest_text(
        "WEBVTT\n\n1\n00:00:00.000 --> 00:00:05.000\n<v Ben>Premium costs five dollars.</v>\n",
        "b.vtt",
    )
    assert any("Premium" in h.text for h in r.search("premium price", k=3))


def test_diversify_caps_passages_per_meeting():
    from minutes.retrieval import diversify

    hits = [SearchHit("a", "A", i, 0, 1, "t", 1.0) for i in range(5)] + [
        SearchHit("b", "B", 0, 0, 1, "t", 0.5)
    ]
    out = diversify(hits, k=4, per_meeting=3)
    assert [(h.meeting_id, h.chunk_index) for h in out] == [("a", 0), ("a", 1), ("a", 2), ("b", 0)]
    # Not enough other meetings: the overflow fills the rest.
    assert len(diversify(hits, k=6, per_meeting=3)) == 6
