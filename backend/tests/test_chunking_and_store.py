from minutes.chunking import chunk_segments
from minutes.service import MeetingService, title_from_filename
from minutes.store import Store, fts_query
from minutes.transcript import Segment, parse_transcript


def segs(word_counts):
    return [
        Segment(f"S{i % 2}", i * 10.0, i * 10.0 + 5, " ".join(["w"] * n))
        for i, n in enumerate(word_counts)
    ]


def test_chunks_cover_every_segment_without_splitting_turns():
    segments = segs([50, 50, 50, 50, 50, 50, 50])
    chunks = chunk_segments(segments, max_words=120, overlap=1)
    covered = set()
    for c in chunks:
        covered.update(range(c.first_segment, c.last_segment + 1))
        assert c.start == segments[c.first_segment].start
        assert c.end == segments[c.last_segment].end
    assert covered == set(range(len(segments)))


def test_chunks_overlap_by_one_turn():
    chunks = chunk_segments(segs([50] * 6), max_words=100, overlap=1)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.first_segment == prev.last_segment


def test_oversized_turn_gets_its_own_chunk_and_terminates():
    chunks = chunk_segments(segs([500, 10, 10]), max_words=100, overlap=1)
    assert chunks[0].first_segment == chunks[0].last_segment == 0
    assert chunks[-1].last_segment == 2


def test_fts_query_is_injection_safe():
    assert fts_query('What did "Aisha" say AND NEAR(x*)?') == '"aisha" OR "say" OR "near"'
    assert fts_query("what is the") == ""


def test_title_from_filename():
    assert title_from_filename("2026-09-08_smart-budgets-sync.vtt") == "Smart budgets sync"
    assert title_from_filename(".vtt") == "Untitled meeting"


class NoLLM:
    name = "none"


def test_search_ranks_the_right_meeting(sample_paths):
    store = Store(":memory:")
    service = MeetingService(store, NoLLM(), chunk_words=120)
    ids = {p.name: service.ingest_text(p.read_text(), p.name) for p in sample_paths}

    hits = store.search("duplicate transactions bug Chase", k=3)
    assert hits[0].meeting_id == ids["2026-10-06_smart-budgets-launch-go-no-go.vtt"]
    assert "Chase" in hits[0].text
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)

    only = ids["2026-09-08_smart-budgets-sprint-planning.vtt"]
    assert {h.meeting_id for h in store.search("Plaid", meeting_id=only)} == {only}


def test_delete_removes_meeting_from_index():
    store = Store(":memory:")
    service = MeetingService(store, NoLLM())
    meeting_id = service.ingest_text("Ann: the zebra project is late", "a.txt")
    assert store.search("zebra")
    assert store.delete_meeting(meeting_id)
    assert store.search("zebra") == []
    assert store.get_meeting(meeting_id) is None
    assert not store.delete_meeting(meeting_id)


def test_segments_round_trip():
    store = Store(":memory:")
    service = MeetingService(store, NoLLM())
    text = "[00:05] Ann: hello there\n[00:09] Ben: hi"
    meeting_id = service.ingest_text(text, "a.txt")
    assert store.get_segments(meeting_id) == parse_transcript(text, "a.txt")
    assert store.get_meeting(meeting_id)["status"] == "processing"
