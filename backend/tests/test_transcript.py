import json

import pytest

from minutes.transcript import (
    Segment,
    TranscriptError,
    format_timestamp,
    merge_turns,
    parse_timestamp,
    parse_transcript,
)

VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
<v Alice Chen>Let's start with the roadmap.</v>

2
00:00:04.500 --> 00:00:06.000
<v Alice Chen>Then hiring.</v>

3
00:00:07.000 --> 00:00:09.250
<v Bob>Sounds good.</v>
"""


def test_parse_timestamp_formats():
    assert parse_timestamp("00:00:01.500") == 1.5
    assert parse_timestamp("01:02:03,250") == 3723.25
    assert parse_timestamp("2:05") == 125


def test_format_timestamp():
    assert format_timestamp(65) == "01:05"
    assert format_timestamp(3725.9) == "1:02:05"


def test_vtt_voice_tags_and_turn_merging():
    segments = parse_transcript(VTT, "call.vtt")
    assert [s.speaker for s in segments] == ["Alice Chen", "Bob"]
    assert segments[0].text == "Let's start with the roadmap. Then hiring."
    assert (segments[0].start, segments[0].end) == (1.0, 6.0)
    assert segments[1].end == 9.25


def test_vtt_with_speaker_prefix_and_cue_settings():
    vtt = "WEBVTT\n\n00:01.000 --> 00:03.000 align:start\nDana: Ship it.\n"
    [seg] = parse_transcript(vtt, "x.vtt")
    assert (seg.speaker, seg.text, seg.start) == ("Dana", "Ship it.", 1.0)


def test_srt():
    srt = (
        "1\n00:00:01,000 --> 00:00:02,000\nEve: Hello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nFrank: Hi\n"
    )
    segments = parse_transcript(srt, "x.srt")
    assert [(s.speaker, s.text) for s in segments] == [("Eve", "Hello"), ("Frank", "Hi")]


def test_plain_text_with_timestamps():
    text = "[00:10] Alice: First point.\n[01:30] Bob: Second point.\n"
    segments = parse_transcript(text, "notes.txt")
    assert [s.start for s in segments] == [10, 90]
    assert segments[0].end <= segments[1].start


def test_plain_text_without_timestamps_estimates_time():
    text = "Alice: one two three four five\nBob: six seven\ncontinued line"
    segments = parse_transcript(text, "notes.txt")
    assert [s.speaker for s in segments] == ["Alice", "Bob"]
    assert segments[1].text == "six seven continued line"
    assert segments[1].start >= segments[0].end


def test_json_list_and_dict():
    data = [{"speaker": "A", "start": 0, "end": 2, "text": "hi"}, {"text": "  "}]
    assert len(parse_transcript(json.dumps(data), "t.json")) == 1
    wrapped = json.dumps({"segments": data})
    assert parse_transcript(wrapped, "t.json")[0].speaker == "A"


@pytest.mark.parametrize(
    "content,name", [("", "a.txt"), ("WEBVTT\n\n", "a.vtt"), ("{bad json", "a.json")]
)
def test_empty_or_invalid_raises(content, name):
    with pytest.raises(TranscriptError):
        parse_transcript(content, name)


def test_merge_turns_respects_gap_and_length():
    segs = [
        Segment("A", 0, 1, "one"),
        Segment("A", 10, 11, "far later"),
        Segment("A", 11.5, 12, "close"),
    ]
    merged = merge_turns(segs, max_gap=2.0)
    assert [s.text for s in merged] == ["one", "far later close"]
    long = [Segment("A", i, i + 1, "w " * 80) for i in range(3)]
    assert len(merge_turns(long, max_words=120)) == 3


def test_samples_parse(sample_paths):
    for path in sample_paths:
        segments = parse_transcript(path.read_text(), path.name)
        assert len(segments) > 15
        assert all(s.speaker != "Speaker" for s in segments)
