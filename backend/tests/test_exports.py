from datetime import date

from minutes.exports import action_items_ics, follow_up, parse_due
from minutes.llm import DemoLLM

MON = date(2026, 10, 5)  # a Monday


def test_parse_due_relative_to_the_meeting_day():
    cases = {
        "Friday": date(2026, 10, 9),
        "by Monday": date(2026, 10, 12),  # next Monday, not today
        "today": MON,
        "tomorrow": date(2026, 10, 6),
        "end of week": date(2026, 10, 9),
        "end of the month": date(2026, 10, 31),
        "October 20th": date(2026, 10, 20),
        "Oct 24": date(2026, 10, 24),
        "Jan 9": date(2027, 1, 9),  # early next year, not last January
        "the 21st": date(2026, 10, 21),
        "the 2nd": date(2026, 11, 2),  # already past this month
    }
    for said, want in cases.items():
        assert parse_due(said, MON) == want, said
    for said in ("", "soon", "next sprint", "Q1"):
        assert parse_due(said, MON) is None, said


def meeting(items):
    return {
        "id": "m1",
        "title": "Go/no-go",
        "source_name": "2026-10-05_go.vtt",
        "created_at": "2026-11-01T00:00:00",
        "analysis": {
            "summary": "s",
            "decisions": [{"decision": "Move launch to Oct 29", "segment": 0}],
            "action_items": items,
            "open_questions": ["Push or email?"],
        },
    }


def test_ics_has_one_all_day_event_per_dated_item_and_escapes_text():
    m = meeting(
        [
            {
                "task": "Ship the fix; run backfill",
                "owner": "Aisha",
                "due": "October 20th",
                "segment": 0,
            },
            {"task": "Tell sales", "owner": "Marcus", "due": "", "segment": 0},
            {"task": "Email template", "owner": "Jordan", "due": "Friday", "segment": 0},
        ]
    )
    ics, included, skipped = action_items_ics([m])
    assert (included, skipped) == (2, 1)
    assert ics.count("BEGIN:VEVENT") == 2
    assert "DTSTART;VALUE=DATE:20261020" in ics and "DTEND;VALUE=DATE:20261021" in ics
    assert "SUMMARY:Aisha: Ship the fix\; run backfill" in ics
    assert ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith("END:VCALENDAR\r\n")


def test_follow_up_template_in_demo_mode():
    m = meeting([{"task": "Tell sales", "owner": "Marcus", "due": "today", "segment": 0}])
    f = follow_up(m, DemoLLM())
    assert f.subject == "Recap: Go/no-go"
    assert (
        "- Move launch to Oct 29" in f.body
        and "- Marcus - Tell sales (today)" in f.body
        and "Push or email?" in f.body
    )


def test_clean_text_drops_trailing_junk():
    from minutes.exports import clean_text

    junk = 'Best,\n[Your name]"}  }  }  }  }  }\n  }  }  }'
    assert clean_text(junk) == "Best,\n[Your name]"
    assert clean_text("Thanks,\nPriya") == "Thanks,\nPriya"
