"""Exports: action-item due dates as a calendar file, and a follow-up email draft."""

from __future__ import annotations

import calendar
import re
from datetime import UTC, date, datetime, timedelta

from pydantic import BaseModel

from .changes import meeting_date

WEEKDAYS = {d.lower(): i for i, d in enumerate(calendar.day_name)}
MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m} | {
    m.lower(): i for i, m in enumerate(calendar.month_abbr) if m
}


def parse_due(text: str, meeting_day: date) -> date | None:
    """Turn a due date as said in a meeting ("Friday", "October 20th", "end of week", "the 21st",
    "Jan 9") into a calendar date, relative to the day of the meeting. None if it isn't a date."""
    t = text.lower().strip().replace(",", " ")
    t = re.sub(r"\b(by|before|on|until|next|this|the|eod|cob)\b", " ", t).strip()
    if not t:
        return None
    if "today" in t:
        return meeting_day
    if "tomorrow" in t:
        return meeting_day + timedelta(days=1)
    if re.search(r"end of (?:the )?week|\beow\b", text.lower()):
        return meeting_day + timedelta(days=(4 - meeting_day.weekday()) % 7)
    if re.search(r"end of (?:the )?month", text.lower()):
        return meeting_day.replace(day=calendar.monthrange(meeting_day.year, meeting_day.month)[1])
    m = re.search(r"\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
    if m and m[1] in MONTHS:
        month, day = MONTHS[m[1]], int(m[2])
        year = meeting_day.year + (1 if month < meeting_day.month - 2 else 0)
        try:
            return date(year, month, day)
        except ValueError:
            return None
    for name, wd in WEEKDAYS.items():
        if re.search(rf"\b{name}\b", t):
            ahead = (wd - meeting_day.weekday()) % 7 or 7
            return meeting_day + timedelta(days=ahead)
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)", t)
    if m:
        day = int(m[1])
        month, year = meeting_day.month, meeting_day.year
        if day < meeting_day.day:
            month, year = (1, year + 1) if month == 12 else (month + 1, year)
        try:
            return date(year, month, day)
        except ValueError:
            return None
    return None


def _ics_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def action_items_ics(meetings: list[dict]) -> tuple[str, int, int]:
    """An .ics calendar with one all-day event per action item that has a parseable due date.
    Returns (calendar text, items included, items skipped)."""
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Minutes//Action items//EN",
        "CALSCALE:GREGORIAN",
    ]
    included = skipped = 0
    for m in meetings:
        if not m.get("analysis"):
            continue
        day = meeting_date(m)
        for i, item in enumerate(m["analysis"]["action_items"]):
            due = parse_due(item.get("due", ""), day) if item.get("due") else None
            if due is None:
                skipped += 1
                continue
            included += 1
            lines += [
                "BEGIN:VEVENT",
                f"UID:{m['id']}-{i}@minutes",
                f"DTSTAMP:{now}",
                f"DTSTART;VALUE=DATE:{due:%Y%m%d}",
                f"DTEND;VALUE=DATE:{due + timedelta(days=1):%Y%m%d}",
                f"SUMMARY:{_ics_text(item['owner'] + ': ' + item['task'])}",
                "DESCRIPTION:"
                + _ics_text(f"From {m['title']} ({day.isoformat()}). Due as said: {item['due']}"),
                "END:VEVENT",
            ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n", included, skipped


class FollowUp(BaseModel):
    subject: str
    body: str


FOLLOW_UP_SYSTEM = """\
You write the follow-up email a meeting organizer sends right after a meeting.
Plain text, friendly and short: one line of thanks, the decisions as a short list, the action
items as "Owner - task (due date)", then open questions if any. No markdown symbols other
than "-" for list items. Do not invent anything that is not in the notes."""


def follow_up(meeting: dict, llm) -> FollowUp:
    a = meeting["analysis"]
    if getattr(llm, "demo", True):
        lines = ["Hi all,", "", f"Thanks for joining {meeting['title']}. Here's a quick recap.", ""]
        if a["decisions"]:
            lines += ["Decisions:", *[f"- {d['decision']}" for d in a["decisions"]], ""]
        if a["action_items"]:
            lines += ["Action items:"]
            lines += [
                f"- {x['owner']} - {x['task']}" + (f" ({x['due']})" if x["due"] else "")
                for x in a["action_items"]
            ]
            lines.append("")
        if a["open_questions"]:
            lines += ["Open questions:", *[f"- {q}" for q in a["open_questions"]], ""]
        lines += ["Thanks,"]
        return FollowUp(subject=f"Recap: {meeting['title']}", body="\n".join(lines))
    notes = (
        f"Meeting: {meeting['title']} ({meeting_date(meeting).isoformat()})\n"
        f"Summary: {a['summary']}\n"
        "Decisions:\n" + "\n".join(f"- {d['decision']}" for d in a["decisions"]) + "\n"
        "Action items:\n"
        + "\n".join(
            f"- {x['owner']}: {x['task']} (due: {x['due'] or 'none'})" for x in a["action_items"]
        )
        + "\n"
        "Open questions:\n" + "\n".join(f"- {q}" for q in a["open_questions"])
    )
    return llm.complete_json(FOLLOW_UP_SYSTEM, notes, FollowUp)
