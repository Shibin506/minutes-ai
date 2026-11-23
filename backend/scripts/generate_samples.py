"""Writes the extra sample meetings (fictional, LLM-written from the storyline below) as VTT files.

The first three samples were written by hand; these continue the same project so search,
the eval and "what changed" have more than a few chunks to work with. Run once:

    GROQ_API_KEY=... python scripts/generate_samples.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from minutes.llm import GroqLLM

SAMPLES = Path(__file__).resolve().parents[2] / "samples"
CAST = (
    "Priya (product manager, runs the meetings), Marcus (head of sales and partnerships), "
    "Aisha (backend engineer, owns bank sync and Plaid), Jordan (ML engineer, owns the "
    "transaction categorizer and the weekly summary email), Elena (product designer)"
)
CONTEXT = (
    "Smart Budgets is a budgeting feature in a personal-finance app. Earlier meetings: Sep 8 "
    "sprint planning (launch set for Oct 15, Plaid chosen at 30 cents per connected account per "
    "month, rule-based categorizer 70% accurate); Sep 22 design review (three-step onboarding, "
    "LLM categorizer at 93%, shared budgets for couples moved to Q1, notifications left open); "
    "Oct 6 go/no-go (duplicate-transaction bug on Chase and Wells Fargo after token refresh, "
    "launch moved to Oct 29, dedupe fix due Oct 20, email-only weekly summaries for launch, push "
    "in November)."
)

MEETINGS = [
    (
        "2026-10-13",
        "dedupe-fix-status",
        "Status check on the dedupe fix",
        [
            "Aisha: the dedupe-on-Plaid-transaction-ID fix is in review; the backfill dry run found a new edge case: Capital One pending transactions change ID when they post, so dedupe also has to match on amount, date and merchant",
            "Aisha now expects the fix to land October 19th, one day earlier than planned, and the backfill on the 21st",
            "Marcus: sales told 14 enterprise prospects about the Oct 29 date; two asked for a demo the week of launch",
            "Jordan: weekly summary email template is drafted; open rate target is 40 percent",
            "decision: freeze new features until launch; only launch blockers get merged",
            "action: Elena to write the support FAQ about duplicate transactions by Friday",
        ],
    ),
    (
        "2026-10-27",
        "launch-readiness",
        "Final launch readiness check",
        [
            "Aisha: dedupe fix shipped October 19th; backfill corrected 41,000 duplicate transactions; zero duplicates in the last five days",
            "decision: go for launch on Wednesday October 29th at 9am Pacific",
            "Marcus: support team trained; a status page will be live on launch day",
            "Jordan: categorizer at 95 percent on production traffic",
            "risk: Plaid rate limits at launch spike; Aisha will add a queue with exponential backoff",
            "action: Priya to post the launch announcement in the company all-hands channel on the 29th",
            "action: Jordan to watch categorization error dashboards for the first 48 hours",
        ],
    ),
    (
        "2026-11-10",
        "launch-retrospective",
        "Launch retrospective",
        [
            "results: 12,400 signups in the first week, app store rating 4.6, onboarding completion 81 percent",
            "weekly summary email open rate 52 percent, above the 40 percent target",
            "what went wrong: the Plaid rate-limit queue backed up for two hours on launch morning; Aisha will raise worker concurrency and add an alert",
            "Marcus renegotiated Plaid: price drops from 30 cents to 22 cents per connected account per month above 50,000 accounts",
            "top user request is still shared budgets for couples (mentioned in 31 percent of feedback)",
            "decision: write blameless postmortems for any incident over 30 minutes",
            "action: Elena to run five user interviews on shared budgets before the planning meeting",
        ],
    ),
    (
        "2026-11-17",
        "premium-tier-pricing",
        "Premium tier and pricing",
        [
            "proposal: Smart Budgets Premium with shared budgets, custom categories and export to CSV",
            "debate between 3.99 and 4.99 per month; Marcus cites competitor pricing at 5.99",
            "decision: Premium at 4.99 per month or 49 per year; the free tier keeps everything that exists today",
            "decision: Premium launches February 1st",
            "Jordan: custom categories need the categorizer to learn per-user rules; estimates three weeks",
            "open question: do couples on shared budgets each need Premium or does one subscription cover both? left for the next meeting",
        ],
    ),
    (
        "2026-12-01",
        "q1-planning",
        "Q1 planning",
        [
            "push notifications shipped December 8th is the plan (mobile team now free), replacing email-only for budget alerts; the weekly email stays",
            "shared budgets: engineering estimate is ten weeks because of permissions and sync conflicts, so it cannot make February 1st; decision: Premium launches Feb 1 without shared budgets, and shared budgets move to Q2 (target April 15th)",
            "answer to the open question: one Premium subscription covers both partners in a shared budget",
            "hiring: two backend engineers approved; Aisha will interview",
            "action: Marcus to update the sales deck to remove shared budgets from the February launch by Friday",
        ],
    ),
    (
        "2026-12-08",
        "security-review",
        "Security and compliance review",
        [
            "Plaid access tokens are encrypted at rest but with an app-level key; decision: move to a cloud KMS with key rotation every 90 days, Aisha owns it, due January 15th",
            "SOC 2 Type I audit targeted for March 31st; auditor chosen: Prescient Assurance",
            "penetration test before Premium launch, vendor Cobalt, week of January 12th",
            "Jordan: categorizer prompts never include account numbers, only merchant names and amounts",
            "open question: how long to keep raw transaction data after an account is disconnected; legal to advise",
        ],
    ),
    (
        "2026-12-15",
        "shared-budgets-kickoff",
        "Shared budgets kickoff",
        [
            "Elena's five interviews: couples want separate private spending plus shared categories, not one merged budget",
            "design: each person keeps a private budget; shared categories like rent and groceries are visible to both",
            "change: with the two new hires starting January 5th, engineering now estimates seven weeks, so shared budgets move back to Q1 with a target of March 15th instead of April 15th",
            "Aisha: the owner list added to the budget table in September means no data migration is needed",
            "decision: invite-by-email flow for the partner, invites expire after 7 days",
            "action: Elena to deliver shared-budget designs by January 9th; Aisha to write the permissions spec by January 12th",
        ],
    ),
]

PROMPT = """Write a realistic meeting transcript for this meeting of a small product team.

Team: {cast}
Background: {context}
Meeting: {title} on {date}.
It must include all of these points, said naturally by the right people (use exact numbers and dates):
{points}

Rules: 22-32 turns, each line "Name: what they said" (1-3 sentences), conversational but on-topic,
Priya opens and ends with a short recap. Use only the five names above. No stage directions, no headings."""


def to_vtt(lines: list[str]) -> str:
    out, t = ["WEBVTT", ""], 2.0
    for i, line in enumerate(lines, start=1):
        name, text = line.split(":", 1)
        dur = max(2.5, len(text.split()) / 2.6)
        stamp = lambda s: f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"  # noqa: E731
        out += [
            str(i),
            f"{stamp(t)} --> {stamp(t + dur)}",
            f"<v {name.strip()}>{text.strip()}</v>",
            "",
        ]
        t += dur + 0.6
    return "\n".join(out)


def main() -> None:
    llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"])
    for date, slug, title, points in MEETINGS:
        path = SAMPLES / f"{date}_smart-budgets-{slug}.vtt"
        if path.exists():
            continue
        text = llm.complete(
            "You write natural, specific meeting dialogue.",
            PROMPT.format(
                cast=CAST,
                context=CONTEXT,
                title=title,
                date=date,
                points="\n".join(f"- {p}" for p in points),
            ),
            max_tokens=6000,
        )
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if re.match(r"^(Priya|Marcus|Aisha|Jordan|Elena):\s*\S", ln.strip())
        ]
        path.write_text(to_vtt(lines))
        print(path.name, len(lines), "turns")


if __name__ == "__main__":
    main()
