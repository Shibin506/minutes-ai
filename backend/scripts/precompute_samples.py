"""Analyze the sample meetings once and save the results next to them, so the hosted demo
(and "Load sample meetings") shows real model output instantly, with no model call.

    GROQ_API_KEY=... python scripts/precompute_samples.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from minutes.changes import what_changed
from minutes.llm import GroqLLM
from minutes.service import MeetingService
from minutes.store import Store

SAMPLES = Path(__file__).resolve().parents[2] / "samples"
OUT = SAMPLES / "analyses"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # Free-tier Groq allows 8,000 tokens a minute: wait out rate limits patiently.
    llm = GroqLLM(api_key=os.environ["GROQ_API_KEY"], retry_delays=(20, 30, 45, 60, 60, 60))
    store = Store(":memory:")
    service = MeetingService(store, llm)
    for path in sorted(SAMPLES.glob("*.vtt")):
        meeting_id = service.ingest_text(path.read_text(), path.name)
        saved = OUT / f"{path.stem}.json"
        if saved.exists():
            from minutes.schemas import MeetingAnalysis

            store.set_analysis(meeting_id, MeetingAnalysis.model_validate_json(saved.read_text()))
            continue
        service.analyze(meeting_id)
        m = store.get_meeting(meeting_id)
        if m["status"] != "ready":
            raise SystemExit(f"{path.name}: {m['error']}")
        saved.write_text(json.dumps(m["analysis"], indent=1, ensure_ascii=False) + "\n")
        print("analyzed", path.name, "-", m["analysis"]["title"], flush=True)

    changes = what_changed(store, llm)
    by_id = {m["id"]: m["source_name"] for m in store.list_meetings()}
    for thread in changes["threads"]:
        for entry in thread["history"]:
            entry["source_name"] = by_id[entry.pop("meeting_id")]
            entry.pop("meeting_title", None)
    (OUT / "_changes.json").write_text(json.dumps(changes, indent=1, ensure_ascii=False) + "\n")
    print("what changed:", len(changes["threads"]), "threads")
    for t in changes["threads"]:
        print(f"  [{t['status']}] {t['topic']}: {t['summary']}")


if __name__ == "__main__":
    main()
