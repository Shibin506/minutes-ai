"""The MCP server, driven by a real MCP client in-process (demo mode: no API key)."""

import json
from pathlib import Path

import anyio
from mcp import Client

from minutes.llm import DemoLLM
from minutes.mcp_server import build_server
from minutes.service import MeetingService
from minutes.store import Store

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def server_with_samples():
    store = Store(":memory:")
    service = MeetingService(store, DemoLLM())
    for p in sorted(SAMPLES.glob("*.vtt"))[:3]:
        service.analyze(service.ingest_text(p.read_text(), p.name))
    return build_server(store, DemoLLM())


def payload(result):
    if getattr(result, "structured_content", None) is not None:
        data = result.structured_content
        return data.get("result", data) if isinstance(data, dict) else data
    return json.loads(result.content[0].text)


def test_tools_over_mcp():
    async def go():
        async with Client(server_with_samples()) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == {
                "list_meetings",
                "get_meeting_notes",
                "search_meetings",
                "ask_meetings",
                "list_action_items",
                "what_changed_across_meetings",
            }

            meetings = payload(await client.call_tool("list_meetings", {}))
            assert [m["date"] for m in meetings] == ["2026-10-06", "2026-09-22", "2026-09-08"]

            notes = await client.call_tool("get_meeting_notes", {"meeting_id": meetings[0]["id"]})
            assert notes.content[0].text.startswith("# ")

            hits = payload(
                await client.call_tool("search_meetings", {"query": "Plaid pricing", "limit": 3})
            )
            assert hits and all({"meeting", "time", "text"} <= set(h) for h in hits)

            answer = payload(
                await client.call_tool("ask_meetings", {"question": "Why was the launch moved?"})
            )
            assert "[S1]" in answer["answer"] and answer["sources"][0]["id"] == "S1"

            aisha = payload(await client.call_tool("list_action_items", {"owner": "aisha"}))
            assert aisha and all(i["owner"] == "Aisha" for i in aisha)

            bad = await client.call_tool("get_meeting_notes", {"meeting_id": "nope"})
            assert bad.is_error

    anyio.run(go)
