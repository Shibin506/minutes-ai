"""Minutes as an MCP server, so AI assistants (Claude Desktop, Claude Code, Cursor...) can use
your meetings as tools: search them, ask cited questions, list action items, see what changed.

    python -m minutes.mcp_server            # stdio, for a local assistant

Claude Desktop config (claude_desktop_config.json):

    {"mcpServers": {"minutes": {
        "command": "python", "args": ["-m", "minutes.mcp_server"],
        "cwd": "/path/to/minutes-ai/backend",
        "env": {"MINUTES_DB_PATH": "/path/to/minutes-ai/backend/data/minutes.db"}}}}

It reads the same database as the web app, so meetings you upload there show up here.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from .changes import meeting_date, what_changed
from .config import Settings
from .llm import LLM, build_llm
from .retrieval import LSAEmbedder, Retriever
from .service import to_markdown
from .store import Store
from .transcript import format_timestamp

INSTRUCTIONS = """\
Tools over a team's meeting transcripts. Use search_meetings or ask_meetings for questions about
what was said or decided; cite the meeting title and timestamp from the results. Use
list_action_items for who owes what, and what_changed for decisions that moved over time."""


def build_server(store: Store, llm: LLM, retriever: Retriever | None = None) -> MCPServer:
    retriever = retriever or Retriever(store, LSAEmbedder(), llm=llm)
    server = MCPServer("minutes", instructions=INSTRUCTIONS)

    def titled(meeting_id: str) -> dict:
        meeting = store.get_meeting(meeting_id)
        if meeting is None:
            raise ValueError(f"No meeting with id {meeting_id!r}. Call list_meetings for ids.")
        return meeting

    @server.tool()
    def list_meetings() -> list[dict]:
        """All meetings, newest first: id, title, date, duration in minutes, status."""
        return [
            {
                "id": m["id"],
                "title": m["title"],
                "date": meeting_date(m).isoformat(),
                "minutes": round(m["duration"] / 60, 1),
                "status": m["status"],
            }
            for m in sorted(store.list_meetings(), key=meeting_date, reverse=True)
        ]

    @server.tool()
    def get_meeting_notes(meeting_id: str) -> str:
        """A meeting's notes as Markdown: summary, decisions, action items, open questions."""
        return to_markdown(titled(meeting_id), store.get_segments(meeting_id))

    @server.tool()
    def search_meetings(query: str, meeting_id: str | None = None, limit: int = 6) -> list[dict]:
        """Find transcript passages about a topic (hybrid keyword + semantic search)."""
        hits = retriever.search(query, k=max(1, min(limit, 20)), meeting_id=meeting_id)
        return [
            {
                "meeting": h.meeting_title,
                "meeting_id": h.meeting_id,
                "time": f"{format_timestamp(h.start)}-{format_timestamp(h.end)}",
                "text": h.text,
            }
            for h in hits
        ]

    @server.tool()
    def ask_meetings(question: str, meeting_id: str | None = None) -> dict:
        """Answer a question from the transcripts, with [S#] citations that refer to `sources`."""
        hits = retriever.search(question, k=8, meeting_id=meeting_id)
        answer = "".join(llm.stream_answer(question, hits))
        return {
            "answer": answer,
            "sources": [
                {"id": f"S{i}", "meeting": h.meeting_title, "time": format_timestamp(h.start)}
                for i, h in enumerate(hits, start=1)
            ],
        }

    @server.tool()
    def list_action_items(owner: str | None = None) -> list[dict]:
        """Every action item across meetings (optionally one person's), newest meeting first."""
        items = []
        for m in sorted(store.list_meetings(), key=meeting_date, reverse=True):
            for item in (m["analysis"] or {}).get("action_items", []):
                if owner and owner.lower() not in item["owner"].lower():
                    continue
                items.append(
                    {
                        "owner": item["owner"],
                        "task": item["task"],
                        "due": item["due"],
                        "meeting": m["title"],
                        "date": meeting_date(m).isoformat(),
                    }
                )
        return items

    @server.tool()
    def what_changed_across_meetings() -> list[dict]:
        """Decisions tracked across meetings: what moved (dates, prices, scope), flipped or held."""
        result = what_changed(store, llm)
        return [
            {
                "topic": t["topic"],
                "status": t["status"],
                "summary": t["summary"],
                "history": [
                    f"{e['date']} {e['meeting_title']}: {e['statement']}" for e in t["history"]
                ],
            }
            for t in result["threads"]
        ]

    return server


def main() -> None:
    settings = Settings.from_env()
    build_server(Store(settings.db_path), build_llm(settings)).run("stdio")


if __name__ == "__main__":
    main()
