"""Drive Minutes' MCP server over stdio, the way Claude Desktop or Cursor would.

    cd backend && python examples/mcp_client.py

Uses the same database as the web app (MINUTES_DB_PATH).
"""

import json
import os
import re
import sys

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters


def best_line(text, query):
    """The speaker turn in a passage that shares the most words with the query."""
    turns = re.split(r"\n+|(?<=[.!?]) (?=[A-Z][a-z]+: )", text)
    words = set(re.findall(r"[a-z]+", query.lower()))
    return max(turns, key=lambda t: len(words & set(re.findall(r"[a-z]+", t.lower())))).strip()


def show(result):
    data = result.structured_content
    data = data.get("result", data) if isinstance(data, dict) else data
    return data if data is not None else json.loads(result.content[0].text)


async def main():
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "minutes.mcp_server"], env={**os.environ}
    )
    async with Client(params) as c:
        tools = (await c.list_tools()).tools
        print("tools:", ", ".join(t.name for t in tools))
        print()
        query = "dedupe bug in bank sync"
        print(f'> search_meetings(query="{query}", limit=2)')
        for h in show(await c.call_tool("search_meetings", {"query": query, "limit": 2})):
            print(f"  {h['meeting']}  [{h['time']}]")
            print("    " + best_line(h["text"], query)[:120])
        print()
        print('> list_action_items(owner="Aisha")')
        for a in show(await c.call_tool("list_action_items", {"owner": "Aisha"}))[:5]:
            task = a["task"] if len(a["task"]) <= 64 else a["task"][:63] + "…"
            print(f"  {a['date']}  {task:64s}  due {a['due'] or '-'}")


anyio.run(main)
