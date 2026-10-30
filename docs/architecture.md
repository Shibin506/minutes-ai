# Architecture and design decisions

## Data flow

```
upload ──► parse_transcript ──► merge_turns ──► chunk_segments ──► SQLite (segments, chunks, FTS5)
                                                                       │
                       background task: llm.analyze(segments) ─────────┘──► meetings.analysis (JSON)

question ──► fts_query ──► BM25 top-k chunks ──► llm.stream_answer ──► SSE to the browser
```

Uploads return immediately with `status: processing`. Analysis runs as a FastAPI background task, and the UI polls until the meeting is `ready` or `failed`. A failed analysis does not lose the meeting: the transcript is already stored and searchable, and **Re-run analysis** retries it.

## Decisions

### The model cites segment numbers, not timestamps

The transcript is rendered as `[#12 03:45] Aisha: ...` and the schema asks for `segment: int` on every action item, decision and topic. Copying a visible integer is much more reliable than producing a clock time. The server turns the number into an exact timestamp and clamps anything out of range (`clamp_segments`), so a bad reference can at worst point at a nearby line. It can never point at a time that doesn't exist.

### Structured output, not "please reply in JSON"

`client.beta.messages.parse(..., output_format=MeetingAnalysis)` constrains the response to the Pydantic schema and returns a validated object. Two failure modes are handled explicitly: `stop_reason == "refusal"` and `max_tokens` (truncated output). Both mark the meeting `failed` with a readable error instead of saving partial notes.

### Refusal fallback

Requests opt into server-side fallbacks (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`). If a safety classifier wrongly declines an ordinary meeting transcript, the API retries on a fallback model instead of failing the upload.

### Whole-transcript analysis, chunked retrieval

Analysis sends the whole meeting in one request: the model's context window holds hours of transcript, and summarizing chunk by chunk loses cross-references ("as Marcus said earlier…"). Q&A across *many* meetings can't send everything, so it retrieves excerpts first.

### BM25 (SQLite FTS5) before embeddings

Meeting questions are full of names, numbers and product terms ("Plaid", "October 29th", "dedupe"), which keyword search matches exactly. FTS5 ships with SQLite, so there is no extra service, no index to rebuild, and deletes stay consistent through triggers. The eval puts the answer in the top 3 for every question. Embeddings are on the roadmap, to be added when the eval includes paraphrased questions that BM25 misses. The eval is what should justify the extra complexity.

### Chunks follow speaker turns and overlap

Chunks are packed from whole turns (never mid-sentence) up to ~160 words, and repeat the previous chunk's last turn. Each chunk has a clean start and end time to cite, and a question with its answer is never split across two chunks.

### Safe query construction

User questions are reduced to lowercase content words, each quoted as an FTS5 phrase and joined with `OR`. FTS5 syntax in the input (`AND`, `NEAR(...)`, `*`, quotes) can't change the query or cause errors. This is covered by a test.

### Demo mode

`DemoLLM` implements the same interface with keyword rules: commitments ("I'll…", "let me…") become action items owned by the speaker, "we decided…/agreed" become decisions, and answers quote the best-matching lines. It is clearly labelled in the UI. It lets anyone run the project without an API key, and lets CI exercise the full HTTP flow.

## Testing strategy

| Layer | How |
|---|---|
| Parsing | Unit tests per format, including cue settings, voice tags, missing timestamps and bad input |
| Chunking | Coverage (every turn in some chunk), overlap, oversized turns |
| Store | Ranking, per-meeting filter, delete removes rows from the FTS index, query escaping |
| Claude backend | Fake client: checks request shape, schema, fallback opt-in, refusal/truncation errors, streaming |
| API | `TestClient` end to end with demo mode: upload → background analysis → notes, action items, SSE answers, 404/422 paths |
| Retrieval | `evals/run_retrieval_eval.py` with a regression threshold in `tests/test_retrieval_eval.py` |
| Frontend | Vitest for citation splitting, SSE parsing and formatting |
