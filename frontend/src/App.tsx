import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Health, type Meeting, type TimeRange } from "./api";
import { ActionItems } from "./components/ActionItems";
import { AskPanel } from "./components/AskPanel";
import { LiveMeeting } from "./components/LiveMeeting";
import { MeetingView } from "./components/MeetingView";
import { WhatChanged } from "./components/WhatChanged";
import { formatDuration } from "./format";

type View =
  | { kind: "ask" }
  | { kind: "actions" }
  | { kind: "changes" }
  | { kind: "live" }
  | { kind: "meeting"; id: string; seek: TimeRange | null };

const SUGGESTIONS = [
  "Why did the launch date move?",
  "When will shared budgets ship, and why did it change?",
  "What did we get from the bank-data vendor?",
  "What is Aisha working on?",
];

export default function App() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [view, setView] = useState<View>({ kind: "ask" });
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // Bumped whenever meeting data changes, so open views refetch.
  const [version, setVersion] = useState(0);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const list = await api.meetings();
    setMeetings(list);
    setVersion((v) => v + 1);
    return list;
  }, []);

  useEffect(() => {
    void refresh().catch((e: Error) => setError(`Backend unreachable: ${e.message}`));
    void api.health().then(setHealth, () => undefined);
  }, [refresh]);

  // Poll while any meeting is still being analyzed.
  const processing = meetings.some((m) => m.status === "processing");
  useEffect(() => {
    if (!processing) return;
    const timer = setInterval(() => void refresh(), 2000);
    return () => clearInterval(timer);
  }, [processing, refresh]);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    await run(async () => {
      let last: Meeting | null = null;
      for (const file of Array.from(files)) last = await api.upload(file);
      if (last) setView({ kind: "meeting", id: last.id, seek: null });
    });
  }

  const openSource = (id: string, seek: TimeRange) => setView({ kind: "meeting", id, seek });

  return (
    <div
      className="app"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        void upload(e.dataTransfer.files);
      }}
    >
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">M</span> Minutes
        </div>
        <button className="primary" disabled={busy} onClick={() => fileInput.current?.click()}>
          {busy ? "Uploading…" : "Upload transcript"}
        </button>
        <input
          ref={fileInput}
          type="file"
          hidden
          multiple
          accept=".vtt,.srt,.txt,.json,.mp3,.wav,.m4a,.mp4,.webm"
          onChange={(e) => {
            void upload(e.target.files);
            e.target.value = "";
          }}
        />
        <nav>
          <button
            className={view.kind === "ask" ? "nav active" : "nav"}
            onClick={() => setView({ kind: "ask" })}
          >
            Ask all meetings
          </button>
          <button
            className={view.kind === "actions" ? "nav active" : "nav"}
            onClick={() => setView({ kind: "actions" })}
          >
            Action items
          </button>
          <button
            className={view.kind === "changes" ? "nav active" : "nav"}
            onClick={() => setView({ kind: "changes" })}
          >
            What changed
          </button>
          <button
            className={view.kind === "live" ? "nav active" : "nav"}
            onClick={() => setView({ kind: "live" })}
          >
            <span className="rec-dot small" /> Live meeting
          </button>
        </nav>
        <h3>Meetings</h3>
        <ul className="meeting-list">
          {meetings.map((m) => (
            <li key={m.id}>
              <button
                className={view.kind === "meeting" && view.id === m.id ? "nav active" : "nav"}
                onClick={() => setView({ kind: "meeting", id: m.id, seek: null })}
              >
                <span className={`dot ${m.status}`} title={m.status} />
                <span className="meeting-name">{m.title}</span>
                <span className="muted small">{formatDuration(m.duration)}</span>
              </button>
            </li>
          ))}
        </ul>
        {meetings.length === 0 && (
          <button className="secondary" disabled={busy} onClick={() => void run(api.loadSamples)}>
            Load sample meetings
          </button>
        )}
        {health && (
          <p className="model muted small">
            Model: {health.model}
            <br />
            Search: {health.retrieval === "bm25" ? "keyword (BM25)" : "hybrid: BM25 + vectors"}
            {health.retrieval.includes("rerank") ? " + LLM rerank" : ""}
          </p>
        )}
        {health?.hosted_demo && (
          <p className="demo-note small">
            Live demo with fictional sample meetings. Uploads work but aren't kept: storage resets.
          </p>
        )}
      </aside>

      <main className="main">
        {error && <p className="banner error">{error}</p>}
        {view.kind === "ask" && (
          <div className="page narrow">
            <h1>Ask your meetings anything</h1>
            <p className="muted">
              Answers come only from your transcripts, with numbered citations that jump to the
              exact moment something was said.
            </p>
            {meetings.length === 0 ? (
              <p className="empty">
                Upload a transcript (.vtt, .srt, .txt, .json) or drop it anywhere on the page, or
                load the sample meetings to try it out.
              </p>
            ) : (
              <AskPanel
                meetingId={null}
                placeholder="e.g. What did we decide about the launch date?"
                suggestions={SUGGESTIONS}
                onOpenSource={openSource}
              />
            )}
          </div>
        )}
        {view.kind === "actions" && <ActionItems version={version} onOpenSource={openSource} />}
        {view.kind === "changes" && <WhatChanged version={version} onOpenSource={openSource} />}
        {view.kind === "live" && (
          <LiveMeeting
            serverTranscription={!!health?.live}
            onSaved={(m) => {
              void refresh();
              setView({ kind: "meeting", id: m.id, seek: null });
            }}
          />
        )}
        {view.kind === "meeting" && (
          <MeetingView
            meetingId={view.id}
            seek={view.seek}
            version={version}
            onOpenSource={openSource}
            onDeleted={() => {
              setView({ kind: "ask" });
              void refresh();
            }}
          />
        )}
      </main>
    </div>
  );
}
