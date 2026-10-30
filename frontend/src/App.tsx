import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Meeting, type TimeRange } from "./api";
import { ActionItems } from "./components/ActionItems";
import { AskPanel } from "./components/AskPanel";
import { MeetingView } from "./components/MeetingView";
import { formatDuration } from "./format";

type View =
  { kind: "ask" } | { kind: "actions" } | { kind: "meeting"; id: string; seek: TimeRange | null };

const SUGGESTIONS = [
  "Why did the launch date move?",
  "What did we decide about notifications?",
  "What is Aisha working on?",
  "How accurate is transaction categorization?",
];

export default function App() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [view, setView] = useState<View>({ kind: "ask" });
  const [model, setModel] = useState("");
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
    void api.health().then(
      (h) => setModel(h.model),
      () => undefined,
    );
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
        {model && <p className="model muted small">Model: {model}</p>}
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
