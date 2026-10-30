import { useRef, useState } from "react";
import { ask, type Source, type TimeRange } from "../api";
import { formatTime, splitCitations } from "../format";

interface Props {
  meetingId: string | null;
  placeholder: string;
  suggestions?: string[];
  onOpenSource: (meetingId: string, range: TimeRange) => void;
}

export function AskPanel({ meetingId, placeholder, suggestions = [], onOpenSource }: Props) {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  async function submit(q: string) {
    if (q.trim().length < 2 || busy) return;
    abort.current?.abort();
    abort.current = new AbortController();
    setAsked(q);
    setAnswer("");
    setSources([]);
    setError(null);
    setBusy(true);
    try {
      await ask(
        q,
        meetingId,
        {
          onSources: setSources,
          onDelta: (text) => setAnswer((prev) => prev + text),
          onError: setError,
        },
        abort.current.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const openSource = (index: number) => {
    const source = sources[index - 1];
    if (source) onOpenSource(source.meeting_id, source);
  };

  return (
    <section className="ask">
      <form
        className="ask-form"
        onSubmit={(e) => {
          e.preventDefault();
          void submit(question);
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={placeholder}
          aria-label="Question"
        />
        <button type="submit" disabled={busy || question.trim().length < 2}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </form>

      {!asked && suggestions.length > 0 && (
        <div className="suggestions">
          {suggestions.map((s) => (
            <button
              key={s}
              className="chip"
              onClick={() => {
                setQuestion(s);
                void submit(s);
              }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {asked && (
        <div className="answer" aria-live="polite">
          <p className="answer-question">{asked}</p>
          {error && <p className="error">{error}</p>}
          <div className="answer-text">
            {splitCitations(answer).map((part, i) =>
              part.kind === "text" ? (
                <span key={i}>{part.text}</span>
              ) : (
                <button
                  key={i}
                  className="cite"
                  title={sources[part.index - 1]?.meeting_title}
                  onClick={() => openSource(part.index)}
                >
                  {part.index}
                </button>
              ),
            )}
            {busy && <span className="cursor" />}
          </div>
          {sources.length > 0 && (
            <details className="sources">
              <summary>{sources.length} excerpts searched</summary>
              <ol>
                {sources.map((s) => (
                  <li key={`${s.meeting_id}-${s.chunk_index}`}>
                    <button className="link" onClick={() => onOpenSource(s.meeting_id, s)}>
                      {s.meeting_title} · {formatTime(s.start)}
                    </button>
                    <p>{s.text.slice(0, 220)}…</p>
                  </li>
                ))}
              </ol>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
