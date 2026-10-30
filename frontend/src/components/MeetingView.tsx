import { useEffect, useRef, useState } from "react";
import { api, type MeetingDetail, type TimeRange } from "../api";
import { formatDuration, formatTime } from "../format";
import { AskPanel } from "./AskPanel";

interface Props {
  meetingId: string;
  seek: TimeRange | null;
  version: number;
  onOpenSource: (meetingId: string, range: TimeRange) => void;
  onDeleted: () => void;
}

export function MeetingView({ meetingId, seek, version, onOpenSource, onDeleted }: Props) {
  const [meeting, setMeeting] = useState<MeetingDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Highlighted transcript segments, as an inclusive [first, last] index range.
  const [active, setActive] = useState<[number, number] | null>(null);
  const transcript = useRef<HTMLOListElement>(null);

  useEffect(() => {
    api.meeting(meetingId).then(setMeeting, (e: Error) => setError(e.message));
  }, [meetingId, version]);

  // Highlight the segments a citation or action item points at.
  useEffect(() => {
    if (!meeting || seek === null) return;
    const first = meeting.segments.findIndex((s) => s.end > seek.start);
    if (first < 0) return;
    const after = meeting.segments.findIndex((s, i) => i > first && s.start >= seek.end);
    jumpTo(first, after < 0 ? meeting.segments.length - 1 : after - 1);
  }, [meeting, seek]);

  function jumpTo(first: number, last = first) {
    setActive([first, Math.max(first, last)]);
    transcript.current
      ?.querySelector(`[data-index="${first}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: first === last ? "center" : "start" });
  }

  if (error) return <p className="error">{error}</p>;
  if (!meeting) return <p className="muted">Loading…</p>;

  const a = meeting.analysis;
  const at = (i: number) => (
    <button className="time" onClick={() => jumpTo(i)}>
      {formatTime(meeting.segments[i]?.start ?? 0)}
    </button>
  );
  const speakers = [...new Set(meeting.segments.map((s) => s.speaker))];

  return (
    <div className="meeting">
      <header className="meeting-header">
        <div>
          <h1>{meeting.title}</h1>
          <p className="muted">
            {formatDuration(meeting.duration)} · {speakers.join(", ")}
          </p>
        </div>
        <div className="actions">
          <a className="button secondary" href={api.notesUrl(meeting.id)} target="_blank">
            Export notes
          </a>
          <button
            className="secondary"
            onClick={() =>
              api.reanalyze(meeting.id).then(() => setMeeting({ ...meeting, status: "processing" }))
            }
          >
            Re-run analysis
          </button>
          <button
            className="danger"
            onClick={() => {
              if (confirm(`Delete "${meeting.title}"?`))
                void api.remove(meeting.id).then(onDeleted);
            }}
          >
            Delete
          </button>
        </div>
      </header>

      <div className="meeting-body">
        <div className="notes">
          {meeting.status === "processing" && <p className="banner">Analyzing the transcript…</p>}
          {meeting.status === "failed" && (
            <p className="banner error">
              Analysis failed: {meeting.error}. The transcript is still searchable.
            </p>
          )}
          {a && (
            <>
              <section>
                <h2>Summary</h2>
                <p>{a.summary}</p>
              </section>
              {a.action_items.length > 0 && (
                <section>
                  <h2>Action items</h2>
                  <ul className="items">
                    {a.action_items.map((item, i) => (
                      <li key={i}>
                        <span className="owner">{item.owner}</span>
                        <span>{item.task}</span>
                        <span className="meta">
                          {item.due && <span className="due">due {item.due}</span>}
                          {at(item.segment)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </section>
              )}
              {a.decisions.length > 0 && (
                <section>
                  <h2>Decisions</h2>
                  <ul className="items">
                    {a.decisions.map((d, i) => (
                      <li key={i}>
                        <span>{d.decision}</span>
                        {at(d.segment)}
                      </li>
                    ))}
                  </ul>
                </section>
              )}
              {a.key_points.length > 0 && (
                <section>
                  <h2>Key points</h2>
                  <ul>
                    {a.key_points.map((p, i) => (
                      <li key={i}>{p}</li>
                    ))}
                  </ul>
                </section>
              )}
              {a.open_questions.length > 0 && (
                <section>
                  <h2>Open questions</h2>
                  <ul>
                    {a.open_questions.map((q, i) => (
                      <li key={i}>{q}</li>
                    ))}
                  </ul>
                </section>
              )}
              {a.topics.length > 0 && (
                <section>
                  <h2>Topics</h2>
                  <div className="topics">
                    {a.topics.map((t, i) => (
                      <button key={i} className="chip" onClick={() => jumpTo(t.first_segment)}>
                        {formatTime(meeting.segments[t.first_segment]?.start ?? 0)} {t.name}
                      </button>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
          <section>
            <h2>Ask about this meeting</h2>
            <AskPanel
              key={meeting.id}
              meetingId={meeting.id}
              placeholder="What did we decide about…"
              onOpenSource={onOpenSource}
            />
          </section>
        </div>

        <ol className="transcript" ref={transcript} aria-label="Transcript">
          {meeting.segments.map((s, i) => (
            <li
              key={i}
              data-index={i}
              className={active && i >= active[0] && i <= active[1] ? "active" : undefined}
            >
              <div className="segment-meta">
                <strong>{s.speaker}</strong>
                <button className="time" onClick={() => setActive([i, i])}>
                  {formatTime(s.start)}
                </button>
              </div>
              <p>{s.text}</p>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
