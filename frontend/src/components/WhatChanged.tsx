import { useEffect, useState } from "react";
import { api, type Changes, type ChangeStatus, type TimeRange } from "../api";
import { formatTime } from "../format";

interface Props {
  version: number;
  onOpenSource: (meetingId: string, range: TimeRange) => void;
}

const LABEL: Record<ChangeStatus, string> = {
  changed: "Changed",
  reversed: "Reversed",
  consistent: "Held",
};

function shortDate(iso: string): string {
  return new Date(`${iso}T12:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function WhatChanged({ version, onOpenSource }: Props) {
  const [data, setData] = useState<Changes | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"moved" | "all">("moved");

  useEffect(() => {
    setData(null);
    setError(null);
    api.changes().then(setData, (e: Error) => setError(e.message));
  }, [version]);

  const threads = (data?.threads ?? []).filter(
    (t) => filter === "all" || t.status !== "consistent",
  );

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>What changed</h1>
          <p className="muted">
            Decisions followed across meetings: dates that slipped, prices and scope that moved,
            calls that were reversed. Every entry links to the moment it was said.
          </p>
        </div>
        <div className="segmented" role="group" aria-label="Show">
          <button className={filter === "moved" ? "on" : ""} onClick={() => setFilter("moved")}>
            Changed
          </button>
          <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")}>
            All threads
          </button>
        </div>
      </header>
      {error && <p className="banner error">{error}</p>}
      {!data && !error && <p className="muted">Comparing your meetings…</p>}
      {data && data.meetings < 2 && (
        <p className="empty">Upload at least two meetings to see how decisions evolve.</p>
      )}
      {data && data.meetings >= 2 && threads.length === 0 && (
        <p className="empty">Nothing has changed between meetings so far.</p>
      )}
      <ol className="threads" data-testid="threads">
        {threads.map((t, i) => (
          <li key={i} className={`thread ${t.status}`}>
            <div className="thread-head">
              <span className={`status-pill ${t.status}`}>{LABEL[t.status]}</span>
              <h2>{t.topic}</h2>
            </div>
            <p className="thread-summary">{t.summary}</p>
            <ol className="timeline">
              {t.history.map((e) => (
                <li key={e.id}>
                  <span className="when">{shortDate(e.date)}</span>
                  <div>
                    <p>{e.statement}</p>
                    <button
                      className="link small"
                      onClick={() =>
                        onOpenSource(e.meeting_id, { start: e.start ?? 0, end: e.start ?? 0 })
                      }
                    >
                      {e.meeting_title}
                      {e.start !== null && ` · ${formatTime(e.start)}`}
                    </button>
                  </div>
                </li>
              ))}
            </ol>
          </li>
        ))}
      </ol>
      {data && data.threads.length > 0 && (
        <p className="muted small">
          {data.method === "heuristic"
            ? "Demo mode: threads matched by shared words; set an API key for model-based tracking."
            : `Tracked by ${data.method} across ${data.meetings} meetings.`}
        </p>
      )}
    </div>
  );
}
