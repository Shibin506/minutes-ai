import { useEffect, useMemo, useState } from "react";
import { api, type TimeRange, type TrackedActionItem } from "../api";
import { formatTime } from "../format";

interface Props {
  version: number;
  onOpenSource: (meetingId: string, range: TimeRange) => void;
}

export function ActionItems({ version, onOpenSource }: Props) {
  const [items, setItems] = useState<TrackedActionItem[] | null>(null);
  const [owner, setOwner] = useState("everyone");

  useEffect(() => {
    api.actionItems().then(setItems, () => setItems([]));
  }, [version]);

  const owners = useMemo(() => [...new Set((items ?? []).map((i) => i.owner))].sort(), [items]);
  const shown = (items ?? []).filter((i) => owner === "everyone" || i.owner === owner);

  return (
    <div className="page">
      <header className="page-header">
        <div>
          <h1>Action items</h1>
          <p className="muted">
            Every commitment from every meeting, linked to the moment it was made.
          </p>
        </div>
        <select value={owner} onChange={(e) => setOwner(e.target.value)} aria-label="Owner">
          <option value="everyone">Everyone</option>
          {owners.map((o) => (
            <option key={o}>{o}</option>
          ))}
        </select>
      </header>
      {items === null ? (
        <p className="muted">Loading…</p>
      ) : shown.length === 0 ? (
        <p className="muted">No action items yet.</p>
      ) : (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Owner</th>
                <th>Task</th>
                <th>Due</th>
                <th>Meeting</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((item, i) => (
                <tr key={i}>
                  <td className="owner-cell">{item.owner}</td>
                  <td>{item.task}</td>
                  <td>{item.due || "—"}</td>
                  <td>
                    <button
                      className="link"
                      onClick={() =>
                        onOpenSource(item.meeting_id, { start: item.start, end: item.start })
                      }
                    >
                      {item.meeting} · {formatTime(item.start)}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
