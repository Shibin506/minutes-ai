export type Status = "processing" | "ready" | "failed";

export interface ActionItem {
  task: string;
  owner: string;
  due: string;
  segment: number;
}

export interface Analysis {
  title: string;
  summary: string;
  key_points: string[];
  decisions: { decision: string; segment: number }[];
  action_items: ActionItem[];
  open_questions: string[];
  topics: { name: string; first_segment: number; last_segment: number }[];
}

export interface Meeting {
  id: string;
  title: string;
  source_name: string;
  created_at: string;
  duration: number;
  status: Status;
  error: string | null;
  analysis: Analysis | null;
}

export interface Segment {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

export interface MeetingDetail extends Meeting {
  segments: Segment[];
}

export interface TimeRange {
  start: number;
  end: number;
}

export interface Source {
  meeting_id: string;
  meeting_title: string;
  chunk_index: number;
  start: number;
  end: number;
  text: string;
  score: number;
}

export interface TrackedActionItem extends ActionItem {
  meeting: string;
  meeting_id: string;
  start: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const api = {
  health: () => request<{ status: string; model: string }>("/health"),
  meetings: () => request<Meeting[]>("/meetings"),
  meeting: (id: string) => request<MeetingDetail>(`/meetings/${id}`),
  actionItems: () => request<TrackedActionItem[]>("/action-items"),
  loadSamples: () => request<Meeting[]>("/samples", { method: "POST" }),
  remove: (id: string) => request<void>(`/meetings/${id}`, { method: "DELETE" }),
  reanalyze: (id: string) => request(`/meetings/${id}/analyze`, { method: "POST" }),
  notesUrl: (id: string) => `/api/meetings/${id}/notes.md`,
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return request<Meeting>("/meetings", { method: "POST", body });
  },
};

export interface SSEEvent {
  event: string;
  data: unknown;
}

/** Split a server-sent-events buffer into complete events and the unfinished tail. */
export function parseSSE(buffer: string): { events: SSEEvent[]; rest: string } {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events = blocks
    .map((block) => {
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      return data.length ? { event, data: JSON.parse(data.join("\n")) } : null;
    })
    .filter((e): e is SSEEvent => e !== null);
  return { events, rest };
}

export interface AskHandlers {
  onSources: (sources: Source[]) => void;
  onDelta: (text: string) => void;
  onError: (message: string) => void;
}

/** POST a question and stream the answer. Resolves when the stream ends. */
export async function ask(
  question: string,
  meetingId: string | null,
  handlers: AskHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, meeting_id: meetingId }),
    signal,
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    handlers.onError(
      typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`,
    );
    return;
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const parsed = parseSSE(buffer + value);
    buffer = parsed.rest;
    for (const { event, data } of parsed.events) {
      if (event === "sources") handlers.onSources(data as Source[]);
      else if (event === "delta") handlers.onDelta(data as string);
      else if (event === "error") handlers.onError(data as string);
    }
  }
}
