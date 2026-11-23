import { useEffect, useRef, useState } from "react";
import { api, type Analysis, type Meeting, type Segment } from "../api";
import { formatTime } from "../format";

interface Props {
  /** Server-side transcription (Groq Whisper) is available. */
  serverTranscription: boolean;
  onSaved: (meeting: Meeting) => void;
}

const CHUNK_MS = 8000;
const NOTES_EVERY_MS = 30000;
/** RMS level below which a whole chunk counts as silence and isn't sent for transcription. */
const SILENCE_RMS = 0.01;

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult:
    | ((e: {
        resultIndex: number;
        results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal: boolean }>;
      }) => void)
    | null;
  onend: (() => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  start(): void;
  stop(): void;
};

function browserRecognition(): (new () => SpeechRecognitionLike) | null {
  const w = window as unknown as Record<string, unknown>;
  return (w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null) as
    (new () => SpeechRecognitionLike) | null;
}

function pickMime(): string {
  if (typeof MediaRecorder === "undefined") return "";
  for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

export function LiveMeeting({ serverTranscription, onSaved }: Props) {
  const [title, setTitle] = useState("Live meeting");
  const [state, setState] = useState<"idle" | "live" | "saving">("idle");
  const [segments, setSegments] = useState<Segment[]>([]);
  const [notes, setNotes] = useState<Analysis | null>(null);
  const [notesBusy, setNotesBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const stopRef = useRef<() => Promise<void>>(async () => {});
  const segsRef = useRef<Segment[]>([]);
  const transcriptEnd = useRef<HTMLLIElement>(null);

  const engine =
    serverTranscription && pickMime() ? "server" : browserRecognition() ? "browser" : null;

  useEffect(() => {
    segsRef.current = segments;
    transcriptEnd.current?.scrollIntoView({ block: "nearest" });
  }, [segments]);

  useEffect(() => () => void stopRef.current(), []);

  async function refreshNotes() {
    if (!segsRef.current.length) return;
    setNotesBusy(true);
    try {
      setNotes(await api.liveNotes(title, segsRef.current));
    } catch {
      /* the next refresh tries again */
    } finally {
      setNotesBusy(false);
    }
  }

  async function start() {
    setError(null);
    setSegments([]);
    setNotes(null);
    const t0 = performance.now();
    const pending: Promise<unknown>[] = [];
    const cleanups: (() => void)[] = [];
    let running = true;

    const tick = setInterval(() => setElapsed((performance.now() - t0) / 1000), 500);
    const notesTimer = setInterval(() => void refreshNotes(), NOTES_EVERY_MS);
    cleanups.push(
      () => clearInterval(tick),
      () => clearInterval(notesTimer),
    );

    try {
      if (engine === "server") {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        cleanups.push(() => stream.getTracks().forEach((t) => t.stop()));
        const mime = pickMime();
        // Track the loudest moment of each chunk, so silent chunks are skipped: it saves a
        // request, and Whisper tends to invent words ("Thank you.") for silence.
        const audio = new AudioContext();
        const analyser = audio.createAnalyser();
        audio.createMediaStreamSource(stream).connect(analyser);
        const buf = new Float32Array(analyser.fftSize);
        let level = { peak: 0 }; // the chunk being recorded
        const meter = setInterval(() => {
          analyser.getFloatTimeDomainData(buf);
          const rms = Math.sqrt(buf.reduce((a, v) => a + v * v, 0) / buf.length);
          level.peak = Math.max(level.peak, rms);
        }, 100);
        cleanups.push(
          () => clearInterval(meter),
          () => void audio.close(),
        );
        // One recorder per chunk, so every chunk is a complete, playable file.
        const chunks: Segment[][] = [];
        let index = 0;
        const record = () => {
          if (!running) return;
          const i = index++;
          const offset = (performance.now() - t0) / 1000;
          const rec = new MediaRecorder(stream, { mimeType: mime });
          const parts: Blob[] = [];
          const chunkLevel = (level = { peak: 0 });
          rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
          rec.onstop = () => {
            if (chunkLevel.peak < SILENCE_RMS) return;
            const blob = new Blob(parts, { type: mime });
            const prompt = segsRef.current
              .map((s) => s.text)
              .join(" ")
              .slice(-300);
            const p = api.liveTranscribe(blob, offset, prompt).then(
              (segs) => {
                chunks[i] = segs;
                setSegments(chunks.flat());
              },
              (e: Error) => setError(e.message),
            );
            pending.push(p);
          };
          rec.start();
          setTimeout(() => {
            if (rec.state !== "inactive") rec.stop();
            record();
          }, CHUNK_MS);
          cleanups.push(() => rec.state !== "inactive" && rec.stop());
        };
        record();
      } else if (engine === "browser") {
        const Rec = browserRecognition()!;
        const rec = new Rec();
        rec.continuous = true;
        rec.interimResults = true;
        rec.lang = navigator.language || "en-US";
        let phraseStart: number | null = null;
        rec.onresult = (e) => {
          for (let i = e.resultIndex; i < e.results.length; i++) {
            const now = (performance.now() - t0) / 1000;
            phraseStart ??= now;
            if (!e.results[i].isFinal) continue;
            const text = e.results[i][0].transcript.trim();
            if (text)
              setSegments((s) => [
                ...s,
                { speaker: "Speaker", start: phraseStart!, end: now, text },
              ]);
            phraseStart = null;
          }
        };
        rec.onerror = (e) => {
          if (e.error === "not-allowed") setError("Microphone access was blocked.");
        };
        rec.onend = () => running && rec.start(); // browsers stop after silence; keep going
        rec.start();
        cleanups.push(() => rec.stop());
      } else {
        throw new Error(
          "This browser can't transcribe live audio. Try Chrome or Edge, or upload a recording.",
        );
      }
    } catch (e) {
      cleanups.forEach((c) => c());
      setError(e instanceof Error ? e.message : String(e));
      return;
    }

    setState("live");
    stopRef.current = async () => {
      if (!running) return;
      running = false;
      cleanups.forEach((c) => c());
      await new Promise((r) => setTimeout(r, 300)); // let the last chunk's onstop fire
      await Promise.allSettled(pending);
    };
  }

  async function stop() {
    setState("saving");
    await stopRef.current();
    const segs = segsRef.current;
    if (!segs.length) {
      setError("Nothing was transcribed, so there is nothing to save.");
      setState("idle");
      return;
    }
    try {
      onSaved(await api.liveSave(title, segs));
    } catch (e) {
      setError((e as Error).message);
      setState("idle");
    }
  }

  return (
    <div className="page live">
      <header className="page-header">
        <div>
          <h1>Live meeting</h1>
          <p className="muted">
            Minutes listens while you meet, transcribes in near real time and keeps the notes up to
            date. Stop to save it as a meeting you can search and ask about.
          </p>
        </div>
      </header>

      <div className="live-controls">
        <input
          aria-label="Meeting title"
          value={title}
          disabled={state !== "idle"}
          onChange={(e) => setTitle(e.target.value)}
        />
        {state === "idle" ? (
          <button
            className="primary"
            disabled={!engine}
            onClick={() => void start()}
            data-testid="live-start"
          >
            ● Start listening
          </button>
        ) : (
          <button
            className="danger"
            disabled={state === "saving"}
            onClick={() => void stop()}
            data-testid="live-stop"
          >
            {state === "saving" ? "Saving…" : "■ Stop and save"}
          </button>
        )}
        {state === "live" && (
          <span className="live-badge">
            <span className="rec-dot" /> {formatTime(elapsed)}
          </span>
        )}
        <span className="muted small">
          {engine === "server"
            ? "Transcription: Whisper on Groq"
            : engine === "browser"
              ? "Transcription: your browser's speech recognition"
              : "Live transcription isn't available in this browser"}
        </span>
      </div>
      {error && <p className="banner error">{error}</p>}

      <div className="live-body">
        <ol
          className="transcript live-transcript"
          aria-label="Live transcript"
          data-testid="live-transcript"
        >
          {segments.map((s, i) => (
            <li key={i}>
              <div className="segment-meta">
                <strong>{s.speaker}</strong>
                <span className="time">{formatTime(s.start)}</span>
              </div>
              <p>{s.text}</p>
            </li>
          ))}
          {state === "live" && (
            <li className="muted small" ref={transcriptEnd}>
              Listening…
            </li>
          )}
          {state === "idle" && !segments.length && (
            <li className="muted">The transcript appears here as people speak.</li>
          )}
        </ol>
        <aside className="notes live-notes" data-testid="live-notes">
          <div className="live-notes-head">
            <h2>Notes so far</h2>
            <button
              className="secondary small"
              disabled={!segments.length || notesBusy}
              onClick={() => void refreshNotes()}
            >
              {notesBusy ? "Updating…" : "Update now"}
            </button>
          </div>
          {!notes && (
            <p className="muted small">Updated every 30 seconds while the meeting runs.</p>
          )}
          {notes && (
            <>
              <p>{notes.summary}</p>
              {notes.decisions.length > 0 && (
                <>
                  <h3>Decisions</h3>
                  <ul>
                    {notes.decisions.map((d, i) => (
                      <li key={i}>{d.decision}</li>
                    ))}
                  </ul>
                </>
              )}
              {notes.action_items.length > 0 && (
                <>
                  <h3>Action items</h3>
                  <ul>
                    {notes.action_items.map((a, i) => (
                      <li key={i}>
                        <strong>{a.owner}</strong> {a.task}
                        {a.due && <span className="due"> · due {a.due}</span>}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {notes.open_questions.length > 0 && (
                <>
                  <h3>Open questions</h3>
                  <ul>
                    {notes.open_questions.map((q, i) => (
                      <li key={i}>{q}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
