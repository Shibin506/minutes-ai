export function formatTime(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s % 60)}` : `${pad(m)}:${pad(s % 60)}`;
}

export function formatDuration(seconds: number): string {
  const minutes = Math.round(seconds / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

export type AnswerPart =
  { kind: "text"; text: string; bold?: boolean } | { kind: "cite"; index: number };

/** Models sometimes answer in Markdown: turn "* item" / "- item" bullets into "• item". */
function tidyMarkdown(answer: string): string {
  return answer.replace(/^([ \t]*)[*-][ \t]+/gm, "$1• ");
}

/** Text with **bold** spans split out. An unclosed "**" (mid-stream) stays as text. */
function pushText(parts: AnswerPart[], text: string) {
  let last = 0;
  for (const match of text.matchAll(/\*\*([^*\n]+?)\*\*/g)) {
    if (match.index > last) parts.push({ kind: "text", text: text.slice(last, match.index) });
    parts.push({ kind: "text", text: match[1], bold: true });
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
}

/** Split "Launch moved [S2][S3]." into text and citation parts (1-based source indexes). */
export function splitCitations(answer: string): AnswerPart[] {
  const parts: AnswerPart[] = [];
  const text = tidyMarkdown(answer);
  const pattern = /\[S(\d+)(?:\s*,\s*S?(\d+))*\]/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) pushText(parts, text.slice(last, match.index));
    for (const n of match[0].matchAll(/\d+/g)) parts.push({ kind: "cite", index: Number(n[0]) });
    last = match.index + match[0].length;
  }
  if (last < text.length) pushText(parts, text.slice(last));
  return parts;
}
