import { describe, expect, it } from "vitest";
import { parseSSE } from "./api";
import { formatDuration, formatTime, splitCitations } from "./format";

describe("formatTime", () => {
  it("formats minutes and hours", () => {
    expect(formatTime(65.9)).toBe("01:05");
    expect(formatTime(3725)).toBe("1:02:05");
  });
});

describe("formatDuration", () => {
  it("rounds to minutes", () => {
    expect(formatDuration(190)).toBe("3 min");
    expect(formatDuration(3900)).toBe("1 h 5 min");
  });
});

describe("splitCitations", () => {
  it("separates text and citations", () => {
    expect(splitCitations("Launch moved [S2][S3].")).toEqual([
      { kind: "text", text: "Launch moved " },
      { kind: "cite", index: 2 },
      { kind: "cite", index: 3 },
      { kind: "text", text: "." },
    ]);
  });

  it("handles comma lists and plain text", () => {
    expect(splitCitations("x [S1, S4]")).toEqual([
      { kind: "text", text: "x " },
      { kind: "cite", index: 1 },
      { kind: "cite", index: 4 },
    ]);
    expect(splitCitations("no citations")).toEqual([{ kind: "text", text: "no citations" }]);
  });
});

describe("parseSSE", () => {
  it("returns complete events and keeps the partial tail", () => {
    const buffer =
      'event: sources\ndata: []\n\nevent: delta\ndata: "Hi"\n\nevent: delta\ndata: "the';
    const { events, rest } = parseSSE(buffer);
    expect(events).toEqual([
      { event: "sources", data: [] },
      { event: "delta", data: "Hi" },
    ]);
    expect(rest).toBe('event: delta\ndata: "the');
  });
});
