import { describe, expect, it } from "vitest";
import { githubIssueUrl, linearIssueUrl } from "./issues";

const item = {
  task: "Ship the dedupe fix",
  owner: "Aisha",
  due: "October 20th",
  meeting: "Go/no-go",
};

describe("issue links", () => {
  it("pre-fills a GitHub issue for a valid repo only", () => {
    const url = new URL(githubIssueUrl("acme/app", item)!);
    expect(url.pathname).toBe("/acme/app/issues/new");
    expect(url.searchParams.get("title")).toBe("Ship the dedupe fix");
    expect(url.searchParams.get("body")).toContain("Due: October 20th");
    expect(githubIssueUrl("not a repo", item)).toBeNull();
  });

  it("pre-fills a Linear issue", () => {
    const url = new URL(linearIssueUrl(item));
    expect(url.searchParams.get("title")).toBe("Ship the dedupe fix");
    expect(url.searchParams.get("description")).toContain("Owner: Aisha");
  });
});
