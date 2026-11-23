// Pre-filled "new issue" links for GitHub and Linear: no tokens, the user confirms in the tracker.

export interface IssueSource {
  task: string;
  owner: string;
  due: string;
  meeting: string;
}

function body(item: IssueSource): string {
  const lines = [`From the meeting "${item.meeting}".`, "", `Owner: ${item.owner}`];
  if (item.due) lines.push(`Due: ${item.due}`);
  lines.push("", "_Created from Minutes._");
  return lines.join("\n");
}

/** `repo` is "owner/name"; returns null when it doesn't look like one. */
export function githubIssueUrl(repo: string, item: IssueSource): string | null {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo.trim())) return null;
  const q = new URLSearchParams({ title: item.task, body: body(item) });
  return `https://github.com/${repo.trim()}/issues/new?${q}`;
}

export function linearIssueUrl(item: IssueSource): string {
  const q = new URLSearchParams({ title: item.task, description: body(item) });
  return `https://linear.app/new?${q}`;
}

const KEY = "minutes.githubRepo";
export const savedRepo = () => {
  try {
    return localStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
};
export const saveRepo = (repo: string) => {
  try {
    localStorage.setItem(KEY, repo.trim());
  } catch {
    /* storage blocked */
  }
};
