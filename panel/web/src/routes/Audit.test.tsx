/**
 * The audit page, which exists to answer one question under pressure: who did
 * that, and when.
 *
 * What is worth asserting is that the filters reach the server rather than
 * quietly filtering nothing, and that a refusal is visually distinct from a
 * success. A page where 401 and 201 look alike is a page where the interesting
 * line is the one you scroll past.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import Audit from "./Audit";

const ENTRIES = [
  { id: "aaaa000000000001", at: "2026-07-28T21:04:11Z", user: "limegs",
    action: "POST /api/jobs", method: "POST", path: "/api/jobs",
    query: null, status: 201, ms: 34, client: "10.0.0.9" },
  { id: "aaaa000000000002", at: "2026-07-28T20:59:02Z", user: "anonymous",
    action: "POST /api/jobs", method: "POST", path: "/api/jobs",
    query: null, status: 401, ms: 2, client: "203.0.113.7" },
];

const asked: string[] = [];

beforeEach(() => {
  asked.length = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: any) => {
    const path = String(input);
    asked.push(path);
    return {
      ok: true, status: 200,
      json: async () => ({
        entries: ENTRIES, count: ENTRIES.length, limit: 200,
        months: ["2026-07"], users: ["anonymous", "limegs"],
        root: "/state/audit",
        // Verbatim from the endpoint: the page repeats what the server says it
        // keeps rather than making its own claim about it.
        captures: "timestamp, id, user, action, status, duration and client " +
                  "address. Request bodies are never recorded.",
      }),
    };
  }));
});

afterEach(() => vi.unstubAllGlobals());

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Audit />
    </QueryClientProvider>,
  );
}

describe("the audit log", () => {
  it("shows the four columns the log is kept for", async () => {
    draw();
    await screen.findByText("aaaa000000000001");            // id
    expect(screen.getByText("2026-07-28 21:04:11")).toBeTruthy();  // horario
    expect(screen.getAllByText("limegs").length).toBeGreaterThan(0);  // usuario
    expect(screen.getAllByText("POST /api/jobs").length).toBe(2);  // accion
  });

  it("tells a refusal apart from a success", async () => {
    draw();
    const refused = await screen.findByText("401");
    const accepted = screen.getByText("201");
    expect(refused.className).toContain("crit");
    expect(accepted.className).toContain("ok");
  });

  it("filters on the server rather than in the browser", async () => {
    // The page holds the newest few hundred entries. Filtering what has already
    // been fetched would search the last page of a year of history and report
    // nothing found.
    draw();
    await screen.findByText("aaaa000000000001");
    fireEvent.change(screen.getByPlaceholderText("/api/jobs"),
                     { target: { value: "/api/users" } });
    await waitFor(() =>
      expect(asked.some((u) => u.includes("contains=%2Fapi%2Fusers"))).toBe(true));
  });

  it("says out loud that bodies are not captured", async () => {
    // Two of the audited routes carry a password and a set of S3 credentials.
    // Somebody reading this page needs to know which of those it holds.
    draw();
    await screen.findByText("aaaa000000000001");
    expect(document.body.textContent).toContain("Request bodies are never recorded");
  });
});
