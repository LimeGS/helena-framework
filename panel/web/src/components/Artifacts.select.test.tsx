/**
 * Choosing which P0 artifact a mission reads, from the page that registered it.
 *
 * The register lives on P0 and already says which version is in use. Changing
 * that lived somewhere else entirely -- Configuration → Lineage -- so the
 * sequence was: freeze on P0, then leave the phase, find a tab under settings,
 * and pick there. Nothing was broken, but the control that acts on a table sat
 * three navigations away from it, and the queue refuses to take work until it
 * has been used.
 *
 * One table, not two: adding a second list of the same artifacts on P0 would
 * have made "which one is the register" a question.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { ProducedArtifacts } from "./Artifacts";

const ROWS = {
  artifacts: [
    { artifact_id: "p0:PHerc0139:fresh-a7a02eebd470", phase: "P0", sample_id: "PHerc0139",
      kind: "selection", file_count: 1, total_bytes: 900, exists: true,
      selected: false, registered_at_utc: "2026-08-27T00:00:00Z", path: "/p0.json" },
    { artifact_id: "p0:PHerc0139:stale-0000deadbeef", phase: "P0", sample_id: "PHerc0139",
      kind: "selection", file_count: 1, total_bytes: 880, exists: true,
      selected: true, registered_at_utc: "2026-08-26T00:00:00Z", path: "/old.json" },
  ],
};

const posted: any[] = [];

beforeEach(() => {
  posted.length = 0;
  vi.stubGlobal("fetch", vi.fn(async (input: any, init?: any) => {
    const path = String(input);
    if (init?.method === "POST") {
      posted.push({ path, body: JSON.parse(init.body) });
      return { ok: true, status: 201, json: async () => ({ version_id: "sel-1" }) };
    }
    const body = path.includes("/selection")
      ? { current: { choices: { "P0/PHerc0826": "p0:PHerc0826:keepme" } }, versions: [] }
      : path.includes("/artifacts") ? ROWS : {};
    return { ok: true, status: 200, json: async () => body };
  }));
});

afterEach(() => vi.unstubAllGlobals());

vi.mock("../mission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../mission")>()),
  useMission: () => ({
    missionId: "public-ink-control-0139",
    current: { scrolls: ["PHerc0139"], implicit: false },
  }),
}));

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ProducedArtifacts phase="P0" sample={null} />
    </QueryClientProvider>,
  );
}

it("offers the choice on the row that is not in use", async () => {
  draw();

  const row = (await screen.findByText(/p0:PHerc0139:fr/)).closest("tr")!;
  expect(row.querySelector("button")).toBeTruthy();
});

it("does not offer to re-choose the one already in use", async () => {
  draw();
  await screen.findByText(/p0:PHerc0139:fr/);

  const row = screen.getByText(/p0:PHerc0139:st/).closest("tr")!;
  expect(row.querySelector("button")).toBeNull();
});

it("sends the whole selection map, not a patch", async () => {
  /** One entry moving on its own makes "what was selected then" unanswerable. */
  draw();
  const row = (await screen.findByText(/p0:PHerc0139:fr/)).closest("tr")!;

  fireEvent.click(row.querySelector("button")!);

  await waitFor(() => expect(posted.length).toBe(1));
  expect(posted[0].path).toContain("/selection");
  expect(posted[0].body.choices).toEqual({
    "P0/PHerc0826": "p0:PHerc0826:keepme",
    "P0/PHerc0139": "p0:PHerc0139:fresh-a7a02eebd470",
  });
});

it("keeps a reason on the record", async () => {
  draw();
  const row = (await screen.findByText(/p0:PHerc0139:fr/)).closest("tr")!;

  const reason = screen.getByPlaceholderText(/why this one/i);
  fireEvent.change(reason, { target: { value: "the public control reads this P0" } });
  fireEvent.click(row.querySelector("button")!);

  await waitFor(() => expect(posted.length).toBe(1));
  expect(posted[0].body.reason).toBe("the public control reads this P0");
});
