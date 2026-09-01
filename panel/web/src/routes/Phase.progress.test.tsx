/**
 * What the queue says about a job while it is running.
 *
 * The row showed Job, Scroll, State, Try and Result, and for a running job that
 * is `running` and four blanks -- for twenty-six minutes, on a P5 render. The
 * only way to tell a job that was working from one that had wedged was to open
 * a shell on the host, and even there the output was buffered until the process
 * exited.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import Phase from "./Phase";

const RUNNING = {
  job_id: "p5-b7cd63a5e68f4c", sample_id: "PHerc0332", phase: "P5",
  state: "running", attempts: 1, max_attempts: 1, result: null,
  progress: {
    line: "Infer:  63%|######    | 6857/10885 [07:41<04:31, 14.84block/s]",
    source: "stderr",
    at: new Date().toISOString(),
  },
};
const PENDING = {
  job_id: "p5-waiting", sample_id: "PHerc0332", phase: "P5",
  state: "pending", attempts: 0, max_attempts: 3, result: null, progress: null,
};

const PHASE = {
  contract: { id: "P5", name: "Ink detection", one_line: "Find the ink",
              maturity: "WORKING", distributed: true },
  state: {}, artefacts: [], jobs: [RUNNING, PENDING], profiles: [], components: [],
  queueable: true, queueable_reason: null,
};

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async (input: any) => {
    const path = String(input);
    const body = path.startsWith("/api/phase/P5") ? PHASE : {};
    return { ok: true, status: 200, json: async () => body };
  }));
});

afterEach(() => vi.unstubAllGlobals());

vi.mock("react-router", () => ({ useParams: () => ({ phaseId: "P5" }) }));
vi.mock("../mission", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../mission")>()),
  useMission: () => ({ missionId: "test" }),
  useSubject: () => ({ subject: "PHerc0332" }),
}));

function draw() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Phase />
    </QueryClientProvider>,
  );
}

async function openTheQueue() {
  draw();
  // The queue lives under Run, which is not the tab a phase opens on.
  const run = await screen.findByRole("button", { name: /^Run\b/ });
  fireEvent.click(run);
}

it("shows what a running job is doing", async () => {
  await openTheQueue();

  expect(await screen.findByText(/6857\/10885/)).toBeDefined();
});

it("says nothing about a job that has not started", async () => {
  await openTheQueue();
  await screen.findByText(/6857\/10885/);

  // A pending row with a progress line would be reporting somebody else's.
  const row = screen.getByText("p5-waiting").closest("tr")!;
  expect(row.textContent).not.toMatch(/block\/s/);
});
