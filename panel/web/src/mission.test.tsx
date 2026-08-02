import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import {
  MissionProvider,
  SubjectProvider,
  useMission,
  useSubject,
} from "./mission";

const MISSIONS = [
  { mission_id: "mission-a", name: "A", description: "", state: "active",
    scrolls: ["PHerc826"], scrolls_frozen_at_utc: null, created_at_utc: null,
    amendments: [], non_claims: [], path: "/runs/a", run_count: 0 },
  { mission_id: "mission-empty", name: "Empty", description: "", state: "active",
    scrolls: [], scrolls_frozen_at_utc: null, created_at_utc: null,
    amendments: [], non_claims: [], path: "/runs/empty", run_count: 0 },
];

function Probe() {
  const { missionId, setMissionId, missions } = useMission();
  const { subject, setSubject } = useSubject();
  return (
    <>
      <span data-testid="mission">{missionId ?? "none"}</span>
      <span data-testid="subject">{subject ?? "none"}</span>
      <span data-testid="mission-count">{missions.length}</span>
      <button onClick={() => setMissionId("mission-a")}>mission A</button>
      <button onClick={() => setSubject("PHerc826")}>scroll 826</button>
      <button onClick={() => setMissionId("mission-empty")}>empty mission</button>
      <button onClick={() => setMissionId(null)}>clear mission</button>
    </>
  );
}

beforeEach(() => {
  localStorage.clear();
  window.history.replaceState({}, "", "/?tab=new");
  vi.stubGlobal("fetch", vi.fn(async () => ({
    ok: true,
    json: async () => ({ missions: MISSIONS, runs_root: "/runs" }),
  })));
});

afterEach(() => vi.unstubAllGlobals());

it("drops the previous scroll atomically when the mission changes", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MissionProvider>
        <SubjectProvider><Probe /></SubjectProvider>
      </MissionProvider>
    </QueryClientProvider>,
  );

  await waitFor(() => expect(screen.getByTestId("mission-count").textContent).toBe("2"));
  fireEvent.click(screen.getByRole("button", { name: "mission A" }));
  await waitFor(() => expect(screen.getByTestId("mission").textContent).toBe("mission-a"));
  expect(new URL(window.location.href).searchParams.get("mission")).toBe("mission-a");
  fireEvent.click(screen.getByRole("button", { name: "scroll 826" }));
  await waitFor(() => expect(screen.getByTestId("subject").textContent).toBe("PHerc826"));

  fireEvent.click(screen.getByRole("button", { name: "empty mission" }));
  await waitFor(() => {
    expect(screen.getByTestId("mission").textContent).toBe("mission-empty");
    expect(screen.getByTestId("subject").textContent).toBe("none");
  });
  expect(new URL(window.location.href).searchParams.get("mission")).toBe("mission-empty");

  fireEvent.click(screen.getByRole("button", { name: "clear mission" }));
  await waitFor(() => expect(screen.getByTestId("mission").textContent).toBe("none"));
  const url = new URL(window.location.href);
  expect(url.searchParams.has("mission")).toBe(false);
  expect(url.searchParams.get("tab")).toBe("new");
});
