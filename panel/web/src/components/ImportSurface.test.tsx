/**
 * What the uploader sends, and what it does when it cannot finish.
 *
 * The same reasoning as the launcher's test file: every defect worth catching
 * here is invisible from the page and invisible from the server. A form that
 * silently drops a file still shows the file in its list; an upload abandoned
 * halfway still shows an error message. Only reading the fetches catches those.
 *
 * The one that matters most is the last: bytes that landed before a failure sit
 * on the volume that is this deployment's copy of record, and nothing else is
 * ever going to finish that upload.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { ImportSurface } from "./ImportSurface";

const OPENED = {
  upload_id: "a".repeat(32),
  required: ["x.tif", "y.tif", "z.tif", "meta.json"],
  optional: ["generations.tif"],
  max_bytes_per_file: 536870912,
};

let calls: { url: string; method: string; body?: unknown }[] = [];

function answer(url: string, method: string) {
  if (url === "/api/scrolls") return { scrolls: [{ sample_id: "PHerc0332" }] };
  if (url === "/api/segmentation/uploads" && method === "POST") return OPENED;
  if (url.endsWith("/commit")) return { sample_id: "PHerc0332", inserted: 1 };
  return { ok: true };
}

beforeEach(() => {
  calls = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    calls.push({ url, method, body: init?.body });
    return {
      ok: true, status: 200,
      json: async () => answer(url, method),
    } as Response;
  }));
});

afterEach(() => vi.unstubAllGlobals());

function draw(sample: string | null = "PHerc0332") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ImportSurface sample={sample} missionId="test" onDone={() => {}} />
    </QueryClientProvider>,
  );
}

function pick(names: string[]) {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  const files = names.map((name) => new File(["bytes"], name));
  Object.defineProperty(input, "files", { value: files, configurable: true });
  fireEvent.change(input);
  return input;
}

const SURFACE = ["x.tif", "y.tif", "z.tif", "meta.json"];

it("sends every file of the surface and then commits", async () => {
  draw();
  pick(SURFACE);
  fireEvent.click(screen.getByRole("button", { name: /import surface/i }));

  await waitFor(() => expect(calls.some((c) => c.url.endsWith("/commit"))).toBe(true));

  const put = calls.filter((c) => c.method === "PUT").map((c) => c.url.split("/").pop());
  expect(put).toEqual(SURFACE);
});

it("commits the scroll and nothing the bytes already answer", async () => {
  draw();
  pick(SURFACE);
  fireEvent.click(screen.getByRole("button", { name: /import surface/i }));

  const commit = await waitFor(() => {
    const found = calls.find((c) => c.url.endsWith("/commit"));
    expect(found).toBeDefined();
    return found!;
  });
  const body = JSON.parse(String(commit.body));

  expect(body).toEqual({ sample_id: "PHerc0332", owner: "uploaded", mission_id: "test" });
  // Bounds and area are measured from the bytes on arrival. A form that sent
  // them would be sending a restatement, and the restatement would be recorded.
  expect(body).not.toHaveProperty("bbox_xyz");
  expect(body).not.toHaveProperty("area_cm2");
});

it("leaves a file the surface does not contain on the laptop", async () => {
  draw();
  pick([...SURFACE, "notes.txt"]);
  fireEvent.click(screen.getByRole("button", { name: /import surface/i }));

  await waitFor(() => expect(calls.some((c) => c.url.endsWith("/commit"))).toBe(true));
  expect(calls.some((c) => c.url.includes("notes.txt"))).toBe(false);
});

it("refuses half a surface before sending anything", async () => {
  draw();
  pick(["x.tif", "meta.json"]);
  fireEvent.click(screen.getByRole("button", { name: /import surface/i }));

  await screen.findByText(/y\.tif, z\.tif are missing/);
  expect(calls.some((c) => c.method === "PUT")).toBe(false);
});

it("takes its bytes with it when the commit fails", async () => {
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const method = init?.method ?? "GET";
    calls.push({ url, method, body: init?.body });
    const refused = url.endsWith("/commit");
    return {
      ok: !refused, status: refused ? 409 : 200,
      json: async () => refused
        ? { detail: "PHerc0332 has no source snapshot" }
        : answer(url, method),
    } as Response;
  }));

  draw();
  pick(SURFACE);
  fireEvent.click(screen.getByRole("button", { name: /import surface/i }));

  await screen.findByText(/no source snapshot/);
  expect(calls.some((c) => c.method === "DELETE"
                          && c.url === `/api/segmentation/uploads/${OPENED.upload_id}`)).toBe(true);
});

it("does not delete an upload that committed", async () => {
  draw();
  pick(SURFACE);
  fireEvent.click(screen.getByRole("button", { name: /import surface/i }));

  await waitFor(() => expect(calls.some((c) => c.url.endsWith("/commit"))).toBe(true));
  await waitFor(() => expect(screen.getByText(/IMPORTED/)).toBeTruthy());
  expect(calls.some((c) => c.method === "DELETE")).toBe(false);
});
