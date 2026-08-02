import { expect, test } from "vitest";
import { failure } from "./api";

// The bug this replaces: `(await r.json()).detail ?? \`HTTP ${r.status}\`` reads
// as having a fallback and does not have one. When the body is not JSON the
// parse throws before `??` is reached, so what the operator saw was
// "SyntaxError: Unexpected token 'I', "Internal S"... is not valid JSON" --
// a message about the shape of the error, from a card whose real problem was a
// 500. FastAPI's unhandled-exception body is exactly that plain text.
test("a non-JSON error body still reports the status", async () => {
  const error = await failure(new Response("Internal Server Error", { status: 500 }));
  expect(error.message).toContain("500");
  expect(error.message).not.toContain("JSON");
});

test("a JSON detail is preferred when there is one", async () => {
  const error = await failure(new Response(
    JSON.stringify({ detail: "CX_DB is not set; there is no control plane" }),
    { status: 409 },
  ));
  expect(error.message).toBe("CX_DB is not set; there is no control plane");
});

test("JSON without a detail falls back to the status", async () => {
  const error = await failure(new Response(JSON.stringify({ error: "nope" }), { status: 502 }));
  expect(error.message).toContain("502");
});
