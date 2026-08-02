import { expect, test } from "vitest";
import contract from "../../../framework/contracts/pipeline_phases.json";
import { PIPELINE } from "./phases";

// The bundled list is a copy, and a copy is only safe if something notices when
// it stops matching. Rename or add a phase in the contract and the rail would
// draw the stale version until the server's answer replaced it; this fails
// instead. Test-only import, so the contract does not reach the app bundle.
test("the bundled phase list matches the contract", () => {
  const fromContract = contract.phases.map((p) => ({
    id: p.id, slug: p.slug, name: p.name,
  }));
  const fromBundle = PIPELINE.map((p) => ({ id: p.id, slug: p.slug, name: p.name }));

  expect(fromBundle).toEqual(fromContract);
});
