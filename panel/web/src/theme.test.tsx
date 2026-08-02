import { expect, test } from "vitest";
import { resolve } from "./theme";

test("auto follows the system, an explicit choice overrides it", () => {
  expect(resolve("auto", true)).toBe("dark");
  expect(resolve("auto", false)).toBe("light");
  // The point of the control: a dark system with light picked stays light.
  expect(resolve("light", true)).toBe("light");
  expect(resolve("dark", false)).toBe("dark");
});
