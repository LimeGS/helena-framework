/**
 * Standing explanation belongs behind the "i", not in front of the data.
 *
 * These pages carried four and five sentences of context above the table they
 * described. Every sentence was true and worth having; all of them were read
 * once and then became furniture. The page read as dense when the density was
 * prose.
 *
 * Folding it away is only correct if it is still reachable, so that is what
 * these check: hidden at rest, present on demand, and announced to a screen
 * reader either way. Deleting the text would also have made the page shorter.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Card, Info } from "./Bits";

describe("Info", () => {
  it("hides its text until asked", () => {
    render(<Info label="What this does">A long-standing explanation.</Info>);
    expect(screen.queryByText("A long-standing explanation.")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "What this does" }));
    expect(screen.getByText("A long-standing explanation.")).toBeTruthy();
  });

  it("closes again, so it is not a one-way door", () => {
    render(<Info label="What this does">Explanation.</Info>);
    const button = screen.getByRole("button", { name: "What this does" });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(screen.queryByText("Explanation.")).toBeNull();
  });

  it("says what it is, to a screen reader and to assistive tooling", () => {
    // The visible label is a single letter. Without the aria-label this is a
    // button called "i", which is no label at all.
    render(<Info label="Why only certified surfaces">Because of the seam.</Info>);
    const button = screen.getByRole("button", { name: "Why only certified surfaces" });
    expect(button.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(button);
    expect(button.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByRole("tooltip")).toBeTruthy();
  });

  it("can sit in a card title without hiding the title", () => {
    // How every one of these is actually used: the heading stays readable and
    // the explanation moves one click away.
    render(
      <Card title={<>Geometry verdicts <Info label="How certification works">
        Fail-soft in control flow, fail-closed in verdict.
      </Info></>}>
        <div>the table</div>
      </Card>,
    );
    expect(screen.getByText(/Geometry verdicts/)).toBeTruthy();
    expect(screen.getByText("the table")).toBeTruthy();
    expect(screen.queryByText(/fail-closed in verdict/i)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "How certification works" }));
    expect(screen.getByText(/fail-closed in verdict/i)).toBeTruthy();
  });
});
