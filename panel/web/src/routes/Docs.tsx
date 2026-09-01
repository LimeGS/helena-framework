import { lazy, Suspense, useState } from "react";
import { Empty } from "../components/Bits";

// The handbook is the prose: the walkthrough, the phases, and every panel page
// with its controls, built from docs/handbook and served whole. It replaced a
// separate Tutorial and User guide, which had split the path from the reference
// and so answered a first run with the reference missing and a lookup with the
// path in the way. The two references that remain are generated from the code
// they describe rather than written, which is why they stay their own tabs.
const Handbook = lazy(() => import("./Handbook"));
const DeveloperReference = lazy(() => import("./DeveloperReference"));
const ApiReference = lazy(() => import("./ApiReference"));

const TABS = [
  ["handbook", "Handbook"],
  ["developer", "Developer reference"],
  ["api", "API reference"],
] as const;

export default function Documentation() {
  // The handbook by default: the person who has never run this is the one who
  // cannot yet tell which tab they need, and it is the tab that answers that.
  const [tab, setTab] = useState<(typeof TABS)[number][0]>("handbook");
  return (
    <>
      <nav className="subtabs">
        {TABS.map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)}
                  aria-current={tab === k ? "page" : undefined}>
            {label}
          </button>
        ))}
      </nav>
      <Suspense fallback={<Empty>loading…</Empty>}>
        {tab === "handbook" && <Handbook />}
        {tab === "developer" && <DeveloperReference />}
        {tab === "api" && <ApiReference />}
      </Suspense>
    </>
  );
}
