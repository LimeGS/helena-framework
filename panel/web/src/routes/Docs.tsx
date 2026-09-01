import { lazy, Suspense, useState } from "react";
import { Empty } from "../components/Bits";

// Three errands, one menu. Somebody running the pipeline for the first time,
// somebody who already runs it and needs to know what a control does, and
// somebody putting their own tool into it are asking different questions -- and
// a page that serves all three serves none.
//
// Tutorial is the path: press these, in this order, get a result. User guide is
// the reference: every control, what it is for, what to leave it on. They were
// one page, which in practice meant the path with the reference missing.
const Handbook = lazy(() => import("./Handbook"));
const Tutorial = lazy(() => import("./Tutorial"));
const UserGuide = lazy(() => import("./UserGuide"));
const DeveloperReference = lazy(() => import("./DeveloperReference"));
const ApiReference = lazy(() => import("./ApiReference"));

const TABS = [
  ["handbook", "Handbook"],
  ["tutorial", "Tutorial"],
  ["guide", "User guide"],
  ["developer", "Developer reference"],
  ["api", "API reference"],
] as const;

export default function Documentation() {
  // Tutorial first and by default: the person who has never run this is the one
  // who cannot yet tell which tab they need.
  const [tab, setTab] = useState<(typeof TABS)[number][0]>(
    window.location.hash.startsWith("#/docs/") ? "handbook" : "handbook");
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
        {tab === "tutorial" && <Tutorial />}
        {tab === "guide" && <UserGuide />}
        {tab === "developer" && <DeveloperReference />}
        {tab === "api" && <ApiReference />}
      </Suspense>
    </>
  );
}
