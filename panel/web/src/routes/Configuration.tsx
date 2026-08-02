import { lazy, Suspense, useState } from "react";
import { Empty } from "../components/Bits";

// Settings and the machines they point at. Documentation is its own menu:
// what you change and what you read about are different errands.
const Config = lazy(() => import("./Config"));
const Hosts = lazy(() => import("./Hosts"));
const Users = lazy(() => import("./Users"));
const Lineage = lazy(() => import("./Lineage"));
const Audit = lazy(() => import("./Audit"));
const Modules = lazy(() => import("./Modules"));
const Models = lazy(() => import("./Models"));

const TABS = [
  ["settings", "Settings"],
  ["modules", "Modules"],
  ["models", "Models"],
  ["hosts", "Hosts"],
  ["users", "Users"],
  ["lineage", "Lineage"],
  ["audit", "Audit log"],
] as const;

export default function Configuration() {
  const [tab, setTab] = useState<(typeof TABS)[number][0]>("settings");
  return (
    <>
      <nav className="subtabs">
        {TABS.map(([k, label]) => (
          <button key={k} onClick={() => setTab(k)} aria-current={tab === k ? "page" : undefined}>
            {label}
          </button>
        ))}
      </nav>
      <Suspense fallback={<Empty>loading…</Empty>}>
        {tab === "settings" && <Config />}
        {tab === "modules" && <Modules />}
        {tab === "models" && <Models />}
        {tab === "hosts" && <Hosts />}
        {tab === "users" && <Users />}
        {tab === "lineage" && <Lineage />}
        {tab === "audit" && <Audit />}
      </Suspense>
    </>
  );
}
