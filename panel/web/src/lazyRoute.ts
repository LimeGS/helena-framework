import { lazy, type ComponentType } from "react";

const RELOADED = "helena.chunk-reload";

/**
 * `lazy`, for a page that is still open when the panel is redeployed.
 *
 * Route chunks are content-hashed, so a deploy renames every one of them. A tab
 * that loaded the old index.html goes on asking for the old names, and the first
 * navigation after a deploy dies on a 404 with nothing on screen -- the failure
 * lands inside React's lazy boundary and the route simply never renders.
 *
 * A missing chunk means exactly one thing: this tab is running against a build
 * that no longer exists. So it reloads, which fetches the current index.html and
 * the names that go with it.
 *
 * The sessionStorage flag makes that at most once. If the chunk is still missing
 * after a reload the cause is not staleness -- a bad deploy, a broken path -- and
 * looping on it would replace a visible error with a page that flickers forever.
 */
// ComponentType<any> rather than <unknown>: <unknown> refuses every component
// that takes a prop, and P1's view is handed the active tab. T is still
// inferred as the real component, so call sites keep their prop checking.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function lazyRoute<T extends ComponentType<any>>(load: () => Promise<{ default: T }>) {
  return lazy(async () => {
    try {
      const module = await load();
      sessionStorage.removeItem(RELOADED);
      return module;
    } catch (error) {
      if (sessionStorage.getItem(RELOADED)) throw error;
      sessionStorage.setItem(RELOADED, "1");
      window.location.reload();
      // The reload is not instant; resolving nothing keeps the boundary in its
      // pending state rather than flashing an error on the way out.
      return await new Promise<{ default: T }>(() => {});
    }
  });
}

const STALE = "helena.stale-reload";

/**
 * Reload a tab whose build the server has replaced.
 *
 * lazyRoute only recovers when a chunk is *missing*, and superseded chunks are
 * kept deliberately so an open tab does not 404 mid-navigation. The two
 * together mean a tab open across a deploy keeps running the old build for as
 * long as it stays open: no error, no 404, nothing to notice. A route stuck on a
 * loading message that had already been fixed on the server is what this costs.
 *
 * The comparison is between the entry chunk in the served index.html and the
 * script this document was actually loaded from -- no build-time plumbing, and
 * it is the same identity the browser used.
 *
 * Once per stale build, so a server that cannot read its own index.html, or a
 * proxy serving a different one, cannot put the tab in a reload loop.
 */
export function watchForNewBuild(intervalMs = 60_000) {
  const own = Array.from(document.querySelectorAll<HTMLScriptElement>('script[type="module"]'))
    .map((s) => s.src.split("/").pop())
    .find((name) => name?.startsWith("index-"));
  if (!own) return;

  const check = async () => {
    try {
      const r = await fetch("/api/build", { cache: "no-store" });
      if (!r.ok) return;
      const { entry } = (await r.json()) as { entry: string | null };
      if (!entry || entry === own) return;
      if (sessionStorage.getItem(STALE) === entry) return;
      sessionStorage.setItem(STALE, entry);
      window.location.reload();
    } catch {
      // Offline or the panel is restarting. Staleness is not urgent enough to
      // report, and the next tick will find out.
    }
  };
  check();
  return window.setInterval(check, intervalMs);
}
