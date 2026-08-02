import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * The last stop when a route will not load.
 *
 * `lazyRoute` reloads once on a missing chunk, which fixes the ordinary case: a
 * tab open across a deploy. If the chunk is still missing after that, the cause
 * is not staleness and reloading again would only flicker -- so it surfaces
 * here instead. Without this the failure lands inside React's lazy boundary and
 * the page simply goes blank, which is the same thing a hung request looks like.
 */
export class RouteError extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("route failed to load", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    const stale = /dynamically imported module|Importing a module script failed/i
      .test(this.state.error.message);
    return (
      <div className="card">
        <div className="cardhead"><h2>This page did not load</h2></div>
        <div className="body-pad">
          <p>
            {stale
              ? "The panel was redeployed and this tab reloaded once to catch up, but the " +
                "page it needs is still missing from the server. That is a deploy that did " +
                "not finish, not a stale tab."
              : "The page failed while loading."}
          </p>
          <p><code>{this.state.error.message}</code></p>
          <div className="controls">
            <button onClick={() => { sessionStorage.removeItem("helena.chunk-reload");
                                     window.location.reload(); }}>
              try again
            </button>
          </div>
        </div>
      </div>
    );
  }
}
