import { watchForNewBuild } from "./lazyRoute";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Gate } from "./Login";
import { BrowserRouter } from "react-router";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { MissionProvider, SubjectProvider } from "./mission";
import { watchSystemTheme } from "./theme";
import "./styles.css";

const client = new QueryClient({
  defaultOptions: {
    queries: {
      // A dashboard on a private network refetching on every window focus is
      // noise; the volatile query already polls on its own schedule.
      refetchOnWindowFocus: false,
      retry: 1,
      gcTime: 10 * 60 * 1000,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <Gate>
        <MissionProvider>
          <SubjectProvider>
            <App />
          </SubjectProvider>
        </MissionProvider>
        </Gate>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);

// A tab left open across a deploy runs the old build until it is told.
watchForNewBuild();

// index.html has already resolved the theme; this is only for the machine that
// switches itself at sunset while the panel is open on "auto".
watchSystemTheme();
