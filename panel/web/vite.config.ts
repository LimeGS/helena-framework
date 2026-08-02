import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    target: "es2022",
    // Route chunks are emitted by React.lazy; this keeps the vendor half from
    // being re-downloaded whenever a route changes.
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
  server: { proxy: { "/api": "http://127.0.0.1:8800" } },
  // jsdom rather than a browser: what is worth testing here is the logic that
  // decides what to send -- which fields exist, which pair must be exactly one,
  // what the deployment fills in -- and none of that needs a renderer.
  test: { environment: "jsdom", globals: true, include: ["src/**/*.test.tsx"] },
});
