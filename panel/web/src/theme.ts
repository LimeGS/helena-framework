/**
 * Three choices, two outcomes. "auto" is not a third palette: it resolves to
 * light or dark and `data-theme` on <html> always holds the resolved answer,
 * so CSS never has to work out what "auto" means.
 *
 * The same resolution is inlined in index.html to run before the bundle does --
 * that is what stops a dark-theme operator seeing a white page flash on every
 * load. If the key or the attribute changes here, change it there too.
 */
export type Theme = "auto" | "light" | "dark";

const KEY = "helena-theme";
const DARK = "(prefers-color-scheme: dark)";

export function resolve(theme: Theme, systemDark: boolean): "light" | "dark" {
  return theme === "auto" ? (systemDark ? "dark" : "light") : theme;
}

export function readTheme(): Theme {
  const stored = localStorage.getItem(KEY);
  return stored === "light" || stored === "dark" ? stored : "auto";
}

export function applyTheme(theme: Theme): void {
  localStorage.setItem(KEY, theme);
  const resolved = resolve(theme, matchMedia(DARK).matches);
  document.documentElement.dataset.theme = resolved;
  // On a phone the browser chrome is part of the page: a panel forced to light
  // under a dark address bar reads as two different applications.
  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", resolved === "dark" ? "#23242B" : "#F7F4ED");
}

/** While the choice is "auto", follow the system if it changes under us. */
export function watchSystemTheme(): void {
  matchMedia(DARK).addEventListener("change", () => {
    if (readTheme() === "auto") applyTheme("auto");
  });
}
