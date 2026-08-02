import { useCallback, useEffect, useRef } from "react";

/**
 * Column sizing that behaves.
 *
 * Two things go wrong with a full-width table by default. The browser spreads
 * slack across every column, so a table of short values reads as gaps; and if
 * you hand the slack to the first column, a checkbox column swallows it and
 * everything else crams against the right edge -- which is exactly what
 * happened here. So the column that grows is named, not positional: mark it
 * `grow` and the rest size to their content.
 *
 * On top of that, each header gets a drag handle. Widths are inline styles on
 * the header cells, so they survive re-renders and reset on reload.
 */
export function useResizableColumns<T extends HTMLTableElement>() {
  const ref = useRef<T | null>(null);
  const observer = useRef<MutationObserver | null>(null);

  const attach = useCallback((table: HTMLTableElement) => {
    const headers = Array.from(table.querySelectorAll<HTMLTableCellElement>("thead th"));
    headers.forEach((th, index) => {
      if (th.querySelector(".colgrip") || index === headers.length - 1) return;
      const grip = document.createElement("span");
      grip.className = "colgrip";
      grip.setAttribute("aria-hidden", "true");
      th.appendChild(grip);

      grip.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
        const startX = event.clientX;
        const startWidth = th.getBoundingClientRect().width;
        grip.setPointerCapture(event.pointerId);
        table.classList.add("resizing");

        const move = (e: PointerEvent) => {
          const next = Math.max(48, startWidth + (e.clientX - startX));
          th.style.width = `${next}px`;
          th.style.minWidth = `${next}px`;
        };
        const up = () => {
          table.classList.remove("resizing");
          grip.removeEventListener("pointermove", move);
          grip.removeEventListener("pointerup", up);
        };
        grip.addEventListener("pointermove", move);
        grip.addEventListener("pointerup", up);
      });
    });
  }, []);

  // A plain ref is null while the table is still waiting for its data, and an
  // effect that runs then never retries. A callback ref fires exactly when the
  // node appears; the observer then covers headers re-rendered later.
  const setRef = useCallback((table: T | null) => {
    observer.current?.disconnect();
    ref.current = table;
    if (!table) return;
    attach(table);
    observer.current = new MutationObserver(() => attach(table));
    observer.current.observe(table, { childList: true, subtree: true });
  }, [attach]);

  useEffect(() => () => observer.current?.disconnect(), []);

  return setRef;
}
