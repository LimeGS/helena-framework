# Control panel

A FastAPI JSON API plus a React SPA. It reads what the pipeline already writes:
receipts on disk, lane profiles under git, the fleet tables in PostgreSQL and
`nvidia-smi`. It invents no state of its own.

## Build and run

    cd panel/web && npm ci && npm run build      # needs node; produces web/dist
    cd ../.. && python3 -m venv panel/.venv
    panel/.venv/bin/pip install -r panel/requirements.txt

    CX_REPO=/path/to/repo \
    CX_RUNS=/srv/helena/runs \
    CX_DB='postgresql://helena:PASS@127.0.0.1:55432/helena' \
      panel/.venv/bin/uvicorn panel.app:app --host 0.0.0.0 --port 8800

Without `CX_DB` the Fleet page explains what is missing instead of breaking. If
`web/dist` does not exist, the root returns 503 with the build command.

In development, `npm run dev` starts Vite with `/api` proxied to port 8800.

In a container the image's own entrypoint runs instead, which generates a
self-signed certificate into `/state/tls` on first boot and serves over TLS. The
start-up log prints its SHA-256 fingerprint, so the first visit can be verified
rather than clicked through. Point `CX_TLS_CERT` and `CX_TLS_KEY` at a real
certificate to use that instead.

## Performance decisions

The map endpoint serves **luminance**, not colour: probability goes in RGB and
validity in alpha. The client uploads that once as a texture and applies the ramp
and the threshold in the fragment shader, so moving the slider changes a uniform
and never touches the network. Colouring on the server put a 600 KB PNG on the
wire for every pixel the control moved.

The same endpoint serves **only the visible window**, resampled to the canvas
size. A 4096-square `.npy` in float32 is 67 MiB; the viewer never asks for more
texels than it can display, so zooming in costs what zooming out costs.

The TanStack Query cache is split by how each kind of data behaves: a receipt is
written once and never edited, so its query is `staleTime: Infinity`; only
`/api/state` polls. The usual reflex — refresh everything on one interval — is
how a panel like this ends up hammering a database that had nothing new to say.

`/api/docs` is roughly 570 KB of signatures and docstrings read from the code
with `ast`, importing nothing. It is the only large response, which is why gzip
is enabled.

## Design

Documentation → Developer reference, served by the panel itself.
