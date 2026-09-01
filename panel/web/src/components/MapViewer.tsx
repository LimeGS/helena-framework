import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { boxFromDrag, coverage, pointToMap, reviewBox } from "./roiSelection";
import type { Box } from "./roiSelection";

/**
 * Probability map viewer.
 *
 * Two decisions carry the performance here.
 *
 * The server sends *luminance*, not colour: probability in RGB and validity in
 * alpha. The colour ramp and the threshold are applied in the fragment shader,
 * so dragging the threshold slider changes one uniform and repaints -- no
 * request, no decode, no allocation. Colouring on the server would put a
 * 600 KB PNG on the wire for every pixel the slider moves.
 *
 * The server also sends only the window in view, resampled to the canvas size.
 * A 4096x4096 float32 map is 67 MiB; the viewer never asks for more texels than
 * it can display, so zooming in costs the same as zooming out.
 */

const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x, 1.0 - a_pos.y);
  gl_Position = vec4(a_pos * 2.0 - 1.0, 0.0, 1.0);
}`;

// The ramp goes cool below the threshold and warm above it, so a change of
// threshold reads as the same picture re-lit rather than a different picture.
const FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_map;
uniform float u_threshold;
uniform float u_gamma;
out vec4 outColor;

vec3 ramp(float t) {
  const vec3 c0 = vec3(0.102, 0.118, 0.129);
  const vec3 c1 = vec3(0.275, 0.306, 0.329);
  const vec3 c2 = vec3(0.549, 0.549, 0.518);
  const vec3 c3 = vec3(0.839, 0.659, 0.361);
  const vec3 c4 = vec3(0.925, 0.886, 0.816);
  float s = clamp(t, 0.0, 1.0) * 4.0;
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  return mix(c3, c4, s - 3.0);
}

void main() {
  vec4 texel = texture(u_map, v_uv);
  if (texel.a < 0.5) { outColor = vec4(0.071, 0.078, 0.086, 1.0); return; }
  float p = pow(texel.r, u_gamma);
  float t = p >= u_threshold
    ? 0.5 + 0.5 * (p - u_threshold) / max(1.0 - u_threshold, 1e-4)
    : 0.5 * p / max(u_threshold, 1e-4);
  outColor = vec4(ramp(t), 1.0);
}`;

export type MapMeta = {
  width: number;
  height: number;
  p50?: number;
  p90?: number;
  p99?: number;
  max?: number;
  valid_pixels: number;
};

type View = { x: number; y: number; scale: number };

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "shader failed");
  }
  return shader;
}

export const MapViewer = memo(function MapViewer({
  runId,
  name,
  meta,
  threshold,
  gamma = 1,
  height = 520,
  view: controlledView,
  onViewChange,
  selecting = false,
  onSelect,
}: {
  runId: string;
  name: string;
  meta: MapMeta;
  threshold: number;
  gamma?: number;
  height?: number;
  view?: View;
  onViewChange?: (v: View) => void;
  /** Drag selects a region instead of panning. */
  selecting?: boolean;
  /** The region, in map pixels, as it is dragged and when it settles. */
  onSelect?: (box: Box | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGL2RenderingContext | null>(null);
  const progRef = useRef<WebGLProgram | null>(null);
  const texRef = useRef<WebGLTexture | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [localView, setLocalView] = useState<View>({ x: 0, y: 0, scale: 1 });
  const view = controlledView ?? localView;
  const setView = useCallback(
    (next: View) => {
      onViewChange ? onViewChange(next) : setLocalView(next);
    },
    [onViewChange],
  );

  // ---- one-time GL setup -------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
    if (!gl) {
      setError("This browser does not expose WebGL2.");
      return;
    }
    try {
      const program = gl.createProgram()!;
      gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERT));
      gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program) ?? "link failed");
      }
      gl.useProgram(program);

      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(program, "a_pos");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      glRef.current = gl;
      progRef.current = program;
      texRef.current = texture;
    } catch (e) {
      setError(String(e));
    }
    return () => {
      const gl = glRef.current;
      if (!gl) return;
      if (texRef.current) gl.deleteTexture(texRef.current);
      if (progRef.current) gl.deleteProgram(progRef.current);
      glRef.current = null;
    };
  }, []);

  const draw = useCallback(() => {
    const gl = glRef.current;
    const program = progRef.current;
    const canvas = canvasRef.current;
    if (!gl || !program || !canvas) return;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform1f(gl.getUniformLocation(program, "u_threshold"), threshold);
    gl.uniform1f(gl.getUniformLocation(program, "u_gamma"), gamma);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }, [threshold, gamma]);

  // Threshold and gamma only touch uniforms, so this repaint allocates nothing
  // and never refetches. That is the whole point of colouring on the client.
  useEffect(() => {
    draw();
  }, [draw]);

  // ---- fetch the visible window, debounced -------------------------------
  const box = useMemo(() => {
    const w = Math.max(16, Math.round(meta.width / view.scale));
    const h = Math.max(16, Math.round(meta.height / view.scale));
    const x = Math.round(Math.min(Math.max(0, view.x), Math.max(0, meta.width - w)));
    const y = Math.round(Math.min(Math.max(0, view.y), Math.max(0, meta.height - h)));
    return { x, y, w, h };
  }, [meta.width, meta.height, view]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || error) return;
    let cancelled = false;
    const size = Math.min(2048, Math.max(512, Math.round(canvas.clientWidth * (window.devicePixelRatio || 1))));
    const url = `/api/run/${encodeURIComponent(runId)}/map/${encodeURIComponent(name)}/raster?x=${box.x}&y=${box.y}&w=${box.w}&h=${box.h}&size=${size}`;

    const timer = window.setTimeout(() => {
      setLoading(true);
      const image = new Image();
      image.decoding = "async";
      image.onload = () => {
        if (cancelled) return;
        const gl = glRef.current;
        if (!gl || !texRef.current) return;
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        gl.bindTexture(gl.TEXTURE_2D, texRef.current);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        setLoading(false);
        draw();
      };
      image.onerror = () => !cancelled && setError("The raster could not be loaded.");
      image.src = url;
    }, 140); // pan and zoom settle before the network is touched

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
    // draw is intentionally omitted: a threshold change must not refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, name, box.x, box.y, box.w, box.h, error]);

  // ---- pan and zoom ------------------------------------------------------
  const drag = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);

  // ---- region selection ---------------------------------------------------
  // The browser reports which rectangle was dragged and nothing else: the
  // transform, the lineage and the digests are the server's to derive, and a
  // second implementation here would be obliged to agree with the first.
  const anchor = useRef<{ x: number; y: number } | null>(null);
  const [selection, setSelection] = useState<Box | null>(null);

  const mapPoint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    return pointToMap(e.clientX, e.clientY, canvas.getBoundingClientRect(), view, box);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    (e.target as HTMLCanvasElement).setPointerCapture(e.pointerId);
    if (selecting) {
      anchor.current = mapPoint(e);
      setSelection(null);
      onSelect?.(null);
      return;
    }
    drag.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (selecting) {
      if (!anchor.current || !canvasRef.current) return;
      const next = boxFromDrag(anchor.current, mapPoint(e), meta);
      setSelection(next);
      onSelect?.(next);
      return;
    }
    const d = drag.current;
    const canvas = canvasRef.current;
    if (!d || !canvas) return;
    const perPixel = box.w / canvas.clientWidth;
    setView({
      x: d.vx - (e.clientX - d.x) * perPixel,
      y: d.vy - (e.clientY - d.y) * perPixel,
      scale: view.scale,
    });
  };
  const onPointerUp = () => {
    drag.current = null;
    anchor.current = null;
  };

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const factor = e.deltaY < 0 ? 1.25 : 1 / 1.25;
    const nextScale = Math.min(64, Math.max(1, view.scale * factor));
    if (nextScale === view.scale) return;
    // Keep the point under the cursor fixed while zooming.
    const rect = canvas.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    const anchorX = view.x + box.w * fx;
    const anchorY = view.y + box.h * fy;
    setView({
      x: anchorX - (meta.width / nextScale) * fx,
      y: anchorY - (meta.height / nextScale) * fy,
      scale: nextScale,
    });
  };

  if (error) return <div className="empty">{error}</div>;

  // The overlay is a plain element rather than another WebGL pass: it must be
  // crisp at every zoom, and it carries no information the shader has.
  const overlay = selection && {
    left: `${((selection.x0 - view.x) / box.w) * 100}%`,
    top: `${((selection.y0 - view.y) / box.h) * 100}%`,
    width: `${((selection.x1 - selection.x0) / box.w) * 100}%`,
    height: `${((selection.y1 - selection.y0) / box.h) * 100}%`,
  };
  const verdict = selection ? reviewBox(selection, meta) : null;

  return (
    <div className="mapviewer" style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{
          height,
          cursor: selecting ? "crosshair" : drag.current ? "grabbing" : "grab",
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        aria-label={`Mapa de probabilidad ${name}`}
      />
      {overlay && (
        <div
          className="roi-overlay"
          style={{ position: "absolute", pointerEvents: "none", ...overlay }}
          aria-hidden
        />
      )}
      <div className="mapstatus">
        <span>
          {box.w}×{box.h} px of {meta.width}×{meta.height} · {view.scale.toFixed(1)}×
        </span>
        {selection && (
          <span className={verdict?.ok ? "roi-ok" : "roi-refused"}>
            {selection.x0},{selection.y0}–{selection.x1},{selection.y1} ·{" "}
            {(coverage(selection, meta) * 100).toFixed(1)}% of the map
            {verdict && !verdict.ok ? ` · ${verdict.why}` : ""}
          </span>
        )}
        {loading && <span className="loading">loading…</span>}
        {view.scale > 1 && (
          <button onClick={() => setView({ x: 0, y: 0, scale: 1 })}>fit</button>
        )}
      </div>
    </div>
  );
});
