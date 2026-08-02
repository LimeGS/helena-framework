import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, Empty, Pill } from "../components/Bits";

/**
 * The HTTP surface, from FastAPI's own OpenAPI document.
 *
 * This tab used to render /api/docs -- modules, classes, functions and CLI
 * entry points -- which is a code reference wearing an API reference's name.
 * Somebody looking for "how do I queue a job over HTTP" got a list of Python
 * symbols instead. That material is Developer reference's subject.
 *
 * Generated rather than written, and from the running panel rather than from a
 * copy: FastAPI already builds this document out of the route decorators and
 * the Pydantic models, so a route added without touching this file appears
 * anyway, and one removed cannot linger here describing something that no
 * longer answers.
 */

const METHOD_ORDER = ["get", "post", "put", "patch", "delete"];

type Op = {
  method: string;
  path: string;
  summary: string;
  description: string;
  parameters: { name: string; in: string; required: boolean; schema: string }[];
  body: string | null;
  responses: { code: string; description: string }[];
};

/** The leading segment, which is how these routes are already organised:
 *  /api/segmentation/..., /api/jobs/..., /api/missions/... */
function groupOf(path: string): string {
  const parts = path.split("/").filter(Boolean);
  if (parts[0] !== "api") return "other";
  return parts[1] ?? "root";
}

/** A schema reduced to something that fits in a table cell. */
function typeName(schema: Record<string, unknown> | undefined): string {
  if (!schema) return "—";
  const ref = schema["$ref"];
  if (typeof ref === "string") return ref.split("/").pop() ?? "object";
  const union = schema["anyOf"] ?? schema["oneOf"];
  if (Array.isArray(union)) {
    return union
      .map((s) => typeName(s as Record<string, unknown>))
      .filter((n) => n !== "null")
      .join(" | ");
  }
  if (schema["type"] === "array") {
    return `${typeName(schema["items"] as Record<string, unknown>)}[]`;
  }
  return String(schema["type"] ?? "object");
}

function operations(doc: Record<string, unknown>): Op[] {
  const paths = (doc?.["paths"] ?? {}) as Record<string, Record<string, unknown>>;
  const out: Op[] = [];
  for (const [path, methods] of Object.entries(paths)) {
    for (const [method, raw] of Object.entries(methods)) {
      if (!METHOD_ORDER.includes(method)) continue;
      const spec = raw as Record<string, unknown>;
      const parameters = ((spec["parameters"] ?? []) as Record<string, unknown>[]).map((p) => ({
        name: String(p["name"]),
        in: String(p["in"]),
        required: Boolean(p["required"]),
        schema: typeName(p["schema"] as Record<string, unknown>),
      }));
      const content = (spec["requestBody"] as Record<string, unknown> | undefined)?.[
        "content"
      ] as Record<string, { schema?: Record<string, unknown> }> | undefined;
      const jsonBody = content?.["application/json"]?.schema;
      const responses = Object.entries(
        (spec["responses"] ?? {}) as Record<string, { description?: string }>,
      ).map(([code, r]) => ({ code, description: r?.description ?? "" }));
      out.push({
        method,
        path,
        summary: String(spec["summary"] ?? ""),
        description: String(spec["description"] ?? ""),
        parameters,
        body: jsonBody ? typeName(jsonBody) : null,
        responses,
      });
    }
  }
  return out;
}

function Operation({ op }: { op: Op }) {
  const [open, setOpen] = useState(false);
  // FastAPI puts the docstring's first line in summary and the whole docstring
  // in description, so printing both repeats the first line.
  const detail = op.description.startsWith(op.summary)
    ? op.description.slice(op.summary.length).trim()
    : op.description;
  const expandable = Boolean(detail || op.parameters.length || op.body);

  return (
    <div className="apiop">
      <button
        className="apiop-head"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        disabled={!expandable}
      >
        <Pill kind={op.method === "get" ? "ok" : "warn"}>{op.method.toUpperCase()}</Pill>
        <code>{op.path}</code>
        <span className="dim">{op.summary}</span>
      </button>

      {open && (
        <div className="body-pad">
          {detail && <p className="dim">{detail}</p>}

          {op.parameters.length > 0 && (
            <table className="apitable">
              <thead>
                <tr>
                  <th>parameter</th>
                  <th>in</th>
                  <th>type</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {op.parameters.map((p) => (
                  <tr key={`${p.in}-${p.name}`}>
                    <td>
                      <code>{p.name}</code>
                    </td>
                    <td className="dim">{p.in}</td>
                    <td className="dim">{p.schema}</td>
                    <td>{p.required ? <Pill kind="warn">required</Pill> : null}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {op.body && (
            <p>
              Request body <code>{op.body}</code>{" "}
              <span className="dim">as application/json</span>
            </p>
          )}

          <p className="dim">
            Responses:{" "}
            {op.responses.map((r) => `${r.code} ${r.description}`.trim()).join(" · ")}
          </p>
        </div>
      )}
    </div>
  );
}

export default function ApiReference() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["openapi"],
    queryFn: async () => {
      const response = await fetch("/api/openapi.json");
      if (!response.ok) {
        throw new Error(`the panel did not serve its OpenAPI document (${response.status})`);
      }
      return response.json() as Promise<Record<string, unknown>>;
    },
  });

  const groups = useMemo(() => {
    if (!data) return [];
    const byGroup = new Map<string, Op[]>();
    for (const op of operations(data)) {
      byGroup.set(groupOf(op.path), [...(byGroup.get(groupOf(op.path)) ?? []), op]);
    }
    return [...byGroup.entries()]
      .map(([name, ops]) => ({
        name,
        ops: ops.sort(
          (a, b) =>
            a.path.localeCompare(b.path) ||
            METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method),
        ),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [data]);

  if (isLoading) return <Empty>reading the API surface…</Empty>;
  if (error) return <Empty>{String(error)}</Empty>;

  const total = groups.reduce((n, g) => n + g.ops.length, 0);

  return (
    <>
      <Card title="The HTTP API" note={`${total} operations`}>
        <div className="body-pad">
          <p className="dim">
            Generated from the running panel's OpenAPI document, so it describes
            this deployment rather than a copy of it. Every route except{" "}
            <code>/api/session</code> and <code>/api/session/bootstrap</code>{" "}
            needs the session cookie that <code>POST /api/session</code> sets. A
            machine token reaches <code>/api/artifacts/</code> and nothing else.
          </p>
          <p className="dim">
            An interactive form is at <a href="/api/swagger">/api/swagger</a> and
            the raw document at <a href="/api/openapi.json">/api/openapi.json</a>. Python
            modules and CLI entry points are under Developer reference.
          </p>
        </div>
      </Card>

      {groups.map((g) => (
        <Card key={g.name} title={`/api/${g.name}`} note={`${g.ops.length}`}>
          {g.ops.map((op) => (
            <Operation key={`${op.method}-${op.path}`} op={op} />
          ))}
        </Card>
      ))}
    </>
  );
}
