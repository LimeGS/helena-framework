import { memo, useDeferredValue, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { failure, useConfig, type Constant, type EnvSetting } from "../api";
import { FleetSecrets } from "../components/FleetSecrets";
import { Card, Empty, Pill , queryGate} from "../components/Bits";
import { applyTheme, readTheme, type Theme } from "../theme";

/**
 * Configuration is versioned as a whole. Changing one value snapshots the
 * entire settings map, hashes it and gives it an id, because the question asked
 * afterwards is what the configuration was when a run happened -- unanswerable
 * if values move independently. Restoring writes a new version equal to the old
 * one rather than rewinding, so going back stays visible in the record.
 */

type Version = {
  version_id: string; index: number; content_sha256: string;
  parent_id: string | null; restored_from: string | null;
  created_at_utc: string; reason: string;
  changed: Record<string, { from: string | null; to: string | null }>;
};

function Info({ setting }: { setting: EnvSetting }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="infowrap">
      <button
        className="infobtn"
        aria-label={`What ${setting.name} controls`}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        i
      </button>
      {open && (
        <span className="infopop" role="tooltip">
          <b>{setting.name}</b>
          <span>{setting.doc}</span>
          <span className="infometa">
            type: <code>{setting.kind}</code>
          </span>
          {setting.allowed && (
            <span className="infometa">
              accepts: {setting.allowed.map((a) => <code key={a}>{a}</code>)}
            </span>
          )}
          {setting.example && (
            <span className="infometa">
              example: <code>{setting.example}</code>
            </span>
          )}
          <span className="infometa">
            default: <code>{setting.default === "" ? "(empty)" : setting.default}</code>
          </span>
          {setting.requires_restart && (
            <span className="infometa warn">read at startup — a change needs a restart</span>
          )}
          {setting.secret && (
            <span className="infometa warn">holds a credential; it is stored in clear text</span>
          )}
        </span>
      )}
    </span>
  );
}

const EnvRow = memo(function EnvRow({
  setting, exists, onSave, onClear, busy,
}: {
  setting: EnvSetting;
  exists?: boolean;
  onSave: (name: string, value: string) => void;
  onClear: (name: string) => void;
  busy: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const editing = draft !== null;
  const value = editing ? draft : setting.value;

  return (
    <tr>
      <td className="l grow">
        <code>{setting.name}</code>
      </td>
      <td className="l wrap">
        {setting.allowed ? (
          <select
            value={value}
            disabled={busy}
            onChange={(e) => setDraft(e.target.value)}
          >
            {setting.allowed.map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        ) : (
          <input
            className="cfgvalue"
            type={setting.secret ? "password" : "text"}
            value={value}
            disabled={busy}
            placeholder={setting.example || setting.default}
            onChange={(e) => setDraft(e.target.value)}
          />
        )}
      </td>
      <td className="l">
        {setting.source === "override" ? (
          <Pill kind="run">override</Pill>
        ) : setting.source === "environment" ? (
          <Pill kind="ok">environment</Pill>
        ) : (
          <Pill kind="neg">default</Pill>
        )}
      </td>
      <td className="l">
        {exists === undefined ? (
          <span className="dash">—</span>
        ) : exists ? (
          <Pill kind="ok">exists</Pill>
        ) : (
          <Pill kind="crit">missing</Pill>
        )}
      </td>
      <td className="l">
        <div className="rowactions">
          {editing && draft !== setting.value && (
            <button disabled={busy} onClick={() => { onSave(setting.name, draft); setDraft(null); }}>
              save
            </button>
          )}
          {editing && (
            <button disabled={busy} onClick={() => setDraft(null)}>cancel</button>
          )}
          {!editing && setting.source === "override" && (
            <button disabled={busy} onClick={() => onClear(setting.name)}>reset</button>
          )}
          <Info setting={setting} />
        </div>
      </td>
    </tr>
  );
});

const ConstRow = memo(function ConstRow({
  c, onSave, busy,
}: { c: Constant; onSave: (c: Constant, value: string) => void; busy: boolean }) {
  const serialised = JSON.stringify(c.value);
  const [draft, setDraft] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const editing = draft !== null;
  return (
    <tr>
      <td className="l grow"><code>{c.name}</code></td>
      <td className="l wrap">
        <input
          className="cfgvalue"
          value={editing ? draft : serialised}
          disabled={busy}
          onChange={(e) => setDraft(e.target.value)}
        />
      </td>
      <td className="l">{c.module}</td>
      <td className="l wrap">{c.path}:{c.line}</td>
      <td className="l">
        <div className="rowactions">
          {editing && draft !== serialised && (
            <button disabled={busy} onClick={() => { onSave(c, draft); setDraft(null); }}>
              save
            </button>
          )}
          {editing && <button disabled={busy} onClick={() => setDraft(null)}>cancel</button>}
          <span className="infowrap">
            <button className="infobtn" aria-label={`About ${c.name}`}
                    aria-expanded={open} onClick={() => setOpen((v) => !v)}>i</button>
            {open && (
              <span className="infopop" role="tooltip">
                <b>{c.name}</b>
                <span>
                  A module-level constant compiled into the framework. Editing it rewrites{" "}
                  <code>{c.path}</code> in place — a code change, left uncommitted so it shows up
                  as a diff.
                </span>
                <span className="infometa">
                  type: <code>{typeof c.value === "object" ? "list/dict" : typeof c.value}</code>{" "}
                  · the type is held: a number cannot become a string
                </span>
                <span className="infometa">current: <code>{serialised}</code></span>
              </span>
            )}
          </span>
        </div>
      </td>
    </tr>
  );
});

function SubSection({
  title, count, children,
}: { title: string; count: number; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="subsection">
      <button className="subsection-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className={`chevron ${open ? "open" : ""}`} aria-hidden="true">▸</span>
        <span className="subsection-title">{title}</span>
        <span className="subsection-count">{count}</span>
      </button>
      {open && children}
    </div>
  );
}

const THEMES = [
  ["auto", "Auto", "follows the system"],
  ["light", "Light", "Parchment"],
  ["dark", "Dark", "Obsidian"],
] as const satisfies readonly (readonly [Theme, string, string])[];

/**
 * The one setting on this page that is not versioned. Everything else here is
 * a fact about how a run was produced; which of two palettes an operator reads
 * it in is not, so it stays in this browser and out of the config hash.
 */
function Appearance() {
  const [theme, setTheme] = useState(readTheme);
  return (
    <div className="body-pad">
      <p>
        Light or dark, remembered in this browser only. Auto asks the operating
        system and keeps asking, so a machine that dims itself in the evening
        takes the panel with it.
      </p>
      <div className="themepick" role="group" aria-label="Colour theme">
        {THEMES.map(([value, label, hint]) => (
          <button
            key={value}
            aria-pressed={theme === value}
            onClick={() => { applyTheme(value); setTheme(value); }}
          >
            {label}
            <span className="hint">{hint}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function Section({
  title, note, defaultOpen = false, children,
}: { title: string; note?: React.ReactNode; defaultOpen?: boolean; children: React.ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="card">
      <button className="section-head" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <span className={`chevron ${open ? "open" : ""}`} aria-hidden="true">▸</span>
        <h2>{title}</h2>
        {note && <span className="note">{note}</span>}
      </button>
      {open && children}
    </section>
  );
}

export default function Config() {
  const client = useQueryClient();
  const { data, isLoading, error } = useConfig();
  const [text, setText] = useState("");
  const deferred = useDeferredValue(text);

  const versions = useQuery({
    queryKey: ["config-versions"],
    queryFn: async () => {
      const r = await fetch("/api/config/versions");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { versions: Version[]; current_id: string | null };
    },
    staleTime: 10_000,
  });

  const refresh = () => {
    client.invalidateQueries({ queryKey: ["config"] });
    client.invalidateQueries({ queryKey: ["config-versions"] });
  };

  const saveEnv = useMutation({
    mutationFn: async ({ name, value }: { name: string; value: string }) => {
      const r = await fetch(`/api/config/env/${encodeURIComponent(name)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ value, reason: `set ${name} from the panel` }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: refresh,
  });

  const clearEnv = useMutation({
    mutationFn: async (name: string) => {
      const r = await fetch(`/api/config/env/${encodeURIComponent(name)}`, { method: "DELETE" });
      if (!r.ok) throw await failure(r);
      return r.json();
    },
    onSuccess: refresh,
  });

  const saveConstant = useMutation({
    mutationFn: async ({ c, value }: { c: Constant; value: string }) => {
      const r = await fetch("/api/config/constant", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: c.path, name: c.name, value,
                               reason: "edited from the panel" }),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: refresh,
  });

  const restore = useMutation({
    mutationFn: async (versionId: string) => {
      const r = await fetch(`/api/config/versions/${versionId}/restore`, { method: "POST" });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`);
      return body;
    },
    onSuccess: refresh,
  });

  const constants = useMemo(() => {
    if (!data) return [];
    const needle = deferred.trim().toLowerCase();
    if (!needle) return data.constants;
    return data.constants.filter(
      (c) => c.name.toLowerCase().includes(needle) ||
             c.module.toLowerCase().includes(needle) ||
             c.path.toLowerCase().includes(needle),
    );
  }, [data, deferred]);

  // Rendered on both sides of the gate: the theme needs nothing from the API,
  // and an unreachable backend is the worst moment to lose the way out of a
  // white screen.
  const appearance = (
    <Section title="Appearance" note="this browser only">
      <Appearance />
    </Section>
  );

  const gate = queryGate({ isLoading, error, data }, "reading the configuration surface…");
  if (gate) return <>{appearance}{gate}</>;
  // The gate covers every unset case; the compiler cannot see that
  // through a helper.
  if (!data) return null;
  const busy = saveEnv.isPending || clearEnv.isPending || saveConstant.isPending || restore.isPending;
  const groups = [...new Set(constants.map((c) => c.group))];
  const anyError = saveEnv.error ?? clearEnv.error ?? saveConstant.error ?? restore.error;

  return (
    <>
      {appearance}
      <FleetSecrets />
      {anyError && (
        <Card title="That change was refused">
          <div className="body-pad"><p>{String(anyError)}</p></div>
        </Card>
      )}

      <Section
        title="Version"
        note={data.version ? (
          <>
            <code>{data.version.version_id}</code> · {versions.data?.versions.length ?? 0} in history
          </>
        ) : "nothing committed yet"}
      >
        <div className="body-pad">
          <p>
            Configuration is versioned whole. Any change snapshots every setting, hashes it and
            gives it an id. Restoring writes a new version equal to the old one rather than
            rewinding, so going back stays in the record.
          </p>
        </div>
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l">Version</th>
                <th className="l">Hash</th>
                <th className="l">When</th>
                <th className="l">Why</th>
                <th className="l">Changed</th>
                <th className="l"></th>
              </tr>
            </thead>
            <tbody>
              {(versions.data?.versions ?? []).map((v) => {
                const active = v.version_id === versions.data?.current_id;
                return (
                  <tr key={v.version_id} className={active ? "" : "muted"}>
                    <td className="l">
                      <code>{v.version_id}</code>
                      {active && <> <Pill kind="ok">active</Pill></>}
                    </td>
                    <td className="l"><code>{v.content_sha256.slice(0, 12)}</code></td>
                    <td className="l">{v.created_at_utc.slice(0, 19).replace("T", " ")}</td>
                    <td className="l wrap">
                      {v.restored_from ? <>restored <code>{v.restored_from}</code></> : v.reason}
                    </td>
                    <td className="l wrap">
                      {Object.entries(v.changed).map(([k, d]) => (
                        <div key={k}>
                          <code>{k}</code>: {d.from ?? "—"} → {d.to ?? "—"}
                        </div>
                      ))}
                    </td>
                    <td className="l">
                      {!active && (
                        <button disabled={busy} onClick={() => restore.mutate(v.version_id)}>
                          restore
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {!versions.data?.versions.length && <Empty>no version committed yet</Empty>}
      </Section>

      <Section
        title="Environment"
        note={`${data.environment.filter((e) => e.source !== "default").length} of ${data.environment.length} set explicitly`}
      >
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l grow">Setting</th>
                <th className="l">Value</th>
                <th className="l">Source</th>
                <th className="l">Path</th>
                <th className="l"></th>
              </tr>
            </thead>
            <tbody>
              {data.environment.map((e) => (
                <EnvRow
                  key={e.name} setting={e} exists={data.paths_exist[e.name]} busy={busy}
                  onSave={(name, value) => saveEnv.mutate({ name, value })}
                  onClear={(name) => clearEnv.mutate(name)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section
        title="Framework constants"
        note={`${data.constants.length} module-level values, read from the source`}
      >
        <div className="body-pad">
          <input
            className="search" type="search" value={text}
            placeholder="filter by name, module or path…"
            onChange={(e) => setText(e.target.value)}
          />
          <p>
            These are compiled in. Editing one rewrites its source file in place and leaves the
            change uncommitted, so it shows up as a diff — a threshold that decides whether a
            screen passes should leave a trace in git, not only in a database.
          </p>
        </div>
        {groups.map((g) => {
          const inGroup = constants.filter((c) => c.group === g);
          return (
          <SubSection key={g} title={g} count={inGroup.length}>
            <div className="scroller">
              <table>
                <thead>
                  <tr>
                    <th className="l grow">Constant</th>
                    <th className="l">Value</th>
                    <th className="l">Module</th>
                    <th className="l">Defined at</th>
                    <th className="l"></th>
                  </tr>
                </thead>
                <tbody>
                  {inGroup.map((c) => (
                    <ConstRow key={`${c.path}:${c.name}`} c={c} busy={busy}
                              onSave={(constant, value) => saveConstant.mutate({ c: constant, value })} />
                  ))}
                </tbody>
              </table>
            </div>
          </SubSection>
          );
        })}
        {constants.length === 0 && <Empty>nothing matches «{text}»</Empty>}
      </Section>
    </>
  );
}
