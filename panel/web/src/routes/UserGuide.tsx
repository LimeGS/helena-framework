import { useQuery } from "@tanstack/react-query";
import { Card, Empty, Mark, Pill, queryGate } from "../components/Bits";
import { GUIDE, type Figure as FigureSpec, type PhaseGuide } from "./guide-content";
import { AREAS } from "./guide-controls";

// Counted here rather than inside the JSX attribute. An arrow function in a
// `note={...}` puts a `>` inside the opening tag, which is enough to defeat
// anything reading this file as text -- including the test that checks every
// card opens folded shut.
const CONTROL_COUNT =
  `${AREAS.reduce((n, a) => n + a.controls.length, 0)} controls · ${AREAS.length} pages`;

// Imported rather than served from a public directory: the panel serves static
// files from /assets only, so a picture under any other path falls through to
// the single-page fallback and renders as a broken image with a caption
// explaining it. Importing also gets each one a content hash, so a corrected
// screenshot is never served from a stale cache.
import configurationPng from "../assets/guide/configuration.png";
import missionPng from "../assets/guide/mission.png";
import phaseAnatomyPng from "../assets/guide/phase-anatomy.png";
import queueFormPng from "../assets/guide/queue-form.png";
import seedersPng from "../assets/guide/seeders.png";
import sidebarPng from "../assets/guide/sidebar.png";

const FIGURES: Record<string, string> = {
  configuration: configurationPng,
  mission: missionPng,
  "phase-anatomy": phaseAnatomyPng,
  "queue-form": queueFormPng,
  seeders: seedersPng,
  sidebar: sidebarPng,
};

/**
 * Every control in the panel, and what to do with it.
 *
 * A reference, not a walkthrough -- the walkthrough is Tutorial.tsx, and
 * splitting them is why this one can afford to be exhaustive. A page trying to
 * be both was the path with the reference missing: it told you to leave things
 * on their defaults without ever saying what the defaults were for.
 *
 * Two halves. The written half — the order you make decisions in, what to look
 * at afterwards, how a phase goes wrong while reporting success — is in
 * `guide-content.ts`. The machine-readable half is fetched from the same
 * endpoints the forms use: the phase contracts, every queueable field, the
 * segmentation options and the seeders. That half cannot drift, because a field
 * added to the queue appears here the moment it exists.
 */

type Contract = {
  id: string; slug: string; name: string; one_line: string;
  consumes: string; produces: string; how_to_run: string; how_it_fails: string;
  gate: string | null; maturity: string; lives_in: string[];
};

type Field = {
  name: string; type: string; required: boolean; lane: string | null;
  label: string; note: string | null; placeholder: string | null;
  filled_by_deployment: boolean;
};

type Schema = {
  available: boolean; reason?: string; fields: Field[];
  lanes?: { id: string; name: string; note: string | null; validated: string | null }[];
  exactly_one_of?: { lane: string | null; names: string[] }[];
};

type Seeder = {
  id: string; name: string; kind: string; repeatable: boolean; note: string;
  configures: { field: string; label: string; type: string; note?: string;
                options?: string[]; suggestions?: string[] }[];
};

type Option = {
  flag: string; field: string; kind: string; label: string; group: string;
  note: string; choices?: string[]; off_flag?: string;
};

type Growth = {
  note: string;
  parameters: { name: string; range: string; default: unknown }[];
};

/* ---------------------------------------------------------------- pieces */

/** A picture of the thing being described, with the caption doing the teaching.
 *  Lazy, because ten modules of screenshots on one page is a page that loads
 *  slowly to show you something you have not scrolled to. */
function Figure({ src, alt, caption }: FigureSpec) {
  return (
    <figure className="guide-figure">
      <img src={FIGURES[src]} alt={alt} loading="lazy" />
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

function Steps({ steps }: { steps: PhaseGuide["steps"] }) {
  return (
    <ol className="guide-steps">
      {steps.map((s) => (
        <li key={s.do}>
          <b>{s.do}</b>
          <span>{s.why}</span>
        </li>
      ))}
    </ol>
  );
}

function FieldTable({ fields }: { fields: Field[] }) {
  const shown = fields.filter((f) => !f.filled_by_deployment);
  if (!shown.length) return null;
  return (
    <div className="scroller">
      <table>
        <thead>
          <tr>
            <th className="l">Field</th>
            <th className="l">Type</th>
            <th className="l grow">What to put in it</th>
            <th className="l">Only for</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((f) => (
            <tr key={f.name + (f.lane ?? "")}>
              <td className="l scrollid">
                {f.label}
                {f.required && <> <Pill kind="warn">required</Pill></>}
              </td>
              <td className="l">{f.type}</td>
              <td className="l grow">
                {f.note}
                {f.placeholder && <> <code>{f.placeholder}</code></>}
              </td>
              <td className="l">
                {f.lane ? <code>{f.lane}</code> : <span className="dash">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** P1 has no queue schema: its work is planned rather than parameterised, and
 *  its controls are the seeders and the eighteen options. Both are served. */
function SegmentationDetail() {
  const seeders = useQuery({
    queryKey: ["segmentation-seeders"],
    queryFn: async () => {
      const r = await fetch("/api/segmentation");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { planners: Seeder[] };
    },
    staleTime: 5 * 60 * 1000,
  });
  const options = useQuery({
    queryKey: ["segmentation-options"],
    queryFn: async () => {
      const r = await fetch("/api/segmentation/options");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { options: Option[]; growth: Growth };
    },
    staleTime: 5 * 60 * 1000,
  });

  const groups = [...new Set((options.data?.options ?? []).map((o) => o.group))];

  return (
    <>
      <h4>The seeders</h4>
      <p>
        A seeder decides, inside a cell the sweep has already chosen, which single
        point to grow from. They differ in what they are allowed to consider and
        in what they cost. Given the same cell they pick different points and
        grow different sheets — both correct — which is why the seeder is stamped
        on every task.
      </p>
      {seeders.data ? (
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l">Seeder</th>
                <th className="l">Kind</th>
                <th className="l grow">What it does, and when to reach for it</th>
                <th className="l">Takes</th>
              </tr>
            </thead>
            <tbody>
              {seeders.data.planners.map((s) => (
                <tr key={s.id}>
                  <td className="l scrollid">
                    {s.name}
                    <div className="dim"><code>{s.id}</code></div>
                  </td>
                  <td className="l">
                    {s.kind}
                    {s.repeatable
                      ? <div><Pill kind="ok">reproducible</Pill></div>
                      : <div><Pill kind="warn">varies per run</Pill></div>}
                  </td>
                  <td className="l grow">{s.note}</td>
                  <td className="l">
                    {s.configures.length === 0
                      ? <span className="dash">nothing</span>
                      : s.configures.map((c) => (
                          <div key={c.field}>
                            <code>{c.label}</code>
                            {c.note && <div className="dim">{c.note}</div>}
                            {c.options && (
                              <div className="dim">one of: {c.options.join(", ")}</div>
                            )}
                          </div>
                        ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>reading the seeders…</Empty>
      )}
      <p className="dim">
        <b>Reproducible</b> means the same queue and the same history give the same
        seed every time. The router and the model lanes do not promise that: they
        ask something outside the deterministic rule, so two runs of the same cell
        can differ. That is a legitimate choice — it is also why the seeder is part
        of a surface's identity.
      </p>

      <h4>The options</h4>
      <p>
        Eighteen parameters, grouped by <em>when</em> they are decided. The first
        group is fixed when you queue and decides which cells exist at all; the
        rest are read per attempt by the seeder that runs.
      </p>
      {groups.map((group) => (
        <div key={group}>
          <h5>{group}</h5>
          <div className="scroller">
            <table>
              <thead>
                <tr>
                  <th className="l">Option</th>
                  <th className="l">Flag</th>
                  <th className="l grow">What it does</th>
                </tr>
              </thead>
              <tbody>
                {(options.data?.options ?? [])
                  .filter((o) => o.group === group)
                  .map((o) => (
                    <tr key={o.field}>
                      <td className="l scrollid">{o.label}</td>
                      <td className="l"><code>{o.flag}</code></td>
                      <td className="l grow">
                        {o.note}
                        {o.choices && (
                          <div className="dim">
                            one of: {o.choices.map((c) => <code key={c}>{c}</code>)}
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      <h4>VC3D's own growth parameters</h4>
      <p>{options.data?.growth.note}</p>
      <div className="scroller">
        <table>
          <thead>
            <tr>
              <th className="l">Parameter</th>
              <th className="l">Range</th>
              <th className="l grow">Default</th>
            </tr>
          </thead>
          <tbody>
            {(options.data?.growth.parameters ?? []).map((p) => (
              <tr key={p.name}>
                <td className="l scrollid"><code>{p.name}</code></td>
                <td className="l">{p.range}</td>
                <td className="l grow">{String(p.default)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="dim">
        <code>generations</code> and <code>step_size</code> together set how far a
        surface is allowed to travel from its seed: more generations grow further
        and cost proportionally more, a larger step covers ground faster and
        follows curvature worse. <code>min_area_cm</code> is pinned at zero so a
        small surface is still recorded rather than discarded, and{" "}
        <code>use_cuda</code> is pinned false because this stage is CPU-bound —
        the cards belong to rendering and inference.
      </p>
    </>
  );
}

function PhaseModule({ contract }: { contract: Contract }) {
  const schema = useQuery({
    queryKey: ["phase-schema", contract.id],
    queryFn: async () => {
      const r = await fetch(`/api/phases/${contract.id}/parameters`);
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as Schema;
    },
    staleTime: 5 * 60 * 1000,
  });

  const guide = GUIDE[contract.id];
  const deployment = (schema.data?.fields ?? []).filter((f) => f.filled_by_deployment);

  return (
    <details className="guide-phase">
      <summary>
        <span className="rail-id">{contract.id}</span>
        <strong>{contract.name}</strong>
        <span className="dim">{contract.one_line}</span>
      </summary>

      <div className="body-pad">
        <h4>What you are doing</h4>
        <p>{guide?.purpose ?? contract.one_line}</p>

        <dl className="guide-facts">
          <dt>Takes</dt><dd>{contract.consumes}</dd>
          <dt>Produces</dt><dd>{contract.produces}</dd>
        </dl>

        <h4>Before you start</h4>
        <ul>
          {(guide?.before ?? []).map((b) => <li key={b}>{b}</li>)}
        </ul>

        <h4>Doing it, in order</h4>
        <Steps steps={guide?.steps ?? []} />

        {guide?.figure && <Figure {...guide.figure} />}

        {contract.id === "P1" && <SegmentationDetail />}

        {schema.data?.lanes && schema.data.lanes.length > 1 && (
          <>
            <h4>Lanes</h4>
            <ul>
              {schema.data.lanes.map((lane) => (
                <li key={lane.id}>
                  <code>{lane.id}</code> — {lane.name}
                  {lane.note && <>. {lane.note}</>}
                  {lane.validated && <> <Pill kind="ok">{lane.validated}</Pill></>}
                </li>
              ))}
            </ul>
          </>
        )}

        {schema.data?.available && (
          <>
            <h4>Every field on the form</h4>
            <FieldTable fields={schema.data.fields} />
            {(schema.data.exactly_one_of ?? []).map((rule) => (
              <p key={rule.names.join()} className="dim">
                Give exactly one of{" "}
                {rule.names.map((n) => <code key={n}>{n}</code>)
                  .reduce((a, b) => <>{a} or {b}</>)}
                {rule.lane && <> on the <code>{rule.lane}</code> lane</>}.
              </p>
            ))}
            {deployment.length > 0 && (
              <p className="dim">
                Filled in by the deployment rather than asked of you:{" "}
                {deployment.map((f) => f.name).join(", ")}.
              </p>
            )}
          </>
        )}

        {guide?.controls && guide.controls.length > 0 && (
          <>
            <h4>The other controls on the page</h4>
            <dl className="guide-facts">
              {guide.controls.map((c) => (
                <div key={c.name} style={{ display: "contents" }}>
                  <dt>{c.name}</dt>
                  <dd>{c.what}</dd>
                </div>
              ))}
            </dl>
          </>
        )}

        <h4>Reading the result</h4>
        <ul>
          {(guide?.reading ?? []).map((r) => <li key={r}>{r}</li>)}
        </ul>

        <h4>How it goes wrong</h4>
        <p>{contract.how_it_fails}</p>
        <ul>
          {(guide?.traps ?? []).map((t) => <li key={t}>{t}</li>)}
        </ul>
        {contract.gate && (
          <p><Pill kind="warn">gate</Pill> {contract.gate}</p>
        )}

        {contract.lives_in?.length > 0 && (
          <p className="dim">
            Implemented in {contract.lives_in.map((path) => (
              <code key={path}>{path}</code>
            ))}
          </p>
        )}
      </div>
    </details>
  );
}

/* ------------------------------------------------------------------ page */

export default function UserGuide() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["phase-contracts"],
    queryFn: async () => {
      const r = await fetch("/api/phases");
      if (!r.ok) throw new Error(String(r.status));
      return (await r.json()) as { phases: Contract[] };
    },
    staleTime: 5 * 60 * 1000,
  });

  const gate = queryGate({ isLoading, error, data }, "reading the phase contracts…");
  if (gate) return gate;
  if (!data) return null;

  return (
    <>
      <Card title="What this is" note="start here" collapsed>
        <div className="body-pad guide-prose">
          <p>
            This platform turns a CT scan of a rolled scroll into a readable page.
            It does that in ten steps, called phases. Each phase does one thing,
            writes down what it did, and hands the next one something you can
            check.
          </p>
          <p>
            It is not a one-click reader. You drive it, one phase at a time, and
            you decide at each step whether the result is worth continuing from.
          </p>
          <p>
            Every section on this page is folded shut. Open the ones you need —
            the ten modules further down are the reference you will come back to.
          </p>
        </div>
      </Card>

      <Card title="The ten phases" note="one line each" collapsed>
        <div className="body-pad guide-prose">
          <ul className="guide-flow">
            <li><b>P0 · Volume intake</b> Pick the scan. Pin its scale.</li>
            <li><b>P1 · Segmentation</b> Find sheet surfaces inside it.</li>
            <li><b>P2 · Geometry certification</b> Decide which of those are one
              clean sheet.</li>
            <li><b>P3 · Flattening</b> Unroll a certified sheet flat.</li>
            <li><b>P4 · Surface volume rendering</b> Slice the CT along the sheet
              into a stack of images.</li>
            <li><b>P5 · Ink detection</b> Run a model over that stack.</li>
            <li><b>P6 · Liveness</b> Ask whether the result says anything at all.</li>
            <li><b>P7 · Screening</b> Decide whether it looks like text.</li>
            <li><b>P8 · Reconstruction</b> Stitch sheets into one page.</li>
            <li><b>P9 · Rendering</b> Compose that page and try to read it.</li>
          </ul>
          <p>
            You will rarely run all ten. Most work stops at P5 or P7. A phase that
            produced nothing is still a result — write it down.
          </p>
        </div>
      </Card>

      <Card title="Where things are" note="the panel, in two pictures" collapsed>
        <div className="body-pad guide-prose">
          <Figure
            src="sidebar"
            alt="The pipeline sidebar, listing the ten phases with a status mark"
            caption={
              "The sidebar is always there. Each phase shows a coloured edge and a " +
              "mark saying what it is doing right now. Hover any of them and the " +
              "panel tells you why in a sentence. The scroll you are working on is " +
              "chosen below the list; the mission is at the bottom."
            }
          />
          {/* The same drawn marks the rail uses, so the legend cannot teach a
              shape the panel has stopped showing. */}
          <dl className="guide-marks">
            <div><dt><Mark status="running" /></dt><dd>running on a worker</dd></div>
            <div><dt><Mark status="queued" /></dt><dd>queued — nobody has picked it up</dd></div>
            <div><dt><Mark status="failed" /></dt><dd>the last attempt failed</dd></div>
            <div><dt><Mark status="stopped" /></dt><dd>the last attempt was cancelled</dd></div>
            <div><dt><Mark status="done" /></dt><dd>it has produced something</dd></div>
            <div><dt><Mark status="ready" /></dt><dd>ready — you can run it</dd></div>
            <div><dt><Mark status="blocked" /></dt><dd>not ready — something is missing</dd></div>
            <div><dt><Mark status="waiting" /></dt><dd>nothing upstream yet</dd></div>
            <div><dt><Mark status="no-run" /></dt><dd>nothing to run here</dd></div>
            <div><dt><Mark status="elsewhere" /></dt><dd>run somewhere other than this deployment</dd></div>
          </dl>

          <Figure
            src="phase-anatomy"
            alt="The top of a phase page: title, maturity pills, tab bar and summary cards"
            caption={
              "Every phase page looks like this. The title says what the phase is " +
              "for; the ⓘ next to it opens the full contract. The tabs split what " +
              "exists now from the form that makes more of it — the last tab is " +
              "always the one that queues work. The cards below count what this " +
              "phase has done for the scroll you selected."
            }
          />
        </div>
      </Card>

      <Card title="Before you can run anything" note="a mission and a scroll" collapsed>
        <div className="body-pad guide-prose">
          <p>
            <b>A mission.</b> It is the campaign you are working in, and it holds
            the list of scrolls you are attempting. Everything you see is counted
            against it.
          </p>
          <Figure
            src="mission"
            alt="The Mission page listing missions with their scroll, run and job counts"
            caption={
              "Mission → open one, or create one. If a page looks empty and you " +
              "expected work on it, check here first: you are probably in a " +
              "different mission from the one the work belongs to."
            }
          />
          <p>
            <b>A scroll.</b> Pick it in the selector under the phase list. A phase
            runs on one scroll at a time, so this is what the numbers on every page
            refer to.
          </p>
        </div>
      </Card>

      <Card title="Running a phase" note="the same four moves every time" collapsed>
        <div className="body-pad guide-prose">
          <ol className="guide-steps">
            <li>
              <b>Open the phase in the sidebar.</b>
              <span>The page opens on what exists now, not on the form.</span>
            </li>
            <li>
              <b>Go to its last tab.</b>
              <span>That is always the one that queues work.</span>
            </li>
            <li>
              <b>Fill the form and press Queue.</b>
              <span>
                Every field prints its explanation underneath, because the form is
                generated from the queue's own list of parameters.
              </span>
            </li>
            <li>
              <b>Watch the sidebar.</b>
              <span>
                A worker picks the job up and runs it on its own machine. The panel
                never runs anything, so closing this page does not stop the work.
              </span>
            </li>
          </ol>
        </div>
      </Card>

      <Card title="Four things that will save you a day" collapsed>
        <div className="body-pad guide-prose">
          <p>
            <b>Running the same thing twice does nothing.</b> Work is identified by
            what it is, not by when you asked. Queue the same cells under the same
            policy version and the platform inserts nothing and still says it
            succeeded. To really ask again, change the policy version.
          </p>
          <p>
            <b>Success is not the same as a usable result.</b> Every phase here can
            finish cleanly and produce something worthless: a render at the wrong
            depth, a map that is one flat value, a page stitched out of sheets that
            do not touch. Each module below has a <em>Reading the result</em>{" "}
            section for exactly this. Read it before you trust a number.
          </p>
          <p>
            <b>Scale is the thread through everything.</b> Microns per voxel is
            fixed in P0, sets the depth in P4, decides whether the model in P5 sees
            the thickness it was trained on, and decides whether a shape in P7 is
            letter-sized. Most silent failures here are a scale mismatch.
          </p>
          <p>
            <b>Nothing here proves ink.</b> A strong response is a reason for a
            human to look, and nothing more, all the way to the last phase.
          </p>
        </div>
      </Card>

      <Card title="The rest of the panel" collapsed>
        <div className="body-pad guide-prose">
          <Figure
            src="configuration"
            alt="The Configuration page with its Settings, Hosts, Users, Lineage and Audit log tabs"
            caption={
              "Configuration belongs to the deployment, not to your mission: " +
              "settings and where they came from, the modules each phase can run, " +
              "the model weights on this machine, the fleet, the accounts, where " +
              "each artefact came from, and a log of every change anybody made. " +
              "You can open it without picking a mission first."
            }
          />
          <p>
            <b>Models</b> is where weights arrive. A profile identifies a
            checkpoint by its SHA-256 and treats the path as something you supply
            at run time, so this page is not a catalogue of models somebody liked:
            it is the list of hashes the frozen profiles ask for, each shown as
            installed or missing. Ask it to look the missing ones up and it finds
            the repository that publishes exactly those bytes, so a download is
            verified before it starts and again when it lands. You can also give
            it any Hugging Face repository and tag; that file is fetched and
            hashed, and nothing will use it until some profile declares its hash.
          </p>
          <p>
            It fetches <code>.safetensors</code> and nothing else. Every other
            checkpoint format here is a Python pickle, which executes whatever was
            serialised into it the moment it loads, on a GPU worker. One
            checkpoint the profiles name is published only as a pickle; the page
            says so and offers no button, because converting it is a decision
            somebody should make deliberately rather than by clicking.
          </p>
        </div>
      </Card>

      {/*
        The static half. Buttons, filters and page-level dropdowns cannot be
        fetched from a schema the way queue fields can, so they are written down
        in guide-controls.ts -- and a test fails when a page grows a control and
        this does not, which is the only thing that keeps "every control" true.
      */}
      <Card title="Every control, page by page" note={CONTROL_COUNT} collapsed>
        <div className="body-pad guide-prose">
          <p>
            What each control does, when it is the right one to reach for, and what
            to leave it on. The queue forms are documented further down instead,
            straight from the schema the forms themselves use.
          </p>
          {AREAS.map((area) => (
            <section key={area.page}>
              <h3>{area.title}</h3>
              <p>{area.purpose}</p>
              <dl className="controls-ref">
                {area.controls.map((c) => (
                  <div key={c.name}>
                    <dt>
                      {c.name} <Pill>{c.kind}</Pill>
                    </dt>
                    <dd>
                      <p>{c.what}</p>
                      <p><b>Use it when</b> {c.when}</p>
                      <p><Pill kind="ok">recommended</Pill> {c.recommend}</p>
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </Card>

      <Card title="The phases" note={`${data.phases.length} modules · click one to open it`} collapsed>
        <div className="body-pad guide-prose">
          <p className="dim">
            Each module carries the phase's own contract, its live field schema, and
            the written guidance a schema cannot hold. The tables are fetched from
            the endpoints the forms use, so they cannot fall out of date.
          </p>
        </div>
        {data.phases.map((contract) => (
          <PhaseModule key={contract.id} contract={contract} />
        ))}
      </Card>
    </>
  );
}
