import { useQuery } from "@tanstack/react-query";
import { Card, Pill, queryGate } from "../components/Bits";

/**
 * How to put your own tool, model or idea into this pipeline.
 *
 * The audience is somebody who has a segmenter, a detector or an assembler of
 * their own and wants to run it here without rewriting the framework — or
 * without their work being quietly reinterpreted by it.
 */

type Contract = {
  id: string;
  name: string;
  one_line: string;
  consumes: string;
  produces: string;
  lives_in: string[];
  runner?: string | null;
  runnable_here?: boolean;
};

const Code = ({ children }: { children: string }) => (
  <pre>
    <code>{children}</code>
  </pre>
);

const ADAPTER = `# framework/stages/03-ink/fleet/ink_worker.py
#
# A runner is a subprocess with a receipt. The worker builds argv, runs it,
# checks what came out, and publishes it — it does not import your model.

RUNNERS = {
    "P5": {
        "my-detector@1.0.0": REPO / "framework/stages/03-ink/scripts/run_my_detector.py",
    },
}

# Your script owns its own flags. The worker passes what the profile declares
# and nothing else, so a flag you did not declare cannot reach it by accident.`;

const PROFILE = `{
  "profile_id": "my-detector@1.0.0",
  "stage": "03-ink",
  "adapter": "my-detector",
  "training_pixel_um": 7.91,
  "frames": 26,
  "checkpoint_sha256": "490a98f9…a972488",
  "known_limits": "trained on Scroll 1 at 7.91 um; nothing else is claimed"
}`;

const LANE = `# framework/stages/03-ink/fleet/job_store.py
#
# A lane is one way of doing a phase. Registering one is the whole integration:
# nothing in the worker, the panel or the command builder learns about it.

register_lane("P8", "mesh-relations", {
    "name": "relation-driven assembly",
    "runner": "framework/stages/05-reconstruction/scripts/assemble_from_relations.py",
    "required": ("scroll", "out_path"),
    "flags": {"scroll": "--scroll", "out_path": "--out", "work": "--work"},
    "defaults": {"work": lambda out: f"{out}/relations"},
    "note": "what it does differently, and what that buys",
})`;

const SCHEMA = `# framework/stages/03-ink/fleet/job_store.py

PHASE_PARAMETERS["P5"]["my_threshold"] = float
PARAMETER_HELP["my_threshold"] = (
    "Cut-off applied before the shapes are counted",
    "0.5",
)`;

export default function DeveloperReference() {
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
      <Card title="What this framework actually is" note="read before extending" collapsed>
        <div className="body-pad guide-prose">
          <p>
            Helena is a harness, not an algorithm. It contributes no segmenter, no
            detector and no assembler of its own: it orchestrates other people's
            tools and records what they did precisely enough that a result can be
            reproduced or refuted. Extending it means adding a tool, not changing
            what a phase means.
          </p>

          <h4>The three contracts everything obeys</h4>
          <p>
            <strong>A phase is a transformation with a declared input and output.</strong>{" "}
            The vocabulary lives in one committed file,{" "}
            <code>framework/contracts/pipeline_phases.json</code>, and this panel
            reads it rather than holding a copy. If your tool does not fit an
            existing phase's <em>takes</em> and <em>produces</em>, it is a new phase,
            not a variant of an old one.
          </p>
          <p>
            <strong>A run is a manifest plus a receipt plus hashed artefacts.</strong>{" "}
            The schemas are in <code>framework/contracts/schemas/</code>. A step that
            produces no receipt did not happen, and a receipt that cannot be
            validated is not evidence. This is what makes somebody else's result
            checkable without trusting you.
          </p>
          <p>
            <strong>The panel never executes anything.</strong> It writes a row into
            the queue. A worker on a machine with the right runtime claims it, runs
            a subprocess, and publishes the output to object storage with a digest.
            If you want your tool to run here, you are adding a runner to a worker
            image — not adding code to the panel.
          </p>

          <h4>What you should not do</h4>
          <p>
            Do not patch a vendored model to make it fit. The point of this harness
            is that a published result was produced by the author's own code, with
            their weights, at their scale. Where the upstream expects a different
            input shape, that is the adapter's problem: put the model's directory on{" "}
            <code>PYTHONPATH</code> and give it what it asks for, rather than editing
            it and reporting its number as theirs.
          </p>
        </div>
      </Card>

      <Card title="Adding a model or a tool to an existing phase" collapsed>
        <div className="body-pad guide-prose">
          <p>
            Four files, in this order. Nothing else in the framework needs to know
            you exist.
          </p>

          <h4>1 · A profile that declares what it is</h4>
          <p>
            Under <code>framework/profiles/&lt;stage&gt;/</code>. A profile is the
            scientific identity of a run: which weights, at what physical scale,
            trained on what, and what is <em>not</em> claimed. The checkpoint digest
            belongs here, and the worker verifies it before inference rather than
            trusting the path.
          </p>
          <Code>{PROFILE}</Code>

          <h4>2 · A runner script that owns its own flags</h4>
          <p>
            Under the stage's <code>scripts/</code>. It reads its input from a path,
            writes its output to a path, and exits non-zero on failure. It should not
            import anything from the panel, the queue or the control plane — a runner
            that can only run inside this framework is a runner nobody can check.
          </p>

          <h4>3 · A route from the profile to the runner</h4>
          <p>
            The worker chooses the runner from the profile id. This is the step that
            was wrong for a week: every lane ran the default runner's flags because
            the profile named its own adapter and nothing consulted it.
          </p>
          <Code>{ADAPTER}</Code>

          <h4>4 · The parameters, declared once</h4>
          <p>
            The queue's schema is the single source for what a phase accepts. The
            form in this panel draws itself from it, the user guide documents itself
            from it, and the worker validates against it. Adding a flag anywhere else
            produces a flag that exists and that nobody can set.
          </p>
          <Code>{SCHEMA}</Code>

          <p>
            Then rebuild the worker image that carries the runtime your tool needs
            and restart that worker. The panel needs no change and no restart.
          </p>
        </div>
      </Card>

      <Card title="Extending each phase" note={`${data.phases.length} extension points`} collapsed>
        <div className="body-pad guide-prose">
          <p className="dim">
            What a replacement has to accept and emit to drop into each phase. The
            input and output columns come from the committed contract, so they are
            the same ones the runtime enforces.
          </p>
        </div>
        <div className="scroller">
          <table>
            <thead>
              <tr>
                <th className="l">Phase</th>
                <th className="l">Your tool must take</th>
                <th className="l">…and produce</th>
                <th className="l">Queued from here</th>
              </tr>
            </thead>
            <tbody>
              {data.phases.map((p) => (
                <tr key={p.id}>
                  <td className="l scrollid">
                    {p.id} <span className="dim">{p.name}</span>
                  </td>
                  <td className="l">{p.consumes}</td>
                  <td className="l">{p.produces}</td>
                  <td className="l">
                    {p.runnable_here === false ? (
                      <Pill>no runner</Pill>
                    ) : (
                      <Pill kind="ok">yes</Pill>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Doing a phase a different way" note="one row, no edits" collapsed>
        <div className="body-pad guide-prose">
          <p>
            A phase is not one program. It is a question — unroll this sheet,
            assemble these segments — and a lane is one answer to it. Every phase
            carries a table of lanes, and adding an answer is adding a row.
          </p>
          <Code>{LANE}</Code>
          <p>
            That is the entire integration. The worker starts the program the
            lane names, the command line is built from the flags it declares,
            the queue form offers the choice as soon as a phase has more than
            one, and the user guide documents it — all of them reading the same
            table.
          </p>

          <h4>What a lane declares</h4>
          <ul>
            <li><code>runner</code> — the program, relative to the repository root.</li>
            <li><code>required</code> — parameters without which the job is refused,
              before a worker wastes time claiming it.</li>
            <li><code>flags</code> — parameter to command-line flag. A boolean
              becomes a bare flag when true and disappears when false.</li>
            <li><code>defaults</code> — a value computed from the job's own output
              directory, for the scratch paths a caller should not have to invent.</li>
            <li><code>build</code> — an escape hatch, for argv that is not a flat
              mapping: a subcommand, or a flag whose value is a file inside a
              directory the caller named.</li>
          </ul>

          <h4>Two rules the table enforces</h4>
          <p>
            A lane cannot shadow another silently — registering a name twice
            raises. And a job that names a lane nobody registered is refused
            rather than quietly given the default: a run that asked for one
            assembler and got another is a result nobody can interpret
            afterwards.
          </p>
          <p>
            Declare the lane's parameters in the queue schema as well, or the
            form will not offer the fields it needs.
          </p>
        </div>
      </Card>

      <Card title="Adding a phase that does not exist yet" collapsed>
        <div className="body-pad guide-prose">
          <p>
            Append it to <code>framework/contracts/pipeline_phases.json</code> with
            its id, what it consumes, what it produces, how it is run and how it
            fails. That file is the vocabulary: the sidebar, the phase pages and the
            user guide all derive from it, so a phase added there appears everywhere
            without a second edit.
          </p>
          <p>
            Then give it a queue schema and a runner as above, and add its id to the
            queueable set in the panel. The panel refuses to queue a phase with no
            registered runner rather than accepting a job nothing will claim.
          </p>
          <p>
            Keep <code>how_it_fails</code> honest. It is the field people read after
            a run that technically succeeded, and it is the difference between a
            framework that records results and one that launders them.
          </p>
        </div>
      </Card>

      <Card title="Running a worker of your own" collapsed>
        <div className="body-pad guide-prose">
          <p>
            A worker is a container that polls the control plane, claims jobs for the
            phases it declares, and publishes results. It needs the database URL, an
            artefact store, and a host id — nothing else, and no inbound port.
          </p>
          <p>
            Declare only the phases whose runtime your image actually carries. A
            worker that claims a phase it cannot execute takes the job out of the
            queue and fails it, and the phase sits queued behind it. The compose
            files under <code>containers/compose/</code> show a segmentation worker
            and an ink worker configured this way.
          </p>
          <p>
            The control plane listens on loopback. Remote workers reach it through an
            SSH forward rather than a published port — see{" "}
            <code>containers/systemd/helena-control-tunnel.service</code>.
          </p>

          <h4>What a worker owes the platform</h4>
          <p>
            Publish before you report success, and verify what you published. A job
            that reports success with its output only on local disk is a job whose
            result dies with the machine — which has happened here, and is why the
            worker checks the digest of what arrives rather than the exit code of
            what ran.
          </p>
        </div>
      </Card>

      <Card title="Naming and versioning" collapsed>
        <div className="body-pad guide-prose">
          <p>
            Six identity layers, each versioned differently, and confusing two of
            them is how a result ends up attributed to the wrong thing.
          </p>
          <ul>
            <li>
              <strong>Semantic Versioning</strong> for anything anybody depends on:
              the framework in <code>VERSION</code>, the images built from it, the
              contracts and the profiles. A breaking change to what a field means is
              a major bump, not a footnote.
            </li>
            <li>
              <strong>Contract</strong> — the schema for a manifest or a receipt.
              Versioned in its filename, and never edited in place: a receipt
              written last year must still validate.
            </li>
            <li>
              <strong>Scientific profile</strong> — <code>name@major.minor.patch</code>,
              e.g. <code>timesformer-gp-scroll1@1.0.0</code>. It names weights, scale
              and declared limits. Different weights are a different profile, never
              the same profile with a different file on disk.
            </li>
            <li>
              <strong>Experiment</strong> — <code>EXP-YYYYMMDD-STAGE-slug-rNN</code>,
              with its plan under <code>PLN-EXP-…</code>. The plan is written before
              the run, so the result cannot choose its own interpretation afterwards.
            </li>
            <li>
              <strong>Run</strong> — one execution of one phase, with a directory of
              its own that no other run may write into.
            </li>
            <li>
              <strong>Receipt</strong> — what that run actually did: commit, image
              digest, arguments, inputs and outputs by hash. Mandatory even on
              failure.
            </li>
          </ul>
          <p>
            A commit hash is an identity, not a version: it says which bytes these
            are and nothing about whether a deployment may take them.
          </p>
        </div>
      </Card>

      <Card title="Operating it" collapsed>
        <div className="body-pad guide-prose">
          <h4>What the platform records about itself</h4>
          <p>
            Every request that could change something is appended to an audit trail
            under the panel's state directory — timestamp, id, user, action, status,
            duration and client address, one JSON object per line, one file per
            month. Read it under <strong>Configuration → Audit log</strong>. Reads
            are not recorded; refusals are, including failed sign-ins. Request bodies
            are never captured, because two of those routes carry a password and a
            set of storage credentials.
          </p>
          <p>
            Worker activity is not in that trail. It is in the queue's own event
            tables, which record state transitions per job. The trail answers who
            asked for something, not what the fleet did about it.
          </p>

          <h4>Logs</h4>
          <p>
            Every long-lived container caps its log at 20 MB × 5 in its compose file.
            Docker's default driver has no limit at all, and a worker in a retry loop
            will write until the volume holding the database runs out — which has
            happened here.
          </p>

          <h4>Errors</h4>
          <p>
            The rule the code follows, enforced by a test: a failure may be swallowed
            in silence only where a named exception type has a fallback that is the
            correct answer. <code>except Exception: pass</code> is never that, because
            it also catches the failure nobody predicted.
          </p>

          <h4>Tests</h4>
          <p>
            <code>python3 -m pytest tests/</code> runs everything that does not need a
            fleet. Tests that recompute a declared number from a hash-bound receipt
            skip unless <code>HELENA_EVIDENCE_ROOT</code> points at an unpacked
            evidence release. The end-to-end suite under <code>tests/e2e/</code> needs
            a running panel and, for its heavy half, a GPU — it is skipped by default,
            because a suite that quietly reserves a card is a suite people stop
            running.
          </p>

          <h4>Backups</h4>
          <p>
            The control plane is dumped, verified by reading it back, and uploaded on
            a schedule, together with the panel's state directory. A dump nobody has
            opened is a hope rather than a backup, so each round checks that the dump
            parses and lists its objects before the upload.
          </p>
        </div>
      </Card>

      <Card title="The API, if you would rather not use this panel" collapsed>
        <div className="body-pad guide-prose">
          <p>
            Everything this panel does is HTTP, and the panel is one client. Sign in
            at <code>POST /api/session</code>, then:{" "}
            <code>GET /api/phases</code> for the vocabulary,{" "}
            <code>GET /api/phases/&lt;id&gt;/parameters</code> for what a phase
            accepts, <code>POST /api/jobs</code> to queue one, and{" "}
            <code>GET /api/jobs</code> to watch it. The harness under{" "}
            <code>scripts/harness/</code> is a worked example in the standard library
            alone.
          </p>
          <p>
            The API reference tab is the HTTP surface — every route with its
            method, parameters and response codes, generated from the panel's
            own OpenAPI document.
          </p>
        </div>
      </Card>
    </>
  );
}
