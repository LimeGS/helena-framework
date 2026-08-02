/**
 * One successful pass through the pipeline.
 *
 * Deliberately not the user guide. The guide answers "what does this control
 * do"; this answers "what do I press, in what order, to get one result". It
 * takes the shortest honest path -- one scroll, one surface, one detection --
 * and says what you should see after each step, because a phase that has gone
 * wrong usually still says it succeeded.
 *
 * Every wait here is a real measurement from these two hosts, not an estimate.
 * A number you can compare against is what tells you something has hung.
 */

export type Stop = {
  /** P0..P9, matching the phase rail. */
  id: string;
  /** What this step gets you, in one line. */
  goal: string;
  /** Roughly how long before it is done, on the reference hosts. */
  takes: string;
  /** The clicks, in order. */
  do: string[];
  /** How you know it actually worked. */
  done: string[];
  /** The one thing that most often goes wrong here. */
  watch?: string;
};

export const OPENING = {
  title: "Run the pipeline once, end to end",
  lede:
    "The shortest path that produces a real result: one scroll, one surface, one " +
    "ink detection. Nothing here is a toy — the same steps at a larger sample size " +
    "are what a campaign is. Expect about two hours of waiting spread across ten " +
    "steps, most of it in P1 and P4.",
  before: [
    "An account on this panel, and a mission — the Mission page makes one in a field and a button.",
    "At least one host reporting healthy on Configuration → Hosts, with a GPU if you want P4 to finish today.",
    "A scroll available in P0. If the list is empty the panel could not reach the scroll source, which is a network problem and not a mission problem.",
  ],
};

export const STOPS: Stop[] = [
  {
    id: "P0",
    goal: "Freeze which CT volume you are working from, and at what physical scale.",
    takes: "a minute",
    do: [
      "Open P0 from the phase rail on the left.",
      "Pick a scroll from the list. PHerc0139 and PHerc0826 are the ones this deployment has been exercised against.",
      "Type a sentence in “why the selection is changing…” — it becomes the record of why this campaign chose this scroll.",
      "Press “Record what P0 decided”.",
    ],
    done: [
      "The selection shows as frozen, with your sentence attached.",
      "Every later phase now defaults to this scroll instead of asking again.",
    ],
    watch:
      "Freezing is the point, not a formality. Phases downstream resolve “the scroll” " +
      "through this record, so a mission that skips P0 makes every later form ask you " +
      "for a sample id you have to remember correctly each time.",
  },
  {
    id: "P1",
    goal: "Grow one surface from a seed.",
    takes: "20–40 minutes for one surface",
    do: [
      "Open P1. The “New run” card is the launcher.",
      "Leave the seeder on its default. It is the one with a validated lane on this deployment.",
      "Set the number of attempts to 1 for this first pass. The default is higher because a campaign wants many; you want one that finishes.",
      "Press Queue.",
    ],
    done: [
      "A row appears under Runs with state PENDING, then CLAIMED within a poll or two.",
      "When it finishes, Segments gains a row with a geometry summary and an artifact URI.",
      "The Review column is empty — that is correct, nobody has judged it yet.",
    ],
    watch:
      "CLAIMED that never becomes done usually means the worker died rather than the " +
      "task being slow. Configuration → Hosts shows whether the host is still reporting; " +
      "a lease that expires puts the task back to PENDING on its own.",
  },
  {
    id: "P2",
    goal: "Decide whether the grown surface is a physically plausible lamina.",
    takes: "5–15 minutes",
    do: [
      "Open P2. It lists the surfaces P1 produced.",
      "Queue QC for the surface you just grew.",
      "When it finishes, open the surface and set the Review dropdown yourself.",
    ],
    done: [
      "The surface carries a verdict, and the run detail shows the QC receipt behind it.",
    ],
    watch:
      "The automatic verdict and your verdict are different fields on purpose. QC " +
      "measures geometry; only a person can say the sheet is the one they meant.",
  },
  {
    id: "P3",
    goal: "Unroll the certified surface into a flat sheet, keeping the coordinate map.",
    takes: "10–20 minutes",
    do: ["Open P3, pick the surface P2 approved, and Queue."],
    done: [
      "A flattened artefact appears against the surface, and P4 stops saying it has nothing to work on.",
    ],
    watch:
      "P3 runs in the same image as segmentation because it needs the whole VC3D " +
      "toolchain. If it fails immediately with a missing binary, the job was claimed " +
      "by a worker that cannot run it — which the queue now prevents, but old rows may " +
      "predate that.",
  },
  {
    id: "P4",
    goal: "Sample the CT along the surface normal to build the layer stack.",
    takes: "10–20 minutes",
    do: [
      "Open P4 and select the sheet P3 flattened.",
      "Leave the layer count on its default — it is what the detectors in P5 were trained against.",
      "Queue.",
    ],
    done: [
      "A layer stack is attached to the surface: the volume resampled along the normal, which is what the detector actually reads.",
    ],
    watch:
      "This is not ink detection, and the distinction matters when something looks " +
      "wrong later: P4 decides what the detector gets to see. A stack sampled at the " +
      "wrong scale produces a confident detection of nothing.",
  },
  {
    id: "P5",
    goal: "Run a detector over the layer stack to get a probability map.",
    takes: "20–40 minutes on a GPU",
    do: [
      "Open P5. Choose a lane — the lane is the model and its adapter, and the page marks which ones are validated here.",
      "Pick the checkpoint. If the list is empty, Configuration → Models downloads one by hash.",
      "Queue.",
    ],
    done: ["An ink probability map attached to the run."],
    watch:
      "A map is produced whether or not the model had anything to say. Whether it " +
      "carries a decision at all is P6's question, not this one — do not read the " +
      "picture and conclude before it.",
  },
  {
    id: "P6",
    goal: "Ask whether that map carries a decision at all.",
    takes: "a minute or two",
    do: ["Open P6 and run liveness against the map from P5."],
    done: [
      "A verdict of ALIVE, and the lane still reproducing on known ink.",
    ],
    watch:
      "DEGENERATE or EMPTY is the outcome worth having. The job ran, produced a file, " +
      "and the file is uniform — before this was checked that read as success and " +
      "cost real GPU time downstream.",
  },
  {
    id: "P7",
    goal: "Turn the probability map into a verdict about text-like structure.",
    takes: "5–10 minutes",
    do: [
      "Open P7 and select the run from P5.",
      "Choose the reference strip qualified for this scroll.",
      "Queue.",
    ],
    done: ["A screening receipt, and a verdict you can hold against the strip."],
    watch:
      "Screening against a strip that is not qualified for this scroll produces a " +
      "number that means nothing. Configuration lists which strip judges what.",
  },
  {
    id: "P8",
    goal: "Stitch segments into one continuous sheet.",
    takes: "15–30 minutes",
    do: [
      "Open P8, choose a lane, and give it the certified surfaces to join.",
      "Queue.",
    ],
    done: [
      "A merged artefact with lineage back to every parent surface, and seam evidence.",
    ],
    watch:
      "The merge fails closed — incompatible parents, a disconnected layout or too " +
      "little overlap all stop it rather than producing a plausible-looking join. It " +
      "never rewrites a parent.",
  },
  {
    id: "P9",
    goal: "Compose the assembled sheet into a readable page.",
    takes: "5–15 minutes",
    do: ["Open P9 and queue plate composition against the sheet P8 assembled."],
    done: [
      "Plates you can read, and a lineage that walks back to the volume P0 froze.",
    ],
    watch:
      "If lineage cannot reach the volume, something in the chain ran outside a " +
      "mission. Recoverable, but easier not to do.",
  },
];

export const AFTERWARDS = [
  "You now have one path through all ten phases, and every artefact on it is traceable back to the scroll P0 froze.",
  "A campaign is this at a larger sample size: more seeds in P1, more lanes in P4, and P6 telling you where to look next.",
  "The user guide is the other half of this — it documents every control on every page, including the ones this walkthrough left on their defaults.",
];
