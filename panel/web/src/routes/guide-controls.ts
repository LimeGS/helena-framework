/**
 * Every control the panel puts in front of you, and what to do with it.
 *
 * The other half of the guide is fetched live: queueable fields, seeders and
 * segmentation options come from the same endpoints the forms use, so a field
 * added to the queue documents itself. This file is for what a schema cannot
 * carry — buttons, page-level filters, toggles, and the judgement about which
 * setting is worth changing and which is noise.
 *
 * `page` matches a route file so a test can check nothing is left out. A page
 * with controls and no entry here fails the suite, which is the only way
 * "documents every control" stays true after somebody adds a button.
 */

export type Control = {
  /** What it says on screen, or what it plainly is. */
  name: string;
  /** button | field | dropdown | toggle | slider | filter */
  kind: string;
  /** What pressing or setting it actually does. */
  what: string;
  /** When it is the right thing to reach for. */
  when: string;
  /** What to leave it on, and why that is the sensible default. */
  recommend: string;
};

export type Area = {
  /** Route file, without extension. Checked against the source by the suite. */
  page: string;
  /** What the page is for, in one line. */
  title: string;
  purpose: string;
  controls: Control[];
};

export const AREAS: Area[] = [
  {
    page: "Mission",
    title: "Mission",
    purpose:
      "A mission is the folder everything else hangs from. Runs, selections and " +
      "findings belong to one, and two people working on different missions do not " +
      "see each other's work.",
    controls: [
      {
        name: "Mission name",
        kind: "field",
        what: "Creates a mission with this name and makes it the active one.",
        when: "Once per line of enquiry — a scroll, a question, a campaign.",
        recommend:
          "Name it after what you are trying to find out, not the date. You will be " +
          "picking it out of a list in three weeks.",
      },
      {
        name: "Mission picker",
        kind: "dropdown",
        what: "Switches which mission the rest of the panel is about.",
        when: "Whenever you come back to work already in progress.",
        recommend:
          "Check it before queueing anything. Work queued under the wrong mission is " +
          "not lost, but it is filed where nobody will look for it.",
      },
    ],
  },
  {
    page: "Phase",
    title: "Any phase page",
    purpose:
      "Every phase page has the same anatomy: a queue form at the top, the queue " +
      "itself, the profiles that phase can run, and the artefacts it has produced.",
    controls: [
      {
        name: "Scroll",
        kind: "field",
        what: "Which scroll this work is about.",
        when: "Filled from P0 automatically. Type one only when you are working outside a mission.",
        recommend:
          "Leave it as P0 set it. Typing it by hand is how two runs end up under " +
          "spellings that do not match.",
      },
      {
        name: "Lane",
        kind: "dropdown",
        what:
          "Which implementation runs the work. A lane is a model and its adapter, not " +
          "a priority.",
        when: "When you want a specific detector or geometry treatment.",
        recommend:
          "The lane marked validated on this deployment. The others are real but have " +
          "not been exercised here, so a failure in one is as likely to be the wiring " +
          "as the science.",
      },
      {
        name: "Queue",
        kind: "button",
        what: "Puts the job on the shared queue. It does not run it here and now.",
        when: "Once the form above it is complete.",
        recommend:
          "Press it once. A second press is a second job — the queue does not " +
          "de-duplicate, and both will run.",
      },
      {
        name: "The i button",
        kind: "toggle",
        what: "Expands what the phase is, what it consumes and how it fails.",
        when: "The first few times you use a phase.",
        recommend:
          "Read “how it fails” before your first run of any phase. It is written from " +
          "what actually went wrong here.",
      },
      {
        name: "Sub-tabs (Queue, Profiles, Artefacts)",
        kind: "toggle",
        what: "Switch between the work waiting, the configurations available, and what came out.",
        when: "Queue while you wait; Artefacts afterwards.",
        recommend:
          "Profiles is worth one read per phase: it shows which checkpoint each profile " +
          "pins, by hash.",
      },
    ],
  },
  {
    page: "Segmentation",
    title: "P1 — Segmentation",
    purpose:
      "The richest page in the panel, because growing surfaces has the most knobs. " +
      "Most of them you should not touch on a first campaign.",
    controls: [
      {
        name: "Seeder",
        kind: "dropdown",
        what: "How starting points are chosen — the strategy, not the parameters.",
        when: "Changing it changes what gets explored, not how well.",
        recommend:
          "The default. It is the seeder the validated lane was measured with.",
      },
      {
        name: "Attempts",
        kind: "field",
        what: "How many surfaces to try to grow from this launch.",
        when: "Higher for a campaign, lower when you are checking something works.",
        recommend:
          "1 for your first run, so you find out in half an hour rather than overnight. " +
          "The page default is tuned for a campaign, not for learning.",
      },
      {
        name: "Review verdict",
        kind: "dropdown",
        what: "Your judgement of a surface, recorded separately from the automatic QC.",
        when: "After looking at the surface yourself.",
        recommend:
          "Leave it empty rather than guessing. Empty means nobody judged; a wrong " +
          "verdict is worse than no verdict, because downstream trusts it.",
      },
      {
        name: "Slice sliders (position, threshold, window, slab)",
        kind: "slider",
        what: "Move through the volume and change how the preview is rendered.",
        when: "Looking at a surface to decide whether it is the sheet you meant.",
        recommend:
          "They change the picture only — nothing is recomputed and nothing is saved. " +
          "Move them freely.",
      },
      {
        name: "Overlay",
        kind: "toggle",
        what: "Draws the surface over the CT slices.",
        when: "Checking that the surface follows the sheet rather than cutting across it.",
        recommend: "On. It is the fastest way to see a surface that has wandered.",
      },
      {
        name: "Three orthogonal slices",
        kind: "button",
        what: "Opens three perpendicular cuts through the surface.",
        when: "When the overlay looks plausible and you want to be sure.",
        recommend: "Use it before approving anything you intend to spend GPU time on.",
      },
    ],
  },
  {
    page: "Coverage",
    title: "Coverage — a tab on P1",
    purpose:
      "A map of what has been looked at and what gave nothing. Not a phase of its " +
      "own: it is one of the tabs on the segmentation page, beside Runs and " +
      "Segments.",
    controls: [
      {
        name: "Re-ask the cells that gave no seed",
        kind: "button",
        what: "Queues fresh P1 work aimed at the empty parts of the grid.",
        when: "After a campaign has run long enough for gaps to be meaningful.",
        recommend:
          "Read “What this does not say” first. An empty cell can mean nothing is " +
          "there or that nobody has asked yet, and the map does not distinguish them.",
      },
    ],
  },
  {
    page: "Intake",
    title: "P0 — Intake",
    purpose: "Choosing the scroll, and freezing that choice so the rest of the pipeline can rely on it.",
    controls: [
      {
        name: "filter by scroll id…",
        kind: "filter",
        what: "Narrows the scroll list.",
        when: "The list is long.",
        recommend: "Type the number, not the prefix — “0139” finds it faster than “PHerc”.",
      },
      {
        name: "why the selection is changing…",
        kind: "field",
        what: "The reason, recorded with the selection.",
        when: "Every time you freeze or change a selection.",
        recommend:
          "Write the actual reason. This is what the lineage view shows somebody asking " +
          "in six months why this scroll.",
      },
      {
        name: "Record what P0 decided",
        kind: "button",
        what: "Freezes the selection and makes it the mission's answer to “which scroll”.",
        when: "Once you are sure.",
        recommend:
          "Do it before queueing anything else. Phases resolve the scroll through this.",
      },
    ],
  },
  {
    page: "Models",
    title: "Configuration — Models",
    purpose: "The checkpoints this deployment can run, and how to get more.",
    controls: [
      {
        name: "owner/name",
        kind: "field",
        what: "A Hugging Face repository to fetch a checkpoint from.",
        when: "Adding a model this deployment does not have.",
        recommend:
          "Prefer a repository that publishes safetensors. Only safetensors are " +
          "downloaded, and the reason is on the page: the other formats are pickles " +
          "that execute code when loaded, on a GPU host.",
      },
      {
        name: "main, a tag, or a commit",
        kind: "field",
        what: "Which revision to fetch.",
        when: "Pinning to something reproducible.",
        recommend:
          "A commit, not “main”. What is recorded is the resolved commit and the file " +
          "hash, so pinning here means the record and the intent agree.",
      },
      {
        name: "Download",
        kind: "button",
        what: "Fetches the checkpoint and registers it by hash.",
        when: "After resolving the repository.",
        recommend:
          "Check the resolved hash against what the profile expects. A profile pins a " +
          "checkpoint by hash and will refuse one that does not match — correctly.",
      },
    ],
  },
  {
    page: "Hosts",
    title: "Configuration — Hosts",
    purpose: "The machines that do the work, and whether they are still answering.",
    controls: [
      {
        name: "gpu-2 / 4x A100 / ink, segment",
        kind: "field",
        what: "Register a host: its name, what hardware it has, and which phases it may claim.",
        when: "Adding capacity.",
        recommend:
          "Name it exactly as the machine calls itself. The host report matches on that " +
          "name, and a mismatch shows as a host that registered and never reported.",
      },
      {
        name: "Enable / disable",
        kind: "toggle",
        what: "Whether the host may claim new work.",
        when: "Draining a machine before maintenance.",
        recommend:
          "Disable and let it finish rather than stopping it. Work in flight requeues " +
          "when the lease expires, but it is repeated work.",
      },
    ],
  },
  {
    page: "Users",
    title: "Configuration — Users",
    purpose:
      "Accounts and machine tokens. There are no roles: everyone who can sign in " +
      "can do everything, which the page says out loud.",
    controls: [
      {
        name: "Add an account",
        kind: "button",
        what: "Creates a person's account.",
        when: "Someone new needs the panel.",
        recommend: "One account per person. Shared accounts make the audit log useless.",
      },
      {
        name: "Mint token",
        kind: "button",
        what:
          "Creates a machine token for a worker on another host. Shown once and never " +
          "again.",
        when: "A worker that is not on the panel's own machine needs to publish surfaces.",
        recommend:
          "Name it after the worker, not the person. A machine token reaches the " +
          "artifact endpoints and nothing else, so a leaked one cannot queue work.",
      },
      {
        name: "Revoke",
        kind: "button",
        what: "Kills one machine token immediately.",
        when: "A host is decommissioned or a token may have leaked.",
        recommend:
          "Revoke per worker rather than rotating everything. That is the reason tokens " +
          "are named individually.",
      },
    ],
  },
  {
    page: "Config",
    title: "Configuration — Settings",
    purpose:
      "Every setting this deployment has, what it currently is, and where the value " +
      "came from.",
    controls: [
      {
        name: "filter by name, module or path…",
        kind: "filter",
        what: "Narrows the settings list.",
        when: "Always — the list is long.",
        recommend: "Search by what you are trying to change, not by what you think it is called.",
      },
      {
        name: "reset",
        kind: "button",
        what: "Puts one setting back to its default.",
        when: "You changed something and want the shipped behaviour back.",
        recommend:
          "Prefer this to typing the default back in. What the default is may have moved.",
      },
      {
        name: "restore",
        kind: "button",
        what: "Rolls the whole configuration back to an earlier version.",
        when: "A change broke something and you are not sure which one.",
        recommend:
          "Versions are kept for exactly this. Restoring is cheaper than bisecting by hand.",
      },
    ],
  },
  {
    page: "Modules",
    title: "Configuration — Modules",
    purpose: "Which parts of the platform this deployment has switched on.",
    controls: [
      {
        name: "Enable / disable a module",
        kind: "toggle",
        what: "Turns a phase or feature off across the panel.",
        when: "A deployment that does not do a kind of work should not offer it.",
        recommend:
          "Turning a module off hides its page; it does not stop work already queued " +
          "against it. Drain the queue first.",
      },
    ],
  },
  {
    page: "Audit",
    title: "Configuration — Audit",
    purpose: "Who did what, from where, and what the panel answered.",
    controls: [
      {
        name: "user / path filters",
        kind: "filter",
        what: "Narrows the trail to one account or one endpoint.",
        when: "Working out what changed and who changed it.",
        recommend:
          "Machine tokens appear as machine:name, so a worker's uploads are " +
          "distinguishable from a person's clicks.",
      },
      {
        name: "Refresh",
        kind: "button",
        what: "Re-reads the trail from the server without reloading the page.",
        when: "Watching something happen live.",
        recommend: "The trail is append-only; refreshing cannot lose anything.",
      },
    ],
  },
  {
    page: "Command",
    title: "The queue form",
    purpose:
      "The form at the top of every phase page. Its fields come from that phase's " +
      "schema, so they differ per phase and are listed further down, from the same " +
      "endpoint the form itself reads.",
    controls: [
      {
        name: "Required fields",
        kind: "field",
        what: "Marked by the schema; Queue stays disabled until they are filled.",
        when: "Always — the button not working is usually one of these, not a fault.",
        recommend:
          "If a field says it is filled by the deployment, leave it empty. Typing a " +
          "value there overrides something the host already knows.",
      },
      {
        name: "Exactly-one-of groups",
        kind: "field",
        what:
          "Some phases accept one of several inputs and refuse more than one. The " +
          "form marks the group.",
        when: "Choosing between, say, a surface and a run as the thing to work from.",
        recommend:
          "Fill one and clear the rest. Two filled is refused at queue time rather " +
          "than silently preferring one.",
      },
    ],
  },
  {
    page: "Screening",
    title: "P7 — Screening and adjudication",
    purpose:
      "Turns a probability map into a verdict about text-like structure, against a " +
      "reference strip.",
    controls: [
      {
        name: "Strip picker",
        kind: "dropdown",
        what: "Which reference strip to screen against.",
        when: "Every screening run.",
        recommend:
          "The strip qualified for the scroll you are on. Configuration lists which " +
          "strip is qualified to judge what; screening against an unqualified strip " +
          "produces a number that means nothing.",
      },
    ],
  },
  {
    page: "RunDetail",
    title: "A single run",
    purpose:
      "Everything one run produced: its receipt, its artefacts, the profile and " +
      "checkpoint behind it.",
    controls: [
      {
        name: "Artefact list",
        kind: "dropdown",
        what: "Switches between the files this run produced.",
        when: "Checking what actually came out rather than what was meant to.",
        recommend:
          "Open the receipt first. It names the profile and the checkpoint hash, which " +
          "is what makes the run reproducible.",
      },
    ],
  },
  {
    page: "Configuration",
    title: "Configuration",
    purpose:
      "Everything that is not a mission: settings, hosts, accounts, modules, models " +
      "and the audit log. It sits outside the mission gate because none of it " +
      "belongs to a mission.",
    controls: [
      {
        name: "Sub-tabs",
        kind: "toggle",
        what: "Switch between settings, hosts, users, modules, models and audit.",
        when: "Each is documented separately above.",
        recommend:
          "Hosts is the one to check first when work is queued and nothing is running.",
      },
    ],
  },
  {
    page: "Compare",
    title: "Compare",
    purpose: "Two runs side by side, with what differs between them named.",
    controls: [
      {
        name: "Run pickers",
        kind: "dropdown",
        what: "Choose the two runs to put beside each other.",
        when: "Deciding whether a change to a profile or checkpoint did anything.",
        recommend:
          "Compare runs that differ in one thing. Two runs differing in checkpoint and " +
          "geometry tell you nothing about either.",
      },
    ],
  },
  {
    page: "Lineage",
    title: "Lineage",
    purpose: "Walks backwards from a finding to the scroll it came from.",
    controls: [
      {
        name: "Artefact picker",
        kind: "dropdown",
        what: "Chooses what to trace.",
        when: "Asking where a result actually came from.",
        recommend:
          "If lineage stops short of the scroll, something in the chain was run outside " +
          "a mission. That is the signal to look for, not a display bug.",
      },
    ],
  },
];
