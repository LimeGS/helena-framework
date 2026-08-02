/**
 * The written half of the user guide.
 *
 * The machine-readable half — every queueable field, every segmentation option,
 * every seeder — is fetched from the API the forms themselves use, so it cannot
 * drift. What lives here is what a schema cannot carry: the order you make
 * decisions in, which control matters and which is noise, and what a phase looks
 * like when it has gone wrong while reporting success.
 */

export type Step = { do: string; why: string };

export type Figure = { src: string; alt: string; caption: string };

export type PhaseGuide = {
  /** One paragraph: what you are actually doing in this phase. */
  purpose: string;
  /** What must exist before you open the form. */
  before: string[];
  /** The decisions, in the order the form asks them. */
  steps: Step[];
  /** Controls that are not fields: buttons, toggles, modes on the page. */
  controls?: { name: string; what: string }[];
  /** A picture of the part of the page this module is about. */
  figure?: Figure;
  /** How to tell a good result from an expensive one. */
  reading: string[];
  /** Ways it succeeds and produces nothing usable. */
  traps: string[];
};

export const GUIDE: Record<string, PhaseGuide> = {
  P0: {
    purpose:
      "You are choosing which scan everything downstream argues about, and pinning " +
      "the number that makes those arguments comparable: how many microns one voxel " +
      "is. Nothing here computes; it freezes a choice so that a result produced in " +
      "six months can still say what it was produced from.",
    before: [
      "Nothing. This is the first step.",
      "Know which scroll and which scan you mean — a scroll often has several, at " +
        "different energies and resolutions, and they are not interchangeable.",
    ],
    steps: [
      { do: "Pick the scroll from the catalog on the Scrolls tab.",
        why: "The catalog is the frozen list of what this deployment can reach. A " +
             "scroll not in it is not a scroll this platform can start from." },
      { do: "Check the declared scale — µm per voxel — against the scan you think " +
            "you chose.",
        why: "Every later phase resamples in microns. A detector trained at 7.91 µm " +
             "fed a stack at 9.362 µm sees the wrong physical thickness, and nothing " +
             "downstream will notice: it will simply produce a worse map." },
      { do: "Add the scroll to your mission from the Mission page.",
        why: "Scoping. Coverage, backlogs and job counts are all computed against the " +
             "mission's scrolls, so a scroll outside it is work that will not be " +
             "counted anywhere." },
    ],
    controls: [
      { name: "Refresh catalog",
        what: "Re-reads the public bucket rather than the cached inventory. The cache " +
              "lasts a day; use this after a new scan is published." },
      { name: "Add / remove scrolls (Mission)",
        what: "Amends which scrolls the mission covers. Removing is refused for a " +
              "scroll that already has runs or queued jobs — that would orphan work " +
              "rather than tidy it." },
    ],
    reading: [
      "A scroll with a declared pixel size and an energy is ready. One without a " +
        "scale is not usable and later phases will refuse it.",
      "Two spellings of the same scroll — PHerc0826 and PHerc826 — are two scrolls to " +
        "the control plane. Use the catalog spelling everywhere.",
    ],
    traps: [
      "Choosing a scan by name and assuming the resolution. Read the number.",
      "Working outside a mission: pages then count nothing and look empty.",
    ],
  },

  P1: {
    purpose:
      "You are looking for sheet surfaces inside the volume. The work is divided into " +
      "grid cells; for each cell a seeder picks one point to start from, and " +
      "VC3D/m7 grows a surface outward from that point until it stops. Two decisions " +
      "matter and they are separate: which cells are worth attempting, and — inside " +
      "a chosen cell — which point to start at.",
    before: [
      "A scroll frozen in P0, with its scale declared.",
      "For the fleet path: a worker running with the segmentation image and a card " +
        "it can use.",
      "For supplied points: coordinates in CT-L0 voxels, from an official " +
        "segmentation, a previous run, or read off a slice by hand.",
    ],
    steps: [
      { do: "Choose the seed source: Found in a cell, or Points I supply.",
        why: "They answer different questions. The first sweeps the volume and asks " +
             "the prediction where to start; the second grows exactly where you say, " +
             "which is how you reproduce somebody else's surface or test a place the " +
             "prediction dismissed." },
      { do: "Choose the seeder. This is the single most consequential choice on the " +
            "page.",
        why: "Given the same cell, two seeders pick different points and grow " +
             "different sheets — both correct. Which one ran is part of what the " +
             "surface means, and it is stamped on every task." },
      { do: "Open Options and set where to look: grid step, query radius, clearance, " +
            "cell ranking.",
        why: "These decide which cells are queued at all, and they are fixed at " +
             "queue time. Everything else is decided per attempt by the seeder." },
      { do: "Decide whether the CT gate is on. Leave it on.",
        why: "It rejects a seed the raw scan has no material at. Growing from a point " +
             "with nothing there costs hours and cannot succeed. The first 142 tasks " +
             "this fleet queued ran without it, trusting the prediction alone." },
      { do: "Set the grid version and the policy version, then queue.",
        why: "A task is identified by volume, grid version, cell and policy version. " +
             "Re-queueing the same cells under the same policy inserts nothing and " +
             "reports success. To genuinely ask again, change the policy version." },
    ],
    figure: {
      src: "seeders",
      alt: "The New run form on the Segmentation page, showing the seed source " +
           "switch and the five seeders",
      caption:
        "Segmentation → New run. Top row: which scroll, and which P0 snapshot it " +
        "reads. Then how many cells to attempt. The bordered block is the one " +
        "choice that matters most — where the seed comes from, and which seeder " +
        "picks it. Options (18) opens the parameters below. Note what the form " +
        "says at the bottom: you are not choosing coordinates.",
    },
    controls: [
      { name: "Found in a cell / Points I supply",
        what: "The seed source. Supplied points skip the prediction entirely, so the " +
              "CT-material gate is the only screen left between a typo and hours of " +
              "growing." },
      { name: "Options (n)",
        what: "Reveals the eighteen parameters below, grouped by when they are " +
              "decided. Collapsed by default because most runs touch four of them." },
      { name: "Seed probe (Off / Shadow / Select)",
        what: "Runs frozen 10-20 generation micro-grows on the first one to three " +
              "candidates before committing to a full grow, and asks only which of " +
              "them produces geometry that can be measured. It claims nothing about " +
              "the correct lamina, ink or text. Shadow records the probes and then " +
              "runs the planner unchanged, which is the phase to use; Select lets one " +
              "probe steer the grow and stays closed until the benchmark and " +
              "source-lock gates pass. An inconclusive comparison abstains rather " +
              "than turning a score into a winner." },
      { name: "CT (Segments tab)",
        what: "Three orthogonal slices through that surface's own bounding box, with " +
              "the sheet drawn on top from its TIFXYZ. This is how you tell a surface " +
              "that follows one lamina from one that crossed to the next: the dots " +
              "should run along the layering, not across it. Slices are bounded by " +
              "the surface, not the volume, because a full slice of one of these " +
              "scans is hundreds of megabytes." },
      { name: "open (Segments tab)",
        what: "A bundle naming the exact volume, prediction, coordinate frame, " +
              "surface and hashes, plus the vc_grow_seg_from_seed command that opens " +
              "them. Pointers, not copies: you already have the volume or you do not, " +
              "and a bundle carrying one would be a bundle nobody can download." },
      { name: "Review (Segments tab)",
        what: "Approved, Defective, Reviewed or Inspect, attributed to you. This is " +
              "your opinion and not a scientific verdict: geometry certification and " +
              "CT support live in their own columns, are written by the fleet, and " +
              "are what P3 and P4 ask before consuming a surface. Nothing you choose " +
              "here can admit a surface downstream." },
      { name: "Replan (Coverage tab)",
        what: "Queues the cells a previous sweep never attempted, under a new policy " +
              "version. It is the honest way to widen a campaign: the old tasks keep " +
              "their identity and the new ones are attributable to the new rules." },
      { name: "Republish (Surfaces tab)",
        what: "Copies a surface that was written to a worker's local disk into object " +
              "storage, and re-points its record. It verifies the digest against the " +
              "one the surface was recorded with rather than recomputing it — a " +
              "republish that quietly changed the digest would be a different " +
              "artefact wearing the original's verdicts." },
    ],
    reading: [
      "A cell that yields no seed is a measurement, not a failure. Coverage counts it " +
        "as attempted, and that is what stops the same empty region being swept " +
        "forever.",
      "Surface area is gross area, not deduplicated lamina coverage. Two surfaces " +
        "over the same sheet count twice.",
      "POLICY_REJECTED means the seeder proposed something the frozen policy forbids. " +
        "That is the contract working, not a bug.",
    ],
    traps: [
      "Re-running with the same policy version and reading the success as a new sweep. " +
        "Nothing was queued.",
      "Turning the CT gate off to get more seeds. You get more seeds and no more " +
        "surfaces, at hours per attempt.",
      "Supplied coordinates in the wrong resolution level. They are CT-L0 voxels — " +
        "full resolution — not the level you happened to be viewing.",
    ],
  },

  P2: {
    purpose:
      "You are asking whether a grown surface is a physically plausible single lamina, " +
      "or two sheets welded together by a segmenter that lost the thread. This is a " +
      "gate, not a measurement: it produces a verdict per surface.",
    before: [
      "One or more surfaces from P1 that carry no verdict yet.",
      "Nothing else. An imported surface qualifies too, and usually needs this more " +
        "than a grown one: growing certifies at finalisation, importing skips that " +
        "entirely, so a catalogue surface arrives unmeasured.",
    ],
    steps: [
      { do: "Set the batch size and queue.",
        why: "It runs over surfaces with no verdict. A bounded batch keeps a run " +
             "reviewable; an unbounded one produces a wall of verdicts nobody reads." },
      { do: "Use Dry run first on a new scroll.",
        why: "It lists what would be certified and stops, so you see the population " +
             "before you spend the fleet on it." },
    ],
    controls: [
      { name: "Dry run",
        what: "Lists what would be done and changes nothing." },
      { name: "Scroll",
        what: "Restricts the batch to one scroll. Empty means every scroll in the " +
              "mission." },
      { name: "certify (Runs tab, maintenance)",
        what: "The same verdict, run as a one-shot from the Runs page. Safe to " +
              "repeat: it only touches surfaces that have none. This is what " +
              "releases an imported surface from WAITING_GEOMETRY." },
    ],
    reading: [
      "A surface is in one of three positions and the middle one is easy to miss. " +
        "Certified means the gate looked and found no hard defect. Rejected -- " +
        "bridge, lamina switch, distortion, coverage -- means it looked and found " +
        "one, and those are named apart because they are different problems with " +
        "different causes. Unmeasured means nobody looked.",
      "Unmeasured is not a pass. Its QC job is created WAITING_GEOMETRY rather than " +
        "PENDING, and the model can only claim PENDING, so the surface waits instead " +
        "of quietly proceeding. A verdict promotes it the moment one arrives, which " +
        "is why waiting is not the same as stranded.",
      "There are two independent QC axes and both must pass before anything " +
        "downstream may use a surface: geometry (is it a clean sheet) and physical " +
        "support (does the CT actually contain material along it).",
      "GEOMETRY_CERTIFIED alone is not admissible. A surface can be geometrically " +
        "immaculate and still be a bridge between two laminae.",
    ],
    traps: [
      "Reading certification as 'this is correct'. It means no hard defect was found " +
        "at the sampling the gate uses.",
      "Assuming an imported surface was checked because it is in the catalogue. It " +
        "arrived with a bounding box and a hash and no verdict, and it waits until " +
        "this phase gives it one.",
      "Treating a human Approved from the Segments tab as a substitute. It is a " +
        "different column, written by a person, and no gate reads it.",
    ],
  },

  P3: {
    purpose:
      "You are unrolling a certified curved surface into a flat sheet, keeping the " +
      "mapping back to the volume so that anything found on the flat sheet can be " +
      "located in the scroll again.",
    before: [
      "A surface P2 has certified on both axes.",
      "A flattening store configured on the deployment — the phase refuses without " +
        "one.",
    ],
    steps: [
      { do: "Pick the flattening profile.",
        why: "It fixes the parameters of the unrolling. Two profiles produce two " +
             "different sheets from one surface, and every downstream render inherits " +
             "which one ran." },
      { do: "Queue a bounded batch.",
        why: "Flattening is the slowest CPU stage in the pipeline." },
      { do: "Check the backlog arithmetic before adding to it.",
        why: "The Flattening page reports how many certified surfaces have no sheet " +
             "yet. If that number does not move after a batch, the batch produced " +
             "nothing and the queue is where to look, not the store." },
    ],
    controls: [
      { name: "Flattening profile",
        what: "The frozen parameters of the unrolling. It is recorded on the output, " +
              "so two sheets from one surface can always be told apart by which " +
              "profile made them -- which matters because they are not " +
              "interchangeable downstream." },
      { name: "Allow unvalidated",
        what: "Includes surfaces the CT never confirmed. The default takes only " +
              "surfaces the scan supports; this exists to compare against what the " +
              "older gate admitted, not as a way to get more sheets." },
    ],
    reading: [
      "A sheet published only to a worker's disk is lost with the worker. The phase " +
        "refuses rather than producing something nobody else can read.",
      "Distortion is inherent: a curved sheet cannot flatten without stretching " +
        "somewhere. The quality map says where.",
      "Read the quality map before the sheet. A region with high distortion can " +
        "produce letter-shaped artefacts that survive every downstream screen, " +
        "because the screens measure shape and the distortion made the shape.",
      "The mapping back to the volume is the point of the phase, not a by-product. " +
        "A flat sheet you cannot locate in the scroll again is an image, not " +
        "evidence.",
    ],
    traps: [
      "Flattening a surface with more than one boundary loop. The UV initialiser does " +
        "not accept it and the job fails late.",
      "Turning on Allow unvalidated to get more sheets. It admits surfaces the CT " +
        "never supported, so what comes out the far end is a flat picture of " +
        "something the scan does not agree is there.",
      "Comparing two sheets that came from different profiles. They will differ, and " +
        "the difference is the profile rather than the scroll.",
    ],
  },

  P4: {
    purpose:
      "You are sampling the CT along the surface normal to build the layer stack the " +
      "detector eats: N images, each one a slice at a fixed depth through the sheet. " +
      "This is the phase that decides depth, and depth is the thing that most often " +
      "goes wrong while the job exits zero.",
    before: [
      "A flattened sheet from P3, or a surface path.",
      "The volume, reachable from the worker — locally or as a remote OME-Zarr it can " +
        "cache.",
      "A decision about depth: how many slices, and how far apart.",
    ],
    steps: [
      { do: "Choose the lane. The default reads a tifxyz directly; the chunk-gather " +
            "lane is Scroll 3 only and takes a PPM.",
        why: "Which renderer ran is part of what a layer stack means." },
      { do: "Give exactly one of a flattened surface id or a surface path.",
        why: "The first makes the worker fetch the sheet P3 published; the second " +
             "points at a tifxyz on disk. Both is ambiguous and the form refuses it." },
      { do: "Set num_slices and slice_step to match the detector you intend to use.",
        why: "A detector trained on 26 frames at 7.91 µm needs a slab of that physical " +
             "thickness. Handing it the right count at the wrong spacing is a slab of " +
             "the wrong thickness, and it will still produce a map." },
      { do: "If the map that follows is flat, come back and reverse the normals.",
        why: "A back-to-front slab is a correct render of the wrong side of the sheet. " +
             "It looks fine and detects nothing." },
    ],
    figure: {
      src: "queue-form",
      alt: "The P4 queue form, with each field's explanation printed under it",
      caption:
        "Every queueable phase has this shape. The form is generated from the " +
        "queue's parameter schema, so it always matches what the runner accepts, " +
        "and each field carries its explanation underneath. A field marked * is " +
        "required. Fields tied to one lane appear only when that lane is chosen.",
    },
    controls: [
      { name: "Stripe",
        what: "Renders one horizontal band instead of the whole sheet. A cheap look " +
              "before committing an hour." },
      { name: "Byte budget / chunk cache",
        what: "Caps what the job may write and how much disk the staged CT chunks may " +
              "use. Relevant when the volume is remote." },
    ],
    reading: [
      "Check two things on the result before believing it: the slice count is the one " +
        "you asked for, and the middle slice is not a constant.",
      "The stack is published with a digest. A stack that stayed on the worker is one " +
        "no other machine can read.",
    ],
    traps: [
      "Trusting the exit code. This phase's most common failure is a complete, " +
        "well-formed stack of the wrong depth.",
      "Rendering at full resolution to 'see more'. Group index 0 is full resolution " +
        "and costs accordingly.",
    ],
  },

  P5: {
    purpose:
      "You are running a detector over the layer stack to get a probability map: for " +
      "each pixel of the flattened sheet, how strongly this model responds. A map is " +
      "not ink and not text — it is one model's response, and the phases after this " +
      "exist to decide whether it means anything.",
    before: [
      "A published layer stack from P4.",
      "A lane, and a checkpoint the worker can read. The digest is verified before " +
        "inference rather than the path being trusted.",
      "The stack's physical scale, so the depth window can be computed.",
    ],
    steps: [
      { do: "Pick the lane, which picks the model and its runner.",
        why: "Each lane has its own flags and its own training scale. The Lanes tab " +
             "lists which are routable on this deployment and why the others are not." },
      { do: "Name the render by its job id rather than a directory.",
        why: "It is what makes the chain something the control plane can express, and " +
             "what lets the worker fetch the stack itself." },
      { do: "Leave the depth centre empty unless you have a reason.",
        why: "The worker fits the window to the stack it was given, and refuses a " +
             "stack too shallow for the model rather than handing it padding." },
      { do: "Use several depth centres and two tiling offsets when you want a claim " +
            "to survive scrutiny.",
        why: "The maps are combined by minimum, so a response present at only one " +
             "depth or only one grid alignment disappears. That is the point." },
      { do: "Set device: cpu if no card is free.",
        why: "It works and is far slower. Nothing in the worker or the queue requires " +
             "a GPU for this phase." },
    ],
    controls: [
      { name: "Batch size",
        what: "Tiles per forward pass. Lower it if the card runs out of memory; it " +
              "changes speed, not the result." },
      { name: "Minimum valid ratio",
        what: "Skips a tile with less real data than this, so padding is never scored " +
              "as signal." },
      { name: "On a degenerate map",
        what: "What to do when the output carries no decision at all: fail the job, or " +
              "record it and continue." },
    ],
    reading: [
      "A map that is one value everywhere is what a wrong depth window or a " +
        "back-to-front slab produces, and it exits zero.",
      "A high global response is not good news. It is what an extended texture " +
        "confounder looks like; what discriminates is bounded shapes that repeat.",
    ],
    traps: [
      "Comparing two maps produced at different depths or scales as if they disagreed " +
        "about ink.",
      "Reading the brightest map as the best one.",
    ],
  },

  P6: {
    purpose:
      "A check, not a stage you queue. After P5 produces a map, this asks whether the " +
      "map carries a decision at all — whether it has structure, or is a constant, or " +
      "is noise with no spatial organisation.",
    before: [
      "Nothing to run: it happens inside the P5 runner.",
      "A P5 result to read it from. Until one exists the phase shows no-run and " +
        "offers no button, which is correct rather than broken.",
    ],
    steps: [
      { do: "Read the liveness verdict on the P5 result.",
        why: "ALIVE means the map has something to argue about. It does not mean ink." },
      { do: "Check it before comparing two models or two renders.",
        why: "A dead map compared against another dead map produces a difference, " +
             "and the difference is noise. Liveness is what tells you the comparison " +
             "was between two things that said something." },
    ],
    controls: [
      { name: "None: this phase has no form",
        what: "It runs inside P5 and the rail shows it as no-run for that reason. A " +
              "phase that cannot be queued must not offer a button, so there is " +
              "nothing here to press." },
    ],
    reading: [
      "A dead map is a statement about the map, not about the scroll. It says this " +
        "render and this model produced nothing to decide on.",
      "The three ways a map fails this are different: a constant says the model " +
        "returned one value everywhere, noise says it returned structure with no " +
        "spatial organisation, and an empty response says it returned nothing at " +
        "all. Only the first two look like a result at a glance.",
    ],
    traps: [
      "Reporting a dead map as evidence of no ink.",
      "Reading ALIVE as a quality score. It is a floor, not a grade: it says the " +
        "map is worth screening, and P7 is what decides whether it is worth reading.",
    ],
  },

  P7: {
    purpose:
      "You are turning a probability map into a verdict about text-like structure: are " +
      "there bounded shapes, do they repeat across replicas, do they organise into " +
      "rows. This is where a map becomes a priority for human review — or does not.",
    before: [
      "A screened probability map from P5, ideally with several replicas.",
      "A bounding box, if you are asking about one region. This one does not come " +
        "from a phase: a person decides where to look, which is why the rail can " +
        "show P7 blocked while everything upstream is green.",
    ],
    steps: [
      { do: "Set the pixel scale so shapes can be argued in microns.",
        why: "A shape the right size for a letter at 2.4 µm is a blob at 9.4 µm. " +
             "Without the scale the screen counts pixels and means nothing." },
      { do: "Restrict to a bounding box when you are testing a specific region.",
        why: "A whole-sheet screen on a large sheet buries one interesting window." },
      { do: "Run it against every replica you have, not the best one.",
        why: "Repeatability across replicas is the whole signal here. One replica " +
             "cannot distinguish a shape that is there from a shape one run of one " +
             "model produced once." },
    ],
    controls: [
      { name: "Bounding box",
        what: "The window to screen. Blank means the whole sheet, which is the right " +
              "default for a first pass and the wrong one for a follow-up." },
      { name: "Pixel scale",
        what: "Microns per pixel of the render being screened. Everything the gate " +
              "measures about shape size depends on it, and a wrong value does not " +
              "fail: it silently rescales what counts as a letter." },
    ],
    reading: [
      "A positive gate is a priority for CT review, not a finding. It says this " +
        "response is localised and repeated, which is the minimum for a human to " +
        "spend time on it.",
      "Shapes must survive the minimum across depths and offsets. A diffuse response " +
        "evaporates there; a localised one does not.",
    ],
    traps: [
      "Presenting a passing gate as letters. Nothing before human CT adjudication is " +
        "a claim about text.",
      "Reading blocked as broken. P7 waits for a probability map and a bounding box, " +
        "and it names both. A phase that is waiting for an input says so; a phase " +
        "that is broken does not.",
    ],
  },

  P8: {
    purpose:
      "You are stitching adjudicated segments into one continuous sheet: deciding who " +
      "neighbours whom, and in what order the windings run. An assembly asserts a " +
      "physical relation between sheets that were segmented independently.",
    before: [
      "Two or more adjudicated segments believed to be neighbours.",
      "Their published meshes. The rail can show P8 ready while nothing has run, " +
        "because ready means the inputs exist and not that anybody has asked.",
    ],
    steps: [
      { do: "Subsample first on a large assembly.",
        why: "A coarse assembly costs a fraction and shows whether the relations are " +
             "plausible before you pay for the full one." },
      { do: "Assemble neighbours you have a reason to believe are neighbours.",
        why: "The phase relates what you give it. Handing it every segment on a " +
             "scroll and reading the result as discovery is asking the assembler to " +
             "make a claim you have not made." },
      { do: "Look at the seams before the whole.",
        why: "A wrong assembly is continuous everywhere except at the joins, and the " +
             "joins are small. The overall picture is the least informative view of " +
             "it." },
    ],
    controls: [
      { name: "Work directory",
        what: "Scratch for intermediates. It is not the published output and is not " +
              "kept." },
      { name: "Subsample",
        what: "Assembles at reduced resolution. Use it for the first pass on anything " +
              "large: the relations it finds are the same ones, and they cost a " +
              "fraction to look at." },
    ],
    reading: [
      "An assembly that looks continuous because two segments were forced into " +
        "alignment is worse than no assembly: it invents a page.",
      "Winding order is the part with no automatic check. Nothing in the platform " +
        "yet computes which turn of the spiral a surface belongs to, so an assembly " +
        "that orders two sheets wrongly is not caught here or anywhere downstream.",
    ],
    traps: [
      "Assembling across a lamina jump. The seam will look like a fold.",
      "Reading area as coverage. Two assembled segments that overlap contribute their " +
        "overlap twice, and identity across patches is unresolved, so the number is " +
        "an upper bound rather than how much sheet you have.",
    ],
  },

  P9: {
    purpose:
      "You are composing the assembled sheet into a page image, and attempting a " +
      "reading of it. This is the last phase and the weakest link: everything before " +
      "it can be checked mechanically, and this cannot.",
    before: [
      "An assembled sheet from P8, or published ink maps to compose.",
      "The ink maps in particular are what this phase waits for, and no phase here " +
        "produces them: they come from outside. That is why the rail can show P9 " +
        "blocked on a scroll where everything upstream succeeded.",
    ],
    steps: [
      { do: "Compose the plate first and look at it before reading anything.",
        why: "If the plate is wrong the reading is worse than useless, because it is " +
             "persuasive." },
      { do: "Keep the plate and the reading as separate artefacts.",
        why: "The plate can be checked by anybody with the inputs. The reading " +
             "cannot, and folding them into one document makes the unverifiable half " +
             "inherit the credibility of the verifiable half." },
    ],
    controls: [
      { name: "None yet on this deployment",
        what: "P9 is blocked until published ink maps exist for the scroll, so the " +
              "page shows what it is waiting for rather than a form." },
    ],
    reading: [
      "What comes out is an image and a proposed reading. Neither converts into a " +
        "claim about text without a papyrologist.",
    ],
    traps: [
      "Treating a rendered page as a transcription.",
    ],
  },
};
