import { useState } from "react";
import { Artifacts } from "../components/Artifacts";

/**
 * The audit trail, where audit trails belong.
 *
 * Nothing here is part of doing the work: inputs are chosen when a run is
 * started, and outputs are registered by the run that produced them. This is
 * for the two questions asked afterwards -- what did that read, and when did
 * the answer change -- plus the one-off import of runs that predate the
 * register.
 */
const PHASES = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"];

export default function Lineage() {
  const [phase, setPhase] = useState("P1");
  return (
    <>
      <div className="body-pad">
        <div className="controls">
          <label className="inlinecheck">
            phase
            <select value={phase} onChange={(e) => setPhase(e.target.value)}>
              {PHASES.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        </div>
      </div>
      <Artifacts phase={phase} sample={null} />
    </>
  );
}
