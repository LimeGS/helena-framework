import { Card, Pill } from "../components/Bits";
import { AFTERWARDS, OPENING, STOPS } from "./tutorial-content";

/**
 * One successful pass through the pipeline, start to finish.
 *
 * Split out of the user guide because they answer different questions. The
 * guide is a reference — every control on every page, what it is for, what to
 * leave it on. This is a path: press these, in this order, and you get a
 * result. Somebody arriving at this panel for the first time needs the path;
 * somebody who already ran something needs the reference, and a single page
 * that tried to be both was mostly the path with the reference missing.
 */
export default function Tutorial() {
  return (
    <div className="prose">
      <Card title={OPENING.title} note="about two hours, mostly waiting">
        <div className="body-pad">
          <p>{OPENING.lede}</p>
          <h3>Before you start</h3>
          <ul>
            {OPENING.before.map((b) => <li key={b}>{b}</li>)}
          </ul>
        </div>
      </Card>

      {STOPS.map((stop) => (
        <Card key={stop.id} title={`${stop.id} — ${stop.goal}`} note={stop.takes} collapsed>
          <div className="body-pad">
            <h3>Do this</h3>
            <ol>
              {stop.do.map((step) => <li key={step}>{step}</li>)}
            </ol>
            <h3>You know it worked when</h3>
            <ul>
              {stop.done.map((d) => <li key={d}>{d}</li>)}
            </ul>
            {stop.watch && (
              <>
                <h3>Watch out</h3>
                {/* Named rather than left to be discovered: every one of these
                    is a way the step reports success and gives you nothing. */}
                <p><Pill kind="warn">the usual trap</Pill> {stop.watch}</p>
              </>
            )}
          </div>
        </Card>
      ))}

      <Card title="What you have now">
        <div className="body-pad">
          <ul>
            {AFTERWARDS.map((a) => <li key={a}>{a}</li>)}
          </ul>
        </div>
      </Card>
    </div>
  );
}
