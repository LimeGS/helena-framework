# Two lanes: certified and exploratory

A gate should answer *may this be certified*, not *may this run*.

This is an exploration framework with a dynamic workflow. Running one phase
today, against something arbitrary, mid-campaign, with an upload bound to
nothing, is an ordinary thing to want. The platform refusing it outright is the
platform deciding which questions may be asked. What such a run does not get is
a receipt: no certification, no place in the chain of trust.

`framework/contracts/execution_mode.py` is the mechanism. This note is the plan
for applying it, and the boundary of what must never become a lane.

## The two lanes

`CERTIFIED` is the default and is what every existing caller gets by passing
nothing. Every precondition holds exactly as it does today; nothing in this work
relaxes one.

`EXPLORATORY` is declared by the caller. An unmet chain-of-trust precondition no
longer stops the work: it is recorded, and the output says in itself that it
certifies nothing and why.

The lane is a declaration, not a score earned along the way. An exploratory run
that happened to satisfy everything still certifies nothing — otherwise one
request would certify or not depending on the state of the system when it ran,
which is the opposite of a chain of trust.

## The part that has to be right

The permissive half is easy. The exclusion is what keeps this from being an
acceptance-gate bypass with a friendlier name, and it fails closed four ways:

- a document with **no stamp** is not certified. Everything written before this
  existed carries no stamp, and absence of a claim is not a claim.
- a stamp that says `CERTIFIED` while also recording why it is not is a
  **forgery**. Adding one key must not launder an exploratory receipt.
- a certified document that **quotes** an uncertified one, at any depth, is not
  certified. The chain is only as good as what it stands on.
- `require_certified_input` refuses at the point of reading, so a certified run
  cannot consume exploratory evidence even by accident.

## What never becomes a lane

Two families of refusal stay refusals, in every phase:

**Integrity.** A tampered binding, a drifted identity, a hash that does not
match. Running anyway means acting on evidence known to be corrupt: the answer
would be *wrong*, not merely uncertified. In `panel/app.py` this is 22 of the
409s — "job has a tampered persisted control binding", "selected P0 artifact
identity drifted", "selected P0 contains a partial or tampered control marker".

**Authorization.** Who may act, and on what. A lane is not a login.

Also unchanged: immutability triggers, lease ownership on queue writes, and
ink-blindness. An exploratory run is still ink-blind — the lane is about what a
result may claim, not about what the system is allowed to look at.

## Where it applies, phase by phase

Counted from the refusals in `panel/app.py` and
`framework/stages/01-segmentation/fleet/campaign_decision.py`. The split below
is a first pass by shape of message; each site still needs reading before it
changes, because a few are integrity checks phrased as preconditions.

| Phase | Refuses today because | Becomes a lane |
|---|---|---|
| P0 source freeze | the mission selected no scrolls; the source lock is incomplete | yes — freeze what is there, uncertified |
| P1 candidate preflight | no explicit selected P0; sample is not the frozen control sample | yes — measure the snapshot named, uncertified |
| P1 seed creation | the mission is a controlled campaign and the path is not `bootstrap` with a budget pair | yes, **and this one needs review before it moves**: see below |
| P2–P5 phase runs | prerequisite phase produced no certified artifact | yes — run against the artifact named |
| QC / routing | the surface has no routing receipt; area unusable | partly — an unusable measurement is integrity, a missing receipt is a precondition |
| P7 ink | the profile is not validated; no certified P5 input | yes — run the model, certify nothing |
| Human review | no certified upstream chain | yes — record the opinion, uncertified |
| Promotion / acceptance | anything | **no.** Promotion is the act of certifying |

Rough counts in `panel/app.py`: 154 refusals with status 409, of which 109 carry
a literal message — 22 integrity, 87 chain-of-trust preconditions.

## The one that needs a decision before it moves

The campaign's P1 creation authority (`_authorize_controlled_p1_creation`) is
the gate the development control hit: for a mission bound to the First Letters
campaign, the only authorized creation path is `bootstrap` with a budget receipt
pair. Turning it into a lane means uncertified tasks can exist inside a campaign
mission.

That is defensible — they cannot be promoted, counted, or read as evidence — but
it puts exploratory work in the same queue as campaign work, where the budget
discipline lives. It should not change without someone deciding that
deliberately.

The development control did not need it: a control is not a campaign, so its
mission carries no campaign binding, and the gate does not apply to it at all.
