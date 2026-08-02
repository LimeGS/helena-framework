"""Which image each role runs, identified by content.

A host is not a machine somebody installed software on. It is a machine that
runs a declared image, and the image is the unit of reproducibility: the same
digest on two hosts is the same bytes, and a tag is not that.

Identity is the RootFS layer chain, and that choice was learned the hard way.

`docker inspect --format {{.Id}}` is not comparable between hosts. The classic
image store reports the *config* digest; the containerd store reports the *OCI
manifest* digest. Two hosts holding the identical image reported
sha256:eabe85d9 and sha256:49ad80e3 and their sizes differed by 17 MB, purely
from how each store accounts for blobs -- and that looked exactly like two
different images built from one commit.

It was not. Loading one host's checksum-verified export into the other produced
no new image, because the content was already there. The layers matched
outright.

So comparison is by RootFS layers: those are the filesystem itself, and they
are the same string on both stores. A digest that only some daemons agree on is
not an identity, it is a local handle.
"""

from __future__ import annotations

# What each role needs present before it can claim work.
#
# These are the images the current recipes produced, carrying their toolchain
# receipts. They are distributed by shipping the bytes -- docker save piped to
# docker load -- and never by rebuilding per host, because rebuilding is what
# produced two different `campaignx-villa:local` in the first place.
#
# A role with no digest is honest about being undeployed rather than defaulting
# to a tag, which identifies nothing.
ROLE_IMAGES: dict[str, dict[str, str | None]] = {
    "segment": {
        "image": "helena-villa",
        "digest": "d6b1a90bccf1+d85193ee52f1+eea6269de4e3",
        "why": "VC3D grows surfaces. CPU only -- use_cuda is const false in the "
               "growth profile -- so any host can take this role.",
    },
    "render": {
        "image": "helena-villa",
        "digest": "d6b1a90bccf1+d85193ee52f1+eea6269de4e3",
        "why": "vc_render_tifxyz turns a surface into a layer stack. No GPU.",
    },
    "ink": {
        "image": "helena-ink",
        "digest": None,
        "why": "The only stage that needs a GPU worth having.",
    },
    "mesh": {
        "image": "helena-scrollfiesta",
        "digest": None,   # not yet fingerprinted by layer chain
        "why": "Comparative backend. INTERNAL_RESEARCH_ONLY; its surfaces are "
               "not catalogued.",
    },
}


def images_for(roles: list[str]) -> dict[str, dict]:
    """The images a host needs, given what it is being asked to do."""
    return {role: ROLE_IMAGES[role] for role in roles if role in ROLE_IMAGES}


def drift(expected: dict[str, dict], present: dict[str, str]) -> list[dict]:
    """Where a host disagrees with what its roles require.

    `present` maps image name to that host's RootFS layer chain -- not to a
    reported id, which two daemons will disagree about for the same image.

    Three outcomes, and they are not the same problem: missing means the role
    cannot run at all; mismatched means it will run something nobody chose; and
    undeclared means nobody has said which content is correct, so the host
    cannot be wrong yet -- but neither can it be right.
    """
    findings = []
    for role, spec in expected.items():
        name = str(spec["image"])
        want = spec.get("digest")
        have = present.get(name)
        if want is None:
            findings.append({"role": role, "image": name, "state": "UNDECLARED",
                             "have": have,
                             "detail": "no digest is declared for this role, so "
                                       "nothing can be verified against it"})
        elif have is None:
            findings.append({"role": role, "image": name, "state": "MISSING",
                             "want": want, "detail": "the image is not on this host"})
        elif have != want:
            findings.append({"role": role, "image": name, "state": "MISMATCH",
                             "want": want, "have": have,
                             "detail": "same name, different bytes"})
    return findings
