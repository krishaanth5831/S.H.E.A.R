"""The SHEAR leaf rule (v1): which leaves come off one stem in this pass, oldest first.

    sel = select_leaves(nodes, Settings())       nodes from the bottom of the stem up
    sel.cuts                                     node indices to cut, in order (oldest first)
    sel.decision[i]                              what happened to node i, and why
    ok, reason = confirm_cut(ClawView(...))      the veto at the claw, just before each cut

One function, shared by the sim labeller, the sim judge and the real robot, the way B.L.U.S.H shares
`ripeness/gate/birdseye.py`. Plain Python, no mujoco: perception (or sim truth) builds the node list.

Source: Vinea vault, `08 Modules/SHEAR/(C) Deleafing Research — Why, Which Leaves, and the SHEAR Rule
(Oct 2026)`, section 5. The steps below are numbered as there. Cutting is the grower's rule, carried out
by the robot; the learned part of SHEAR is the cut itself, not the choice.

Errors ranked by cost: cutting the main stem > cutting a truss stalk > leaving a stub > skipping a leaf.
Skipping is the cheap failure, so every unclear case here skips.
"""
from dataclasses import dataclass, field
from typing import Optional

LEAF, TRUSS = "leaf", "truss"
# Leaf state. "unknown" = perception can't tell (hidden, or a scar not seen), or the gap-fill added it.
PRESENT, SCAR, UNKNOWN = "present", "scar", "unknown"
GREEN, YELLOW, DISEASED = "green", "yellow", "diseased"
# Truss stages: B.L.U.S.H's ripeness classes, plus harvested (a bare stalk, or gone).
STAGES = ("green", "turning", "ripe", "harvested")
TURNED = ("turning", "ripe")

# A cultivated indeterminate tomato grows three leaves, then a truss, with no end (Development 125:1979,
# in the research note). Used to fill gaps: fewer leaves than this between two trusses = some unseen.
LEAVES_PER_TRUSS = 3


@dataclass
class Node:
    """One node of the stem, bottom (oldest) first."""
    kind: str                        # LEAF or TRUSS
    state: str = PRESENT             # leaves: PRESENT / SCAR / UNKNOWN
    health: str = GREEN              # leaves: GREEN / YELLOW / DISEASED (only meaningful when present)
    stage: str = "green"             # trusses: one of STAGES
    point: Optional[tuple] = None    # leaves: 3D point where the petiole meets the stem surface
    id: Optional[str] = None         # caller's name for it (sim: body prefix)
    filled: bool = False             # True if the gap-fill added it (not seen at all)


@dataclass
class Settings:
    """The grower's settings (research note, section 5 table)."""
    # 1 leaf above the anchor truss comes off too ("all leaves below and one leaf above the bottom truss",
    # Greenhouse Grower high-wire guide). 0 while heat_hold is on.
    strip_above_anchor: int = 1
    # ⚠️ PLACEHOLDER, ask growers (grower-call question 1). 16 comes from an open-source sim (misina-lab),
    # not from practice. Counts present leaves on the whole stem, including those above the work band.
    min_leaves_kept: int = 16
    # "Usually no more than three leaves are removed at a time on one tomato plant" (Greenhouse Grower).
    max_cuts_per_pass: int = 3
    # Hold back in hot, bright weather: no leaf above the anchor comes off (sunscald; weak source).
    heat_hold: bool = False
    # Don't wound a wet canopy: no cuts at all at or over this relative humidity, % (extension sources put
    # the Botrytis line near 90 % RH; weak, grower setting).
    wound_gate_rh: float = 90.0


@dataclass
class Conditions:
    """The greenhouse at the start of the pass. None = not measured (the wound gate then doesn't block)."""
    rh: Optional[float] = None
    canopy_wet: bool = False


@dataclass
class Selection:
    nodes: list                                   # the input, with gap-filled unknowns inserted
    cuts: list = field(default_factory=list)      # indices into nodes, oldest first
    decision: dict = field(default_factory=dict)  # leaf index -> one of DECISIONS
    anchor: Optional[int] = None                  # index of the anchor truss, None for a young crop
    strip_line: Optional[int] = None              # leaves at index <= this are below the line
    blocked: Optional[str] = None                 # why the whole pass is off, if it is


# What happened to each leaf node.
DECISIONS = {
    "cut": "selected for this pass",
    "keep": "stays: above the strip line, or green on a young crop",
    "flag": "yellow or diseased above the strip line: for a person, not cut",
    "guard": "below the line, but cutting it would leave fewer than min_leaves_kept",
    "cap": "below the line, but over max_cuts_per_pass: next pass",
    "unknown": "can't be seen: never cut. Look again from another pose, or skip",
    "scar": "already gone",
    "blocked": "below the line, but the wound gate stopped the pass",
}


def fill_gaps(nodes):
    """Step 1. Between two trusses there are LEAVES_PER_TRUSS leaf nodes (present, scar or unknown). If
    fewer show, the missing ones go in as UNKNOWN right above the lower truss. ⚠️ Where they really sit
    isn't known; putting them at the bottom of the gap is the cautious choice: above the anchor they take
    the strip_above_anchor slot, so a visible leaf further up is never over-stripped.
    Only gaps between two trusses are filled: below the lowest truss there's nothing to count against."""
    trusses = [i for i, n in enumerate(nodes) if n.kind == TRUSS]
    out = list(nodes)
    for lo, hi in reversed(list(zip(trusses, trusses[1:]))):
        missing = LEAVES_PER_TRUSS - sum(n.kind == LEAF for n in nodes[lo + 1:hi])
        for _ in range(max(0, missing)):
            out.insert(lo + 1, Node(LEAF, state=UNKNOWN, filled=True))
    return out


def select_leaves(nodes, settings=None, conditions=None):
    s = settings or Settings()
    c = conditions or Conditions()
    nodes = fill_gaps(nodes)                                            # step 1
    sel = Selection(nodes=nodes)
    leaves = [i for i, n in enumerate(nodes) if n.kind == LEAF]
    for i in leaves:
        sel.decision[i] = {SCAR: "scar", UNKNOWN: "unknown"}.get(nodes[i].state, "keep")

    trusses = [i for i, n in enumerate(nodes) if n.kind == TRUSS]
    turned = [i for i in trusses if nodes[i].stage in TURNED]
    if turned:                                                          # step 2
        sel.anchor = turned[0]
        above = s.strip_above_anchor if not s.heat_hold else 0
        line = sel.anchor                                               # step 3
        for i in (j for j in leaves if j > sel.anchor):
            if above == 0:
                break
            line, above = i, above - 1
        sel.strip_line = line
        cand = [i for i in leaves if i < line + 1 and nodes[i].state == PRESENT]   # step 4
    else:
        # Young crop: nothing has turned. Only yellow or diseased leaves below the lowest truss come off.
        # No truss at all: nothing is cut.
        lowest = trusses[0] if trusses else -1
        sel.strip_line = lowest
        cand = [i for i in leaves if i < lowest and nodes[i].state == PRESENT and nodes[i].health != GREEN]
    for i in leaves:
        if i > sel.strip_line and nodes[i].state == PRESENT and nodes[i].health != GREEN:
            sel.decision[i] = "flag"

    # Step 5, keep guard. Unknowns count as not there, so the guard errs toward keeping.
    present = sum(nodes[i].state == PRESENT for i in leaves)
    allowed = max(0, present - s.min_leaves_kept)
    for i in cand[allowed:]:                       # dropped from the top: the youngest candidates
        sel.decision[i] = "guard"
    cand = cand[:allowed]
    for i in cand[s.max_cuts_per_pass:]:           # step 6, pass cap
        sel.decision[i] = "cap"
    cand = cand[:s.max_cuts_per_pass]

    if c.canopy_wet or (c.rh is not None and c.rh >= s.wound_gate_rh):
        sel.blocked = "canopy wet" if c.canopy_wet else f"RH {c.rh:.0f} % >= {s.wound_gate_rh:.0f} %"
        for i in cand:
            sel.decision[i] = "blocked"
        cand = []
    for i in cand:
        sel.decision[i] = "cut"
    sel.cuts = cand
    return sel


@dataclass
class ClawView:
    """What the claw camera says about the stalk in the jaw, just before the close. Perception fills it
    (sim step 4); until then the sim judge fills it from truth."""
    is_petiole: bool                 # not a truss stalk, a side shoot or the main stem
    truss_clearance_m: float         # closest truss stalk or fruit to the blade path
    blade_at_stem_surface: bool      # the blades close across the petiole at the stem, without the stem


# Blade path to the nearest truss stalk or fruit. ⚠️ unverified guess: set it from step 3's truss contacts.
TRUSS_MARGIN_M = 0.01


def confirm_cut(view, margin=TRUSS_MARGIN_M):
    """Step 7, the veto at the claw. Precision-tuned like B.L.U.S.H's claw veto: any doubt skips.
    v1 never cuts a side shoot, so a side shoot fails is_petiole. Returns (ok, reason)."""
    if not view.is_petiole:
        return False, "not a leaf petiole"
    if view.truss_clearance_m < margin:
        return False, f"blade path {view.truss_clearance_m * 1000:.0f} mm from a truss (margin {margin * 1000:.0f} mm)"
    if not view.blade_at_stem_surface:
        return False, "blade not at the stem surface"
    return True, "ok"
