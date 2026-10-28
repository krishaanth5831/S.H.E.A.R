#!/usr/bin/env python3
"""Golden cases for the SHEAR leaf rule: small stems written by hand, with the cuts worked out by hand
from the research note's section 5. PASS/FAIL per case, exit 1 on a FAIL. No mujoco, runs anywhere:

  python rule/check_rule.py

Stem notation, bottom (oldest) first, one token per node:
  s scar   g green leaf   y yellow leaf   d diseased leaf   ? unknown leaf
  T:g T:t T:r T:h   truss: green, turning, ripe, harvested
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rule.select_leaves import (ClawView, Conditions, LEAF, Node, SCAR, Settings, TRUSS, UNKNOWN,
                                confirm_cut, select_leaves)

LEAF_TOKENS = {"s": dict(state=SCAR), "g": {}, "y": dict(health="yellow"), "d": dict(health="diseased"),
               "?": dict(state=UNKNOWN)}
STAGE_TOKENS = {"g": "green", "t": "turning", "r": "ripe", "h": "harvested"}


def stem(text):
    out = []
    for tok in text.split():
        if tok.startswith("T:"):
            out.append(Node(TRUSS, stage=STAGE_TOKENS[tok[2:]]))
        else:
            out.append(Node(LEAF, **LEAF_TOKENS[tok]))
    return out


# (name, stem, settings, conditions, expected cuts, expected other decisions {index: decision})
# Indices are into the node list after the gap-fill (only case "unknown gap" changes it).
CASES = [
    ("steady state: below the ripe truss + one above",
     "s s g g T:r g g g T:t g g g T:g", Settings(min_leaves_kept=0), None, [2, 3, 5], {6: "keep", 0: "scar"}),
    ("harvested truss isn't the anchor; pass cap 3, oldest first",
     "g g g T:h g g g T:r g g g", Settings(min_leaves_kept=0), None, [0, 1, 2], {4: "cap", 5: "cap", 6: "cap", 8: "cap"}),
    ("heat hold: nothing above the anchor",
     "s s g g T:r g g g T:t g g g T:g", Settings(min_leaves_kept=0, heat_hold=True), None, [2, 3], {5: "keep"}),
    ("young crop: only yellow below the lowest truss; yellow above is flagged",
     "y g g T:g g y g T:g g g g", Settings(min_leaves_kept=0), None, [0], {1: "keep", 5: "flag"}),
    ("unknown gap: 2 leaves between trusses -> an unknown takes the slot above the anchor",
     "s g T:r g g T:g g g g", Settings(min_leaves_kept=0), None, [1], {3: "unknown", 4: "keep", 5: "keep"}),
    ("yellow above the strip line is flagged, not cut",
     "s g T:r g g y T:g", Settings(min_leaves_kept=0), None, [1, 3], {5: "flag", 4: "keep"}),
    ("keep guard: 8 present, keep 6 -> 2 cuts, the youngest candidate stays",
     "s s g g T:r g g g T:t g g g T:g", Settings(min_leaves_kept=6), None, [2, 3], {5: "guard"}),
    ("keep guard at the default 16 on a short stem: no cuts",
     "s s g g T:r g g g T:t g g g T:g", Settings(), None, [], {2: "guard", 3: "guard", 5: "guard"}),
    ("wound gate: RH 92 % stops the pass",
     "s s g g T:r g g g", Settings(min_leaves_kept=0), Conditions(rh=92.0), [], {2: "blocked", 5: "blocked"}),
    ("wet canopy stops the pass",
     "s g T:r g g g", Settings(min_leaves_kept=0), Conditions(canopy_wet=True), [], {1: "blocked"}),
    ("no truss at all: nothing is cut",
     "g y g", Settings(min_leaves_kept=0), None, [], {1: "flag"}),
    ("an unknown below the line is never cut",
     "? s g T:r", Settings(min_leaves_kept=0), None, [2], {0: "unknown", 1: "scar"}),
    ("determinate dwarf (leaves_per_truss 1): a truss every leaf is no gap",
     "s g T:r g T:t g T:g", Settings(min_leaves_kept=0, leaves_per_truss=1), None, [1, 3], {5: "keep"}),
    ("same stem as indeterminate (3): the gaps fill with unknowns and nothing above the anchor is cut",
     "s g T:r g T:t g T:g", Settings(min_leaves_kept=0), None, [1], {3: "unknown"}),
    ("diseased below the line is cut like any other",
     "d g T:t g", Settings(min_leaves_kept=0), None, [0, 1, 3], {}),
]

VETO_CASES = [
    ("petiole, clear of trusses, at the stem", ClawView(True, 0.02, True), True),
    ("truss stalk in the jaw", ClawView(False, 0.02, True), False),
    ("blade path 5 mm from a truss", ClawView(True, 0.005, True), False),
    ("blade off the stem surface", ClawView(True, 0.02, False), False),
]


def main():
    fails = 0
    for name, text, settings, cond, cuts, other in CASES:
        sel = select_leaves(stem(text), settings, cond)
        got_other = {i: sel.decision.get(i) for i in other}
        ok = sel.cuts == cuts and got_other == other
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: cuts {sel.cuts}" + ("" if ok else f" (want {cuts}; "
              f"decisions {got_other}, want {other})"))
    for name, view, want in VETO_CASES:
        ok_cut, reason = confirm_cut(view)
        ok = ok_cut == want
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} veto, {name}: {'cut' if ok_cut else 'skip'} ({reason})")
    print("\nALL PASS" if not fails else f"\n{fails} FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
