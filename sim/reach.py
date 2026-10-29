"""Reach survey (step 3, kinematic): which leaves on the greenhouse's dwarf plants the SO-101 can cut flush.

    python check_approach.py --reach            the survey (about a minute), PASS/FAIL on its own sanity

For every present leaf on the plant in front of the arm (p0), over many random plants, cutpose.CutIK looks for a
flush cut pose, from:
- the base where layout.py puts it (in front of p0, on the aisle centreline),
- the base slid along the aisle (a SPINE move), BASE_SHIFTS,
- the other side of the row (a second aisle; there is none in the mini greenhouse, so that one checks the plants
  only, not the frame).

A pose counts as a clean cut when it's inside what check_approach.py's physics cases cut clean with the stock
fingertips (LEAN_OK, PITCH_OK, TWIST_OK below), and the arm at that pose doesn't overlap the stem, a truss, the
pot or the greenhouse frame (leaves give way, so they don't count; the target's own petiole runs between the
jaws). Fruit may be pushed up to FRUIT_PUSH_OK. Kinematic only: no approach path, no servo sag. Those are step 5's.
"""
import numpy as np
import mujoco

import cutpose as cp
import layout as lo
import stemgen as sg

# --- what check_approach.py's physics cut clean (2026-10-29, 2 mm flush stub, stock fingertips) ---
LEAN_OK = 2.0             # deg. ⚠️ 5 deg of lean already put the moving pad into the stem
PITCH_OK = (-22.0, 10.0)  # deg, fingers off level: -22..+10 cut clean, +52 put the fixed finger along the stem
TWIST_OK = 30.0           # deg: 25 deg cut a 2.1-3.5 mm stub; 55 deg cut but left 4-9 mm
OFFSET = 0.002            # the stub asked for
OVERLAP_OK = 0.0005       # m, as check_approach.BLOCK_PEN
# Fruit hang on sprung pedicels and give way, as a hand reaching into a truss pushes them: B.L.U.S.H lets the
# open fingers press a neighbour by up to 15 mm in its kinematic sweep (expert.NEIGHBOUR_PUSH_M, commit cec86fb).
# Counted apart ("ok, pushes fruit"). The truss stalk (rachis, pedicel, calyx) stays hard: cutting it loses the truss.
FRUIT_PUSH_OK = 0.015
SEEDS = range(100, 120)   # 20 random dwarf plants in front of the arm
BASE_SHIFTS = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)    # m along the aisle (the greenhouse is 0.74 m long)
NEIGHBOUR_SEEDS = (12, 13)  # the plants next to p0 (layout's default row), so the frame and neighbours are there


def _hard_geoms(m):
    """Geoms the arm must not overlap: stems, pots, truss stalks and the greenhouse frame. And the fruit."""
    hard, fruit = set(), set()
    for g in range(m.ngeom):
        n = m.geom(g).name or ""
        cls = sg.seg_class(n)
        if n.startswith("gh_") or n == "floor" or cls in ("stem", "pot", "rachis", "pedicel", "calyx"):
            hard.add(g)
        elif cls == "fruit":
            fruit.add(g)
    return hard, fruit


def _overlap(m, d, arm, hard):
    deepest, what = 0.0, ""
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if ((g1 in arm and g2 in hard) or (g2 in arm and g1 in hard)) and -c.dist > deepest:
            deepest = -c.dist
            other = g2 if g1 in arm else g1
            what = sg.seg_class(m.geom(other).name) or m.geom(other).name
    return deepest, what


def classify(p):
    if p is None:
        return "no IK"
    if abs(p.lean_deg) > LEAN_OK:
        return "lean"
    if not PITCH_OK[0] <= p.pitch_deg <= PITCH_OK[1]:
        return "pitch"
    if p.twist_deg > TWIST_OK:
        return "twist"
    return "ok"


_IK = {}


def survey_base(seed, arm_xy, arm_yaw, frame=True):
    """Per present leaf on p0: (node, rule cuts it?, verdict, pose)."""
    m, d, info = lo.build((seed,) + NEIGHBOUR_SEEDS, arm_xy=arm_xy, arm_yaw=arm_yaw)
    key = (tuple(arm_xy), arm_yaw)
    if key not in _IK:                     # the IK doesn't depend on the plant: one per base pose
        _IK[key] = cp.CutIK(base=np.array([*arm_xy, 0.0]), yaw=arm_yaw)
    ik = _IK[key]
    spec = info["specs"][0]
    cuts = set(sg.label(spec).cuts)
    arm = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("arm/") and m.geom_contype[g]}
    hard, fruit = _hard_geoms(m)
    if not frame:
        hard = {g for g in hard if not (m.geom(g).name or "").startswith("gh_")}
    adr = [m.joint("arm/" + j).qposadr[0] for j in cp.ARM5]
    out = []
    for (k, i), w in info["where"].items():
        if k != 0:
            continue
        lp, _ = spec.leaf[i]
        sols = ik.solve(w["S"], w["a"], w["stem_r"], OFFSET, lp.petiole_r)
        best, verdict = None, "no IK"
        for p in sols:                   # the best pose that is clean and clear of the plant
            v = classify(p)
            if v != "ok":
                verdict = verdict if verdict != "no IK" else v
                continue
            d.qpos[adr] = p.q
            mujoco.mj_forward(m, d)
            pen, what = _overlap(m, d, arm, hard)
            if pen > OVERLAP_OK:
                verdict = f"hits {what}"
                continue
            push, _ = _overlap(m, d, arm, fruit)
            if push > FRUIT_PUSH_OK:
                verdict = "hits fruit"
                continue
            best, verdict = p, "ok" if push <= OVERLAP_OK else "ok, pushes fruit"
            break
        out.append((i, i in cuts, verdict, best))
    return out


def survey():
    fails = 0
    yaw_far = -np.pi / 2
    far_y = lo.ROW_Y + (lo.ROW_Y - lo.AISLE_Y)          # mirror of the aisle across the row
    sx = lo.PLANT_X[0]
    print(f"{len(SEEDS)} dwarf plants in front of the arm (seeds {SEEDS.start}-{SEEDS.stop - 1}); clean = lean <= "
          f"{LEAN_OK:.0f}, finger pitch {PITCH_OK[0]:+.0f}..{PITCH_OK[1]:+.0f}, twist <= {TWIST_OK:.0f} deg "
          f"(check_approach.py) and the arm clear of stem, trusses, pot and frame. Stub {OFFSET * 1000:.0f} mm.\n", flush=True)
    bases = [(f"aisle {dx:+.2f} m", (sx + dx, lo.AISLE_Y), lo.ARM_YAW, True) for dx in BASE_SHIFTS]
    bases += [(f"far side {dx:+.2f} m", (sx + dx, far_y), yaw_far, False) for dx in (-0.10, 0.0, 0.10)]
    table = {}
    for n_done, seed in enumerate(SEEDS):
        print(f"  plant {n_done + 1}/{len(SEEDS)} (seed {seed})", flush=True)
        for name, xy, yaw, frame in bases:
            for i, rule_cut, verdict, _ in survey_base(seed, xy, yaw, frame):
                table.setdefault((seed, i), {"rule": rule_cut})[name] = verdict
    leaves = list(table)
    rule = [k for k in leaves if table[k]["rule"]]
    print(f"{len(leaves)} present leaves, {len(rule)} of them the rule's cuts\n")
    print(f"{'base':18s} {'all leaves':>12s} {'rule cuts':>10s}   why not (all leaves)")
    for name, *_ in bases:
        ok_all = sum(table[k][name].startswith("ok") for k in leaves)
        ok_rule = sum(table[k][name].startswith("ok") for k in rule)
        why = {}
        for k in leaves:
            v = table[k][name]
            if v != "ok":
                why[v] = why.get(v, 0) + 1
        why_s = ", ".join(f"{v} {n}" for v, n in sorted(why.items(), key=lambda t: -t[1]))
        print(f"{name:18s} {ok_all:5d} ({100 * ok_all / len(leaves):3.0f}%) {ok_rule:4d} ({100 * ok_rule / max(len(rule), 1):3.0f}%)   {why_s}")
    print()
    combos = (("one base (today's layout)", ["aisle +0.00 m"]),
              ("base slides along the aisle", [b[0] for b in bases if b[0].startswith("aisle")]),
              ("one base + the far side", ["aisle +0.00 m", "far side +0.00 m"]),
              ("both: slides, both sides", [b[0] for b in bases]))
    for what, names in combos:
        ok_all = sum(any(table[k][n].startswith("ok") for n in names) for k in leaves)
        ok_rule = sum(any(table[k][n].startswith("ok") for n in names) for k in rule)
        print(f"INFO {what:30s}: {ok_all}/{len(leaves)} leaves ({100 * ok_all / len(leaves):.0f}%), "
              f"{ok_rule}/{len(rule)} rule cuts ({100 * ok_rule / max(len(rule), 1):.0f}%) cut flush and clean")
    sane = len(leaves) > 100 and any(table[k]["aisle +0.00 m"].startswith("ok") for k in leaves)
    print(("PASS" if sane else "FAIL") + f" the survey ran on {len(leaves)} leaves and found clean cuts")
    fails += not sane
    return 1 if fails else 0
