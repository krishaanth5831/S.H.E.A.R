#!/usr/bin/env python3
"""Step 2 of the SHEAR sim plan: random tomato plants, and the leaf rule's labels on them.

  python check_plant.py --windowed --seed 3    watch one plant settle. Leaves the rule cuts are red, flagged
                                               (yellow/diseased above the line: for a person) orange, held
                                               back by the keep guard or the pass cap blue. --truth shows the
                                               plant's own colours instead. The terminal prints the stem.
  python check_plant.py                        headless checks, PASS/FAIL per check, exit 1 on a FAIL

Checks:
- the rule's golden cases (rule/check_rule.py: hand-built stems, cuts worked out by hand)
- over many seeds (no physics): the plant keeps three leaves per truss; every cut is a present band leaf
  below the strip line, oldest first, within the pass cap and the keep guard; flags are yellow/diseased
  leaves above the line; and the labels vary (no cuts, some cuts, full cap, guard, young crop, flags)
- a few seeds built in MuJoCo: compiles inside the geom budget with the arm, every leaf and fruit settles,
  nothing is on the floor, every geom has a segmentation class, every cut point is on the stem surface

Run in the lerobot env (it has mujoco), under the 9 GB cap:
  systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 python check_plant.py
Writes nothing. Needs ./fetch_so101.sh once (only for the arm's geom count).
"""
import argparse
import collections
import pathlib
import sys
import time

import mujoco
import numpy as np

import plant as pl
import scene as sc
import stemgen as sg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rule import check_rule  # noqa: E402
from rule.select_leaves import LEAF, PRESENT, TRUSS, GREEN, LEAVES_PER_TRUSS, Settings  # noqa: E402

N_SEEDS = 400          # property checks (rule + generator only, fast)
PHYS_SEEDS = (0, 1, 2, 3, 4, 5)
SETTLE_S = 2.0
LEAF_SPEED = 0.01      # m/s: a leaf base or fruit still moving faster than this after SETTLE_S hasn't settled
HINGE_SPEED = 0.05     # rad/s: a midrib hinge still swinging faster than this hasn't settled (step 1 reached 0.25
                       # ringing before MID_ARMATURE; unverified threshold)
POINT_TOL = 0.001      # label cut point vs the petiole's real base after settling, m
FRUIT_SPEED = 0.02     # fruit hang on soft tethers and sprung pedicels (B.L.U.S.H settles in ~1 s)
SURFACE_TOL = 1e-6     # cut point to stem surface, m

# Label colours in the window (leaflets and midrib of each leaf).
LABEL_RGBA = {"cut": (0.85, 0.10, 0.10, 1), "flag": (0.95, 0.55, 0.10, 1), "guard": (0.20, 0.40, 0.90, 1),
              "cap": (0.20, 0.40, 0.90, 1), "blocked": (0.20, 0.40, 0.90, 1)}


def build(spec):
    body, eq, where = sg.plant_xml(spec)
    m = mujoco.MjModel.from_xml_string(sc.world_xml(body, eq))
    return m, mujoco.MjData(m), where


def free_bodies(m, prefix_test):
    """dof addresses of free joints whose body name passes prefix_test."""
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and prefix_test(m.body(m.jnt_bodyid[j]).name):
            out.append(m.jnt_dofadr[j])
    return out


def stem_table(spec, sel):
    lines = []
    for i in reversed(range(len(spec.nodes))):
        n = spec.nodes[i]
        where = "band" if spec.in_band(i) else "above"
        if n.kind == TRUSS:
            nf = spec.truss.get(i, (0,))[0]
            mark = "  <- anchor" if i == sel.anchor else ""
            lines.append(f"  {i:2d} {where:5s} truss  {n.stage:9s} {nf} fruit{mark}")
        else:
            line = "  -- strip line" if i == sel.strip_line else ""
            lines.append(f"  {i:2d} {where:5s} leaf   {n.state:7s} {n.health if n.state == PRESENT else '':8s} "
                         f"{sel.decision[i]}{line}")
    return "\n".join(lines)


def property_checks(report):
    s = Settings()
    stats = collections.Counter()
    bad = collections.defaultdict(list)
    for seed in range(N_SEEDS):
        spec = sg.make_plant(seed)
        sel = sg.label(spec, s)
        nodes = spec.nodes
        if len(sel.nodes) != len(nodes):
            bad["gap-fill inserted unknowns into sim truth"].append(seed)
            continue
        tr = [i for i, n in enumerate(nodes) if n.kind == TRUSS]
        if any(sum(nodes[k].kind == LEAF for k in range(a + 1, b)) != LEAVES_PER_TRUSS for a, b in zip(tr, tr[1:])):
            bad["not three leaves per truss"].append(seed)
        for i in sel.cuts:
            n = nodes[i]
            if n.kind != LEAF or n.state != PRESENT:
                bad["a cut that isn't a present leaf"].append(seed)
            if i > sel.strip_line:
                bad["a cut above the strip line"].append(seed)
            if not spec.in_band(i):
                bad["a cut outside the built band"].append(seed)
            if spec.young and n.health == GREEN:
                bad["a green leaf cut on a young crop"].append(seed)
        if sel.cuts != sorted(sel.cuts):
            bad["cuts not oldest first"].append(seed)
        if len(sel.cuts) > s.max_cuts_per_pass:
            bad["over the pass cap"].append(seed)
        present = sum(n.kind == LEAF and n.state == PRESENT for n in nodes)
        if sel.cuts and present - len(sel.cuts) < s.min_leaves_kept:
            bad["keep guard broken"].append(seed)
        for i, dcs in sel.decision.items():
            if dcs == "flag" and (i <= sel.strip_line or nodes[i].health == GREEN):
                bad["a flag that isn't yellow/diseased above the line"].append(seed)
        dec = set(sel.decision.values())
        stats["0 cuts"] += not sel.cuts
        stats[">=1 cut"] += bool(sel.cuts)
        stats["full cap"] += len(sel.cuts) == s.max_cuts_per_pass
        stats["guard bites"] += "guard" in dec
        stats["cap bites"] += "cap" in dec
        stats["flags"] += "flag" in dec
        stats["young crop"] += spec.young
        stats["missed leaf below a scar"] += any(
            nodes[i].state == PRESENT and any(nodes[k].kind == LEAF and nodes[k].state == "scar"
                                              for k in range(i + 1, len(nodes)))
            for i in range(len(nodes)) if nodes[i].kind == LEAF)
    for name in ("gap-fill inserted unknowns into sim truth", "not three leaves per truss",
                 "a cut that isn't a present leaf", "a cut above the strip line", "a cut outside the built band",
                 "a green leaf cut on a young crop", "cuts not oldest first", "over the pass cap",
                 "keep guard broken", "a flag that isn't yellow/diseased above the line"):
        seeds = bad.get(name, [])
        report(not seeds, f"{N_SEEDS} plants, never {name}" + (f": seeds {seeds[:8]}" if seeds else ""))
    need = ("0 cuts", ">=1 cut", "full cap", "guard bites", "cap bites", "flags", "young crop", "missed leaf below a scar")
    missing = [k for k in need if stats[k] == 0]
    report(not missing, "labels vary across seeds: " + ", ".join(f"{k} {stats[k]}" for k in need)
           + (f" (never: {missing})" if missing else ""))


def physics_checks(report):
    arm = sc.arm_spec().compile()
    n_arm = arm.ngeom
    for seed in PHYS_SEEDS:
        spec = sg.make_plant(seed)
        t0 = time.time()
        m, d, where = build(spec)
        sel = sg.label(spec)
        total = m.ngeom - 1 + n_arm               # floor counted once (arm model has none)
        report(total <= sg.GEOM_BUDGET, f"seed {seed}: compiles, {m.ngeom - 1} plant geoms + {n_arm} arm = {total} "
               f"(soft budget {sg.GEOM_BUDGET}); {len(where)} leaves, "
               f"{sum(1 for i in spec.truss)} trusses, {m.nbody} bodies")
        names = [m.geom(g).name for g in range(m.ngeom) if m.geom(g).name != "floor"]
        unclassed = [g for g in names if sg.seg_class(g) is None]
        wrong_node = [g for g in names if g != "stem" and (sg.node_of(g) is None or
                      (spec.nodes[sg.node_of(g)].kind == TRUSS) != (sg.seg_class(g) in ("rachis", "calyx", "pedicel", "fruit")))]
        report(not unclassed and not wrong_node, f"seed {seed}: every geom has a segmentation class and its node "
               f"({len(names)} geoms; unclassed {unclassed[:3]}, wrong node {wrong_node[:3]})")
        offs = [abs(np.linalg.norm(np.asarray(spec.nodes[i].point)[:2] - np.asarray(sg.STEM_XY)) - spec.stem_r)
                for i in where]
        report(all(o < SURFACE_TOL for o in offs), f"seed {seed}: {len(where)} cut points (label), all on the stem "
               f"surface (worst {max(offs, default=0) * 1e3:.4f} mm)")

        leaf_dofs = free_bodies(m, lambda b: b.endswith("leaf"))
        fruit_dofs = free_bodies(m, lambda b: "_t" in b and not b.endswith("_ped"))
        floor = m.geom("floor").id
        for _ in range(round(SETTLE_S / m.opt.timestep)):
            mujoco.mj_step(m, d)
        v_leaf = max((np.linalg.norm(d.qvel[a:a + 3]) for a in leaf_dofs), default=0.0)
        v_fruit = max((np.linalg.norm(d.qvel[a:a + 3]) for a in fruit_dofs), default=0.0)
        hinges = [m.jnt_dofadr[j] for j in range(m.njnt) if "midrib" in m.joint(j).name]
        w_hinge = max((abs(d.qvel[a]) for a in hinges), default=0.0)
        # Where each petiole really is after settling: leaf body origin (on the stem axis) + its x axis * r_s
        drift = [np.linalg.norm(d.xpos[m.body(w["prefix"] + "leaf").id]
                                + d.xmat[m.body(w["prefix"] + "leaf").id].reshape(3, 3)[:, 0] * w["r_s"]
                                - np.asarray(spec.nodes[i].point)) for i, w in where.items()]
        # Droop: the leaf tip vs where a straight leaf at its petiole's angle would put it (negative = drooped)
        tips = [d.geom_xpos[m.geom(w["prefix"] + "leaflet_tip").id][2] - spec.nodes[i].point[2]
                - spec.leaf[i][0].length * np.sin(spec.leaf[i][1]) for i, w in where.items()]
        welds = int(d.eq_active.sum()), m.neq
        on_floor = sorted({m.geom(c.geom1 if c.geom2 == floor else c.geom2).name
                           for c in d.contact[:d.ncon] if floor in (c.geom1, c.geom2)})
        finite = bool(np.all(np.isfinite(d.qpos)))
        report(finite and v_leaf < LEAF_SPEED and v_fruit < FRUIT_SPEED and w_hinge < HINGE_SPEED
               and welds[0] == welds[1],
               f"seed {seed}: settles in {SETTLE_S:.0f} s: leaf bases {v_leaf:.4f} m/s, midrib hinges {w_hinge:.4f} "
               f"rad/s, fruit {v_fruit:.4f} m/s, welds {welds[0]}/{welds[1]} on ({time.time() - t0:.1f} s wall)")
        report(max(drift, default=0) < POINT_TOL and all(t < 0 for t in tips),
               f"seed {seed}: label cut points match the settled petioles (worst {max(drift, default=0) * 1e3:.2f} mm); "
               f"every leaf droops below a straight leaf ({-max(tips, default=0) * 100:.0f} to {-min(tips, default=0) * 100:.0f} cm)")
        report(not on_floor, f"seed {seed}: nothing on the floor" + (f": {on_floor[:4]}" if on_floor else ""))


def run_checks():
    fails = 0

    def report(ok, msg):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + msg, flush=True)

    print("-- the rule's golden cases (rule/check_rule.py)")
    fails += check_rule.main() != 0
    print("\n-- the generator and the rule over many plants")
    property_checks(report)
    print("\n-- plants built in MuJoCo")
    physics_checks(report)
    print("\nALL PASS" if not fails else f"\n{fails} FAIL")
    return 1 if fails else 0


def windowed(seed, truth, leaf_len):
    import mujoco.viewer
    spec = sg.make_plant(seed, leaf_len)
    m, d, where = build(spec)
    sel = sg.label(spec)
    print(f"seed {seed}{' (young crop)' if spec.young else ''}: stem top first, 'above' = abstract (no geoms)")
    print(stem_table(spec, sel))
    print(f"cuts {[spec.nodes[i].id for i in sel.cuts]}  (red = cut, orange = flag, blue = guard/cap)")
    if not truth:
        for g in range(m.ngeom):
            i = sg.node_of(m.geom(g).name)
            if i is not None and sg.seg_class(m.geom(g).name) in ("leaflet", "midrib", "petiole"):
                rgba = LABEL_RGBA.get(sel.decision.get(i))
                if rgba:
                    m.geom_rgba[g] = rgba
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (0, 0, 0.4)
        v.cam.distance, v.cam.elevation = 1.2, -10
        while v.is_running():
            t = time.time()
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0.0, m.opt.timestep - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowed", action="store_true", help="watch one plant settle, leaves coloured by the rule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--truth", action="store_true", help="windowed: the plant's own colours, not the labels")
    ap.add_argument("--leaf-len", type=float, default=pl.LEAF_LEN, help="stem surface to leaf tip, m (unverified)")
    a = ap.parse_args()
    if a.windowed:
        windowed(a.seed, a.truth, a.leaf_len)
        return 0
    return run_checks()


if __name__ == "__main__":
    sys.exit(main())
