#!/usr/bin/env python3
"""Step 5 of the SHEAR sim plan: the scripted cutter (cutter.py) and its RRT routes (rrt.py), checked.

  python check_cutter.py                 PASS/FAIL checks on one known plant, exit 1 on a FAIL (~30 s)
  python check_cutter.py --windowed      watch that row's whole pass in the MuJoCo window, in real time (it loops)
  python check_cutter.py --start N       another row: generate_cuts.py's attempt number N (default START)

Checks:
- the known plant has a cut the cutter can plan (pot turned, as generate_cuts.py does)
- the RRT route clears stems, pots and the frame by the clearance at every point, checked 4x finer than it was
  planned; the blades stay clear of truss stalks
- when the route bends, the straight joint-space line would have hit something (INFO when the straight line is
  already clear)
- played in the physics from the start pose, the judge passes it: the right leaf, a flush stub, nothing pressed
- the arm gets back to the start pose on a new RRT route without pressing anything

Run in the lerobot env under the 9 GB cap. Writes nothing.
"""
import argparse
import sys
import time

import numpy as np

import cutter as ct
import rrt

START = 11                   # generate_cuts.py row attempt 11 = seeds 700033..35, a row with a plannable cut
# ⚠️ ...until the v1 rig (2026-11-06: the arm's base 5 cm back, floor an obstacle). Since then this row plans no cut
# and the first check FAILs (it still passes at 6c1c31a^). Pick a new row before trusting this check again.


def planned(start):
    """The same row, pot turns and first plan on p0 that generate_cuts.py makes for this attempt."""
    import generate_cuts as gc
    seeds = tuple(gc.SEED_BASE + 3 * start + j for j in range(3))
    rng = np.random.default_rng(seeds[0])
    scn = ct.Scene(seeds, turns=gc.pot_turns(seeds, rng))
    p = ct.plan_cut(scn, rng, plant=0)
    return scn, (p if isinstance(p, ct.Plan) else None), rng


def run_checks(start):
    fails = 0

    def report(ok, msg):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + msg, flush=True)

    scn, p, rng = planned(start)
    report(p is not None, f"plant {700_000 + 3 * start}: the cutter plans a cut" +
           (f" on leaf {scn.info['specs'][0].nodes[p.node].id} ({len(p.to_leaf)} route points, blade pair "
            f"{p.pose.face}, lean {p.pose.lean_deg:.1f} deg)" if p else ""))
    if p is None:
        print(f"\n{fails} FAIL")
        return fails
    chk, target = scn.checker(p.prefix)
    keep = rrt.STEP
    rrt.STEP = keep / 4
    try:
        hard_clear = chk.path_ok(p.to_leaf, soft=False)
    finally:
        rrt.STEP = keep
    report(hard_clear, f"the RRT route keeps {rrt.CLEARANCE * 1000:.0f} mm from stems, pots and the frame, and the "
                       f"blades {rrt.BLADE_CLEARANCE * 1000:.0f} mm from truss stalks, checked 4x finer than planned"
                       + ("" if not p.brushes_leaves else " (it may brush kept leaves: no route cleared them)"))
    if len(p.to_leaf) == 2:
        print("INFO the route is the straight joint-space line: it is already clear"
              + (" of stems, pots and the frame (no route cleared the kept leaves' stalks too)"
                 if p.brushes_leaves else ""))
    else:
        bumps = chk.hits(scn.home) or not chk.segment_free(scn.home, p.to_leaf[-1], soft=not p.brushes_leaves)
        report(bumps, f"the RRT route bends around things ({len(p.to_leaf)} points): the straight joint-space "
                      f"line from HOME to the pre-cut pose would {'hit something' if bumps else 'be clear too'}")
    ep = ct.Episode(scn, rng)
    ep.hold("start", ct.HOLD_S)
    res = ep.cut(p)
    back = ep.go_start()
    report(res["verdict"] == "PASS", f"played in the physics: {res['verdict']}, stub {res['stub_mm']} mm, aim error "
           f"{res['aim_error_mm']} mm, peaks {res['peaks_n']}" + (f" ({'; '.join(res['why'])})" if res["why"] else ""))
    if res["notes"]:
        print("INFO " + "; ".join(res["notes"]))
    report(not back["pressed"], f"back to the start pose on a new RRT route ({back['route_points']} points), "
                                f"peaks {back['peaks_n']}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return fails


def windowed(start):
    import mujoco.viewer
    while True:
        scn, p, rng = planned(start)
        print(f"row {700_000 + 3 * start}: the whole pass from the start pose, plant by plant. Close the window "
              f"(Ctrl+C on Wayland) to stop.", flush=True)
        with mujoco.viewer.launch_passive(scn.m, scn.d) as v:
            v.cam.lookat[:] = (ct.lo.PLANT_X[1], (ct.lo.ROW_Y + ct.lo.AISLE_Y) / 2, 0.22)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.85, 245, -22
            last = [time.time()]

            def frame(phase, ep, last=last):      # bound now: frame only runs inside this iteration's run_pass
                if not v.is_running():
                    raise KeyboardInterrupt
                v.sync()
                wait = 1.0 / ct.FPS - (time.time() - last[0])
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.time()

            try:
                res = ct.run_pass(scn, rng, frame)
            except KeyboardInterrupt:
                return
            print(f"  {res['verdict']}: {res['n_cut']} of the rule's {res['n_rule']} cuts", flush=True)
            if not v.is_running():
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--start", type=int, default=START)
    a = ap.parse_args()
    if a.windowed:
        windowed(a.start)
        return 0
    return 1 if run_checks(a.start) else 0


if __name__ == "__main__":
    sys.exit(main())
