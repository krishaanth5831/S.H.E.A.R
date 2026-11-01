#!/usr/bin/env python3
"""Step 5 of the SHEAR sim plan: the scripted cutter (cutter.py) and its RRT routes (rrt.py), checked.

  python check_cutter.py                 PASS/FAIL checks on one known plant, exit 1 on a FAIL (~30 s)
  python check_cutter.py --windowed      watch that cut in the MuJoCo window, in real time (it loops)
  python check_cutter.py --start 11      another plant (generate_cuts.py's attempt numbering)

Checks:
- the known plant has a cut the cutter can plan (pot turned, as generate_cuts.py does)
- the RRT route clears stems, pots and the frame by the clearance at every point, checked 4x finer than it was
  planned; the blades stay clear of truss stalks
- when the route bends, the straight joint-space line would have hit something (INFO when the straight line is
  already clear)
- played in the physics, the judge passes it: the right leaf, a flush stub, nothing pressed

Run in the lerobot env under the 9 GB cap. Writes nothing.
"""
import argparse
import sys
import time

import numpy as np

import cutter as ct
import rrt
import stemgen as sg

START = 11                   # generate_cuts.py attempt 11 = seeds 700033..35, a plant with a plannable cut
TURN_JITTER_DEG = 10.0       # as generate_cuts.py


def planned(start):
    """The same plant, pot turn and plan generate_cuts.py makes for this attempt."""
    seeds = tuple(700_000 + 3 * start + j for j in range(3))
    rng = np.random.default_rng(seeds[0])
    spec = sg.make_plant(seeds[0], bend=True)
    sel = sg.label(spec)
    by_id = {nd.id: i for i, nd in enumerate(spec.nodes)}
    for j in sel.cuts[:2]:
        i = by_id[sel.nodes[j].id]
        side = rng.choice((0.0, np.pi))
        turn = side + np.radians(rng.uniform(-TURN_JITTER_DEG, TURN_JITTER_DEG)) - spec.azimuth[i]
        scn = ct.Scene(seeds, turns=(turn, 0.0, 0.0))
        p = ct.plan_cut(scn, rng, only=i)
        if isinstance(p, ct.Plan):
            return scn, p, rng
    return None, None, rng


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
        print("INFO the route is the straight joint-space line: it is already clear" +
              (" of stems, pots and the frame (no route cleared the kept leaves' stalks too)" if p.brushes_leaves else ""))
    else:
        bumps = chk.hits(scn.home) or not chk.segment_free(scn.home, p.to_leaf[-1], soft=not p.brushes_leaves)
        report(bumps, f"the RRT route bends around things ({len(p.to_leaf)} points): the straight joint-space "
                      f"line from HOME to the pre-cut pose would {'hit something' if bumps else 'be clear too'}")
    res = ct.run(scn, p, None, rng)
    report(res["verdict"] == "PASS", f"played in the physics: {res['verdict']}, stub {res['stub_mm']} mm, aim error "
           f"{res['aim_error_mm']} mm, peaks {res['peaks_n']}" + (f" ({'; '.join(res['why'])})" if res["why"] else ""))
    if res["notes"]:
        print("INFO " + "; ".join(res["notes"]))
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return fails


def windowed(start):
    import mujoco.viewer
    while True:
        scn, p, rng = planned(start)
        if p is None:
            print("nothing to plan on this plant")
            return
        print(f"plant {700_000 + 3 * start}, leaf {scn.info['specs'][0].nodes[p.node].id}: watch the route, the slide, "
              f"the cut. Close the window (Ctrl+C on Wayland) to stop.", flush=True)
        with mujoco.viewer.launch_passive(scn.m, scn.d) as v:
            sx, sy = scn.info["stems"][0]
            v.cam.lookat[:] = (sx, sy - 0.08, 0.24)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.6, 215, -18
            last = [time.time()]

            def frame(phase):
                if not v.is_running():
                    raise KeyboardInterrupt
                v.sync()
                wait = 1.0 / ct.FPS - (time.time() - last[0])
                if wait > 0:
                    time.sleep(wait)
                last[0] = time.time()

            try:
                res = ct.run(scn, p, frame, rng)
            except KeyboardInterrupt:
                return
            print(f"  {res['verdict']} stub {res['stub_mm']} mm {res['why']} {res['notes']}", flush=True)
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
