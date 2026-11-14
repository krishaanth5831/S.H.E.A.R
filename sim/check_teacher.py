#!/usr/bin/env python3
"""Step 0 of the post-mortem ladder (§6.6) for SHEAR: the three tests teacher v3 (shear_teacher.py) must pass.
Same tests as B.L.U.S.H sim/check_teacher.py (commit b25ea3b).

    python check_teacher.py repeat --row 910003              # 1. same row twice -> identical commands? PASS/FAIL
    python check_teacher.py push --row 910003                # 2. pushed 1-2 cm mid-approach -> still a clean cut?
    python check_teacher.py push --row 910003 --windowed     #    the same, watched in a MuJoCo window
    python check_teacher.py solve --rows 200                 # 3. solve rate on 200 rows + why the rest fail

A row is three plants from seeds (row, row+1, row+2), pots turned by generate_cuts.best_turns as v1 did, no domain
randomisation (off until the sim reaches 50 %, post-mortem §6.2), no start jitter. Judged by shear_teacher.run: the
target leaf cut with a flush stub (cutter.STUB_OK), no other leaf cut, nothing pressed past cutter.FORCE_LIMIT, no
blade on a truss stalk, back at the start pose.

The push (test 2): at a fixed frame of a leg the 5 arm joints are offset so the gripper moves PUSH_MM (damped least
squares on its Jacobian), joint velocities zeroed; the teacher steers back from there. Legs: to_leaf (free move),
standoff_in (standoff -> pre-cut), slide_in (onto the petiole).

Writes only the solve log ~/.cache/shear_sim/teacher/solve_<stamp>.jsonl. Under the memory cap (CLAUDE.md):
  systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 python check_teacher.py solve --rows 200
"""
import argparse
import collections
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if "--windowed" in sys.argv else "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cutter as ct  # noqa: E402
import generate_cuts as gc  # noqa: E402
import shear_teacher as st  # noqa: E402

LOG_DIR = pathlib.Path.home() / ".cache" / "shear_sim" / "teacher"
ROW_START = 910_000             # apart from v1's rows (900000s) and the trials (700000s)
PUSH_MM = (10.0, 20.0)          # §6.1: "a 1-2 cm push mid-approach still ends in a cut"
PUSH_DIRS = {"down": (0, 0, -1.0), "side": (1.0, 0, 0), "back": (0, -1.0, 0)}   # world: -z, along the row, to the aisle
PUSH_FRAME = {"to_leaf": 12, "standoff_in": 6, "slide_in": 10}   # frames into the leg (design choice)


def scene_for(row):
    seeds = tuple(row + j for j in range(3))
    turns, found = gc.best_turns(seeds, np.random.default_rng(0))   # rng unused there (plan_cut route=False)
    if not any(found):
        return None, found
    return ct.Scene(seeds, turns=turns), found


def displacer(direction, metres):
    def displace(scn):
        m, d = scn.m, scn.d
        b = m.body("arm/gripper").id
        jids = [m.joint("arm/" + j).id for j in ct.cp.ARM5]
        qadr, dadr = [m.jnt_qposadr[j] for j in jids], [m.jnt_dofadr[j] for j in jids]
        p0 = d.xpos[b].copy()
        goal = p0 + np.asarray(direction, float) / np.linalg.norm(direction) * metres
        for _ in range(4):
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, None, b)
            J = jacp[:, dadr]
            d.qpos[qadr] += J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), goal - d.xpos[b])
            mujoco.mj_forward(m, d)
        d.qvel[dadr] = 0.0
        mujoco.mj_forward(m, d)
        return float(np.linalg.norm(d.xpos[b] - p0))
    return displace


def windowed_cb(scn):
    import mujoco.viewer
    v = mujoco.viewer.launch_passive(scn.m, scn.d)
    v.cam.lookat[:] = (ct.lo.PLANT_X[1], (ct.lo.ROW_Y + ct.lo.AISLE_Y) / 2, 0.22)
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.85, 245, -22
    last = [time.time()]

    def cb(phase, ep):
        if v.is_running():
            v.sync()
            time.sleep(max(0.0, 1.0 / ct.FPS - (time.time() - last[0])))
            last[0] = time.time()
    return cb, v


def run_row(row, push=None, windowed=False, scn_cache=None):
    """One row with the teacher, judged. push: (leg, direction name, metres). Returns (result, commands (n, 6))."""
    out = dict(row=row)
    t0 = time.time()
    if scn_cache is not None and row in scn_cache:
        seeds, turns = scn_cache[row]
        scn = ct.Scene(seeds, turns=turns)
    else:
        scn, found = scene_for(row)
        if scn is None:
            return dict(out, solved=False, stage="row", reason="no pot angle gives a clean cut" if sum(
                len(ct.sg.label(ct.sg.make_plant(row + j, bend=True)).cuts) for j in range(3))
                else "the rule cuts nothing", wall_s=round(time.time() - t0, 1)), None
        if scn_cache is not None:
            scn_cache[row] = (scn.seeds, scn.turns)
    try:
        T = st.ShearTeacher(scn).plan()
    except st.PlanError as e:
        return dict(out, solved=False, stage="plan", reason=str(e), wall_s=round(time.time() - t0, 1)), None
    cmds = []
    view = None
    cb_view = None
    if windowed:
        cb_view, view = windowed_cb(scn)

    def cb(phase, ep):
        cmds.append(np.r_[scn.d.ctrl[scn.act], scn.d.ctrl[scn.jaw_act]].copy())
        if cb_view:
            cb_view(phase, ep)
    push_at = None if push is None else (push[0], PUSH_FRAME[push[0]])
    res = st.run(scn, T, cb, push=push_at, displace=None if push is None else displacer(PUSH_DIRS[push[1]], push[2]))
    if view is not None:
        view.close()
    out.update(solved=res["verdict"] == "PASS", stage="run", reason="; ".join(res["why"]) or "ok",
               meta={k: v for k, v in T.meta.items() if k != "tried"}, result=res, wall_s=round(time.time() - t0, 1))
    if push is not None:
        out["push"] = dict(leg=push[0], dir=push[1], asked_mm=round(push[2] * 1000, 1), reached_mm=res["pushed_mm"])
    return out, np.array(cmds)


def cmd_repeat(a):
    cache = {}
    r1, c1 = run_row(a.row, windowed=a.windowed, scn_cache=cache)
    r2, c2 = run_row(a.row, windowed=a.windowed, scn_cache=cache)
    if c1 is None or c2 is None:
        print(f"FAIL: no run to compare (row {a.row}: {r1['stage']}: {r1['reason']})")
        return 1
    same = c1.shape == c2.shape
    diff = float(np.abs(c1 - c2).max()) if same else float("inf")
    ok = same and diff == 0.0
    print(f"row {a.row}: run 1 {len(c1)} frames ({r1['reason']}), run 2 {len(c2)} frames ({r2['reason']}); "
          f"max command difference {diff:g} rad")
    print(f"{'PASS' if ok else 'FAIL'}: the same row gives {'identical' if ok else 'DIFFERENT'} commands")
    return 0 if ok else 1


def cmd_push(a):
    cache = {}
    base, _ = run_row(a.row, scn_cache=cache)
    if not base["solved"]:
        print(f"FAIL: row {a.row} isn't solved without a push ({base['stage']}: {base['reason']}); pick another")
        return 1
    legs = [p for p, _ in base["result"]["phases"]]
    rows = []
    for leg in a.legs:
        if leg not in legs:
            print(f"  {leg}: not in this row's motion (standoff 0 cm?), skipped")
            continue
        for dname in a.dirs:
            for mm in a.mm:
                r, _ = run_row(a.row, push=(leg, dname, mm / 1000), windowed=a.windowed, scn_cache=cache)
                rows.append(r)
                print(f"  push {leg:12s} {dname:5s} asked {mm:4.0f} mm, reached {r['push']['reached_mm']} mm -> "
                      f"{'clean cut' if r['solved'] else 'FAIL: ' + r['reason']}", flush=True)
    n_ok = sum(r["solved"] for r in rows)
    ok = bool(rows) and n_ok == len(rows)
    print(f"{'PASS' if ok else 'FAIL'}: {n_ok} of {len(rows)} pushed runs still made a clean flush cut")
    return 0 if ok else 1


def cmd_solve(a):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"solve_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    why, meta_c = collections.Counter(), collections.Counter()
    n_ok, n_row, n = 0, 0, 0
    t_start = time.time()
    print(f"log {path}", flush=True)
    with path.open("w") as log:
        for k in range(a.rows):
            row = a.row_start + 3 * k
            r, _ = run_row(row, windowed=a.windowed)
            n += 1
            n_row += r["stage"] != "row"
            n_ok += r["solved"]
            if r["solved"]:
                md = r["meta"]
                meta_c[f"plant p{md['plant']}"] += 1
                meta_c[f"standoff {md['standoff_mm']} mm"] += 1
                meta_c[f"to_leaf level {md['to_leaf']['level']} via {md['to_leaf']['via']}"] += 1
                meta_c[f"home level {md['home']['level']} via {md['home']['via']}"] += 1
            else:
                why[f"{r['stage']}: {r['reason'][:90]}"] += 1
            log.write(json.dumps(r, default=float) + "\n")
            log.flush()
            print(f"  row {row}: {'solved' if r['solved'] else 'FAIL ' + r['stage'] + ': ' + r['reason'][:100]}"
                  f"  [{r['wall_s']} s]  {n_ok}/{n}", flush=True)
    mins = (time.time() - t_start) / 60
    lo, hi = wilson(n_ok, n_row)
    summary = dict(summary=True, rows=n, rows_with_a_cuttable_leaf=n_row, solved=n_ok,
                   rate_of_cuttable=round(n_ok / max(n_row, 1), 3), wilson95=[lo, hi], minutes=round(mins, 1),
                   fails=dict(why.most_common()), solved_choices=dict(meta_c.most_common()))
    with path.open("a") as log:
        log.write(json.dumps(summary) + "\n")
    print(f"\n{n} rows, {n_row} with a cleanly cuttable leaf (v1's own filter); teacher solved {n_ok} of those "
          f"({n_ok / max(n_row, 1):.0%}, 95 % interval {lo:.0%}-{hi:.0%}) in {mins:.0f} min")
    print("why the rest failed:")
    for k_, v in why.most_common():
        print(f"  {v:4d}  {k_}")
    print("choices in the solved rows:")
    for k_, v in sorted(meta_c.items()):
        print(f"  {v:4d}  {k_}")
    print(f"log {path}")
    return 0


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(float(c - h), 3), round(float(c + h), 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repeat", help="test 1: same row twice, identical commands")
    p.add_argument("--row", type=int, default=910_003)
    p = sub.add_parser("push", help="test 2: pushed mid-approach, still a clean cut")
    p.add_argument("--row", type=int, default=910_003)
    p.add_argument("--legs", nargs="+", default=list(PUSH_FRAME))
    p.add_argument("--dirs", nargs="+", default=list(PUSH_DIRS), choices=list(PUSH_DIRS))
    p.add_argument("--mm", nargs="+", type=float, default=list(PUSH_MM))
    p = sub.add_parser("solve", help="test 3: solve rate on random rows")
    p.add_argument("--rows", type=int, default=200)
    p.add_argument("--row-start", type=int, default=ROW_START)
    for p in sub.choices.values():
        p.add_argument("--windowed", action="store_true", help="watch it live in a MuJoCo window")
    a = ap.parse_args()
    sys.exit({"repeat": cmd_repeat, "push": cmd_push, "solve": cmd_solve}[a.cmd](a))


if __name__ == "__main__":
    main()
