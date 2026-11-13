#!/usr/bin/env python3
"""Step-2 check for teacher-v3 datasets recorded in parts (generate_episodes.py --teacher, one repo per part): play
episodes' RECORDED actions back in the sim, open loop, the way a policy's actions are played, and judge them as
shear_teacher.run judges the teacher (eval_onecut.judge: the target leaf cut, flush stub, nothing else cut or pressed,
arm back at the start). check_replay.py does the same for the scripted v1 data (domain randomisation, n_cut cuts).

  python replay_teacher.py ~/.cache/shear_sim/episodes/shear_sim_teacher_v3_p*.jsonl --n 10
  python replay_teacher.py ~/.cache/shear_sim/episodes/shear_sim_teacher_v3_p*.jsonl --n 2 --windowed     watch it

- clean episodes: the actions ARE what the servos got, so the replay must track the recording (max state difference
  under --tol, norm units) and pass.
- noisy episodes (DART): the servos got label + noise, the dataset holds the clean label, so the replay can't track the
  recording; it's judged on the outcome only ("labels only"), and it doesn't decide PASS/FAIL: the teacher is closed
  loop, so its labels answer the noisy states they were made in, and played open loop from clean states they can land
  elsewhere (first try 2026-11-13: 1 of 2 cut at a 17.4 mm stub, recorded 2.52 mm).
- holds (no cleanly cuttable leaf, cue zero): the arm must stay home and cut nothing.

Reads each part's local dataset ~/.cache/huggingface/lerobot/<user>/<log name minus its time stamp>. Writes nothing.
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if "--windowed" in sys.argv else "egl")

import numpy as np  # noqa: E402

import check_replay as cr  # noqa: E402
import cutter as ct  # noqa: E402
import eval_onecut as eo  # noqa: E402
import generate_cuts as gc  # noqa: E402
import shear_teacher as st  # noqa: E402

USER = "krish5831"
STAMP = re.compile(r"_\d{8}_\d{6}$")


def kept_rows(logs):
    """(repo, record) for every kept episode, in log order (= the part's episode order)."""
    out = []
    for p in logs:
        repo = f"{USER}/{STAMP.sub('', p.stem)}"
        for line in p.read_text().splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("kept") is True and "episode_index" in r:
                out.append((repo, r))
    return out


def replay(rec, states, actions, windowed=False):
    seeds = tuple(rec["seeds"])
    turns, _ = gc.best_turns(seeds, np.random.default_rng(0))      # teacher_batch's set-up
    sim = cr.Replay.__new__(cr.Replay)
    sim.scn = ct.Scene(seeds, turns=turns, domain_rand=bool(rec.get("domain_rand")))
    sim.fm = cr.arm_map.FollowerMap()
    s, d = sim.scn, sim.scn.d
    q0 = sim.fm.norm_to_rad(np.asarray(states[0], float))
    d.qpos[s.arm_adr] = q0[:5]
    d.ctrl[s.act] = q0[:5]
    cr.mujoco.mj_forward(s.m, d)
    sim.ep = ct.Episode(s, None)
    key = tgt = None
    if not rec.get("hold"):
        T = st.ShearTeacher(s).plan()                                 # deterministic: the recording's target
        if T.meta.get("leaf") != rec["plan"]["leaf"] or T.meta.get("plant") != rec["plan"]["plant"]:
            return None, None, f"plans {T.meta.get('plant')}/{T.meta.get('leaf')}, recorded " \
                               f"{rec['plan']['plant']}/{rec['plan']['leaf']}"
        p = T.plan_cut
        key = (p.plant, p.node)
        sim.ep.target = key
        sim.ep.mon.reset(p.prefix)
        tgt = sim.ep.cutters[key]
    err, v = [], None
    if windowed:
        import mujoco.viewer
        v = mujoco.viewer.launch_passive(s.m, d)
        v.cam.lookat[:] = (ct.lo.PLANT_X[1], (ct.lo.ROW_Y + ct.lo.AISLE_Y) / 2, 0.22)   # check_replay's view
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.85, 245, -22
    for i, a in enumerate(actions):
        t = time.time()
        sim.act(a)
        if i + 1 < len(states):
            err.append(np.abs(sim.state_norm() - states[i + 1]).max())
        if v is not None:
            if not v.is_running():
                break
            v.sync()
            time.sleep(max(0.0, 1.0 / ct.FPS - (time.time() - t)))
    if v is not None:
        v.close()
    if rec.get("hold"):
        cut = [k for k, c in sim.ep.cutters.items() if c.cut_t is not None]
        why = ([f"cut {cut}"] if cut else []) + [f"pressed the {k} with {v:.1f} N" for k, v in sim.ep.mon.peak.items()
                                                 if k in ct.FORCE_LIMIT and v > ct.FORCE_LIMIT[k]]
        off = float(np.abs(sim.ep.q_now() - s.start).max())
        if off > st.END_START_TOL:
            why.append(f"arm {np.degrees(off):.1f} deg off the start pose")
        stub = None
    else:
        why = eo.judge(sim.ep, s, key, tgt)
        stub = None if tgt.stub is None else round(tgt.stub * 1000, 2)
    return np.array(err), stub, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=pathlib.Path, help="attempt logs (~/.cache/shear_sim/episodes/*.jsonl)")
    ap.add_argument("--n", type=int, default=10, help="half clean, half noisy, one hold in each half if any")
    ap.add_argument("--tol", type=float, default=cr.TRACK_TOL_NORM, help="clean episodes: max state diff, norm units")
    ap.add_argument("--seed", type=int, default=0, help="which episodes get picked")
    ap.add_argument("--windowed", action="store_true", help="watch each replay live in a MuJoCo window")
    a = ap.parse_args()
    from lerobot.utils.constants import HF_LEROBOT_HOME
    rows = kept_rows(a.logs)
    rng = np.random.default_rng(a.seed)
    pick = []
    for noisy in (False, True):
        pool = [x for x in rows if bool(x[1].get("noise")) == noisy]
        holds = [x for x in pool if x[1].get("hold")]
        cuts = [x for x in pool if not x[1].get("hold")]
        k = a.n // 2 if not noisy else a.n - a.n // 2
        n_h = min(1, len(holds))
        pick += [holds[i] for i in rng.choice(len(holds), n_h, replace=False)]
        pick += [cuts[i] for i in rng.choice(len(cuts), min(k - n_h, len(cuts)), replace=False)]
    n_ok = {False: [0, 0], True: [0, 0]}
    for repo, rec in pick:
        noisy = bool(rec.get("noise"))
        states, actions = cr.load_episode(HF_LEROBOT_HOME / repo, rec["episode_index"])
        t0 = time.time()
        err, stub, why = replay(rec, states, actions, a.windowed)
        if err is None:
            ok, msg = False, why
        else:
            same = len(actions) == rec["frames"]
            tracked = noisy or err.max() <= a.tol
            ok = same and tracked and not why
            msg = (f"{len(actions)} frames ({'=' if same else '!='} log {rec['frames']}), state diff max "
                   f"{err.max():.2f}{' (not judged)' if noisy else ''}, "
                   f"{'hold' if rec.get('hold') else f'stub {stub} mm (recorded {rec['result']['stub_mm']})'}; "
                   f"{'; '.join(why) or 'clean'}")
        n_ok[noisy][0] += ok
        n_ok[noisy][1] += 1
        print(f"{'✅' if ok else '❌'} {repo} ep {rec['episode_index']} (row {rec['row']}, "
              f"{'noisy' if noisy else 'clean'}{', hold' if rec.get('hold') else ''}): {msg} "
              f"[{time.time() - t0:.0f} s]", flush=True)
    (c, nc), (z, nz) = n_ok[False], n_ok[True]
    print(f"\n{'PASS' if c == nc else 'FAIL'}: clean {c}/{nc} reproduce and pass; "
          f"noisy (labels only, not part of the verdict) {z}/{nz} still pass")
    return 0 if c == nc else 1


if __name__ == "__main__":
    sys.exit(main())
