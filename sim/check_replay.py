#!/usr/bin/env python3
"""Pipeline check: play a dataset episode's RECORDED actions back in the sim, open loop, the way a policy's actions
are played (rollout_policy.Sim.act). If the recorded cut doesn't happen, the path from "action" to "the arm moves"
differs between recording and rollout, and no policy trained on this data could ever cut.

  python check_replay.py --log ~/.cache/shear_sim/episodes/shear_sim_scripted_v1_<time>.jsonl            10 episodes
  python check_replay.py --log ... --episodes 0 23 44                                                      these ones
  python check_replay.py --log ... --episodes 0 --windowed                                                  watch it

Per episode: the scene is rebuilt from the row's seeds (pots, domain randomisation), the arm is put where the
recording starts (the scripted pass's jittered start, read from observation.state[0]), then every recorded action
goes to the servos for one frame. Checks, each PASS/FAIL:
- the episode's frame count matches the attempt log (it IS that row)
- tracking: after action i the joints are where the recording's observation i+1 says (max error, norm units)
- the judge (rollout_policy.judge) counts the same cuts the scripted pass made, and nothing pressed too hard

Reads ~/.cache/huggingface/lerobot/<repo> (the local dataset, never the Hub). Writes nothing. CPU only, no cameras.
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if "--windowed" in sys.argv else "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import arm_map  # noqa: E402
import cutter as ct  # noqa: E402
import generate_cuts as gc  # noqa: E402
import rollout_policy as rp  # noqa: E402

REPO = "krish5831/shear_sim_scripted_v1"      # the v3/v3b training set (policy/colab_train_v3.ipynb DATASET)
TRACK_TOL_NORM = 1.0          # guess, unverified: the replay should follow the recording closely (same physics)


def load_episode(root, idx):
    import pandas as pd
    df = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "action", "observation.state"])
                    for p in sorted((root / "data").glob("*/*.parquet"))])
    ep = df[df.episode_index == idx].sort_values("frame_index")
    return np.stack(ep["observation.state"].to_numpy()), np.stack(ep["action"].to_numpy())


class Replay:
    """rollout_policy.Sim without the cameras: same scene, same act()."""

    def __init__(self, seeds, turns, state0):
        self.scn = ct.Scene(seeds, turns=turns, domain_rand=True)
        self.fm = arm_map.FollowerMap()
        s, d = self.scn, self.scn.d
        q0 = self.fm.norm_to_rad(np.asarray(state0, float))
        d.qpos[s.arm_adr] = q0[:5]                     # as cutter._jitter_start: qpos and ctrl both
        d.ctrl[s.act] = q0[:5]
        mujoco.mj_forward(s.m, d)
        self.ep = ct.Episode(self.scn, np.random.default_rng(seeds[0]))

    state_norm = rp.Sim.state_norm
    act = rp.Sim.act


def check(seeds, turns, states, actions, viewer=None):
    sim = Replay(seeds, turns, states[0])
    jaw0 = abs(sim.state_norm()[5] - states[0][5])
    live = viewer(sim) if viewer else None
    err = []
    for i, a in enumerate(actions):
        sim.act(a)
        if i + 1 < len(states):
            err.append(np.abs(sim.state_norm() - states[i + 1]).max())
        if live and not live():
            break
    if live:
        live(close=True)
    return rp.judge(sim), np.array(err), jaw0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=pathlib.Path, required=True, help="generate_episodes' attempt log (.jsonl)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="dataset episode indices (default: the 10 rollout_policy --train-rows picks)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--windowed", action="store_true", help="watch each replay live in a MuJoCo window")
    a = ap.parse_args()
    from lerobot.utils.constants import HF_LEROBOT_HOME
    import json
    root = HF_LEROBOT_HOME / a.repo
    recs = {r["episode_index"]: r for r in map(json.loads, a.log.read_text().splitlines())
            if r.get("kept") is True and "episode_index" in r}
    eps = a.episodes if a.episodes else [e for _, e, _ in rp.train_rows(a.log, 10)]
    viewer = None
    if a.windowed:
        # rollout_policy's live window, reused
        import mujoco.viewer

        def viewer(sim):
            v = mujoco.viewer.launch_passive(sim.scn.m, sim.scn.d)
            v.cam.lookat[:] = (ct.lo.PLANT_X[1], (ct.lo.ROW_Y + ct.lo.AISLE_Y) / 2, 0.22)
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.85, 245, -22
            last = [time.time()]

            def frame(close=False):
                if close or not v.is_running():
                    v.close()
                    return False
                v.sync()
                time.sleep(max(0.0, 1.0 / ct.FPS - (time.time() - last[0])))
                last[0] = time.time()
                return True
            return frame
    n_ok = 0
    for e in eps:
        r = recs[e]
        seeds = tuple(r["seeds"])
        states, actions = load_episode(root, e)
        turns, _ = gc.best_turns(seeds, np.random.default_rng(seeds[0]))
        t0 = time.time()
        res, err, jaw0 = check(seeds, turns, states, actions, viewer)
        same_row = len(actions) == r["frames"]
        tracked = err.max() <= TRACK_TOL_NORM
        cut_ok = res["cut_right"] == r["n_cut"] and res["cut_wrong"] == 0
        ok = same_row and tracked and cut_ok and not res["pressed"]
        n_ok += ok
        print(f"{'PASS' if ok else 'FAIL'}: episode {e} (row {seeds[0]}): {len(actions)} frames "
              f"({'=' if same_row else '!='} log {r['frames']}); tracking max {err.max():.2f} mean {err.mean():.3f} "
              f"norm (frame of max {int(err.argmax()) + 1}), start jaw off {jaw0:.2f}; replay cut "
              f"{res['cut_right']} right + {res['cut_wrong']} wrong (recording: {r['n_cut']}), stubs {res['stubs_mm']} mm, "
              f"pressed {res['pressed'] or 'nothing'} [{time.time() - t0:.0f} s]", flush=True)
    print(f"\n{'PASS' if n_ok == len(eps) else 'FAIL'}: {n_ok} of {len(eps)} recorded episodes replay to the same cuts")
    return 0 if n_ok == len(eps) else 1


if __name__ == "__main__":
    sys.exit(main())
