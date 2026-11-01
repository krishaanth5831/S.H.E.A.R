#!/usr/bin/env python3
"""Step 5 of the SHEAR sim plan: a TRIAL batch of scripted cuts (cutter.py), as videos for K7 to approve.

  python generate_cuts.py --trial 6                     6 cut episodes into ~/.cache/shear_sim/cuts_trial
  python generate_cuts.py --trial 6 --out DIR --overwrite
  python generate_cuts.py --trial 1 --no-video          just plan, play and judge (fast)

Each episode: a new row of three random dwarf plants; the rule picks the leaves to cut on p0, the one in front of
the arm. The pot is turned so the oldest of them points along the row (left or right, +-TURN_JITTER_DEG), the
way a person turns a potted plant on the bench; the arm's base stays put (K7 2026-10-31: no sliding base, SPINE
comes last). ⚠️ Without the turn, 0 of 150 plants had a cut the fixed arm could plan (2026-11-01): most rule cuts
point toward or away from the arm, and on a dwarf a truss stalk often sits in the slide's way. The cutter then
plans it (RRT route, straight slide, flush cut), plays it and judges it. If that leaf can't be planned, the next
one is tried; a plant with none is skipped, and why is logged.

Written: ep_KK.mp4 (side view of the plant | birdseye | claw camera, the phase written on top), summary.json
(every episode and every skipped plant, with the judge's numbers), summary.txt (the same, readable).

⚠️ A trial only: at most TRIAL_MAX episodes, no LeRobot dataset, nothing pushed. The real batch is recorded
through a SHEAR robot plugin after K7 approves this one. Run in the lerobot env under the 9 GB cap.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")      # ⚠️ cameras must render through EGL (B.L.U.S.H trap)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cutter as ct  # noqa: E402
import layout as lo  # noqa: E402

TRIAL_MAX = 12
DEFAULT_OUT = pathlib.Path.home() / ".cache" / "shear_sim" / "cuts_trial"
SEED_BASE = 700_000          # episode attempt k uses seeds SEED_BASE + 3k .. +2
MAX_ATTEMPTS = 150           # plants tried for the batch (about 1 in 20 has a cut the fixed arm can plan, 2026-11-01)
TURN_JITTER_DEG = 10.0       # the turned leaf points along the row within this (design choice)
LEAVES_PER_PLANT = 2         # rule cuts tried per plant (each with its own pot turn)
VIEW_W, VIEW_H = 640, 480


class Video:
    def __init__(self, scn, path):
        import imageio
        from cameras import SimCameras
        self.scn, self.cams = scn, SimCameras(scn.m)
        self.r = mujoco.Renderer(scn.m, VIEW_H, VIEW_W)
        self.cam = mujoco.MjvCamera()
        sx, sy = scn.info["stems"][0]
        # beside the aisle, a little above the work band, looking at p0 and the arm from the side (cosmetic)
        self.cam.lookat[:] = (sx, (sy + lo.AISLE_Y) / 2, 0.24)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.55, 215, -18
        self.w = imageio.get_writer(str(path), fps=ct.FPS, codec="libx264", quality=7, macro_block_size=16)
        self.text = ""

    def frame(self, phase):
        import cv2
        d = self.scn.d
        self.r.update_scene(d, self.cam)
        tiles = [self.r.render(), self.cams.rgb(d, "birdseye"), self.cams.rgb(d, "claw")]
        img = np.ascontiguousarray(np.concatenate(tiles, 1))
        bar = np.full((44, img.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, f"{phase:10s} {self.text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        for k, label in enumerate(("side view", "birdseye camera", "claw camera")):
            cv2.putText(img, label, (10 + k * VIEW_W, VIEW_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        self.w.append_data(np.concatenate([bar, img], 0))

    def close(self):
        self.w.close()
        self.r.close()
        self.cams.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True, help=f"episodes (at most {TRIAL_MAX})")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="first plant attempt (to replay one)")
    a = ap.parse_args()
    if not 0 < a.trial <= TRIAL_MAX:
        sys.exit(f"--trial must be 1..{TRIAL_MAX}: the real batch waits for K7's go on the trial")
    if a.out.exists() and any(a.out.iterdir()):
        if not a.overwrite:
            sys.exit(f"{a.out} has files; pass --overwrite to replace them")
        shutil.rmtree(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    episodes, skipped = [], []
    for k in range(a.start, a.start + MAX_ATTEMPTS):
        if len(episodes) >= a.trial:
            break
        seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
        rng = np.random.default_rng(seeds[0])
        t0 = time.time()
        spec = ct.sg.make_plant(seeds[0], bend=True)
        sel = ct.sg.label(spec)
        by_id = {nd.id: i for i, nd in enumerate(spec.nodes)}
        p, tries, turn = None, [], 0.0
        for j in sel.cuts[:LEAVES_PER_PLANT]:
            i = by_id[sel.nodes[j].id]
            side = rng.choice((0.0, np.pi))                   # point it to the arm's right (+x) or left (-x)
            turn = side + np.radians(rng.uniform(-TURN_JITTER_DEG, TURN_JITTER_DEG)) - spec.azimuth[i]
            scn = ct.Scene(seeds, turns=(turn, 0.0, 0.0))
            p = ct.plan_cut(scn, rng, only=i)
            if isinstance(p, ct.Plan):
                break
            tries += p.tries
        if not sel.cuts or not isinstance(p, ct.Plan):
            why = "the rule cuts nothing on this plant" if not sel.cuts else "no cut could be planned"
            skipped.append(dict(seeds=seeds, why=why, leaves=tries))
            print(f"plant {seeds[0]}: skipped, {why} {tries}", flush=True)
            continue
        p.tries = tries + p.tries
        n = len(episodes)
        vid = None if a.no_video else Video(scn, a.out / f"ep_{n:02d}.mp4")
        if vid:
            vid.text = f"episode {n}: cut leaf {scn.info['specs'][0].nodes[p.node].id[:3]} on the plant in front"
        res = ct.run(scn, p, vid.frame if vid else None, rng)
        if vid:
            vid.text = f"episode {n}: {res['verdict']} {'; '.join(res['why'] + res['notes'])}"
            for _ in range(ct.FPS):                       # a second of the verdict at the end
                vid.frame("result")
            vid.close()
        res.update(episode=n, seeds=seeds, pot_turn_deg=round(float(np.degrees(turn)) % 360, 1),
                   skipped_leaves=p.tries, seconds=round(time.time() - t0, 1))
        episodes.append(res)
        print(f"episode {n} (plant {seeds[0]}, leaf {res['target']}): {res['verdict']} {res['why']} {res['notes']} "
              f"stub {res['stub_mm']} mm, aim error {res['aim_error_mm']} mm [{res['seconds']} s]", flush=True)
    (a.out / "summary.json").write_text(json.dumps(dict(episodes=episodes, skipped=skipped), indent=1, default=str))
    lines = [f"{len(episodes)} episodes, {sum(e['verdict'] == 'PASS' for e in episodes)} PASS; "
             f"{len(skipped)} plants skipped (nothing the rule cuts could be planned)", ""]
    for e in episodes:
        lines.append(f"ep_{e['episode']:02d}.mp4  {e['verdict']}  leaf {e['target']}  stub {e['stub_mm']} mm  "
                     f"aim error {e['aim_error_mm']} mm  " + "; ".join(e["why"] + e["notes"]))
    lines.append("")
    for s in skipped:
        lines.append(f"skipped plant {s['seeds'][0]}: {s['why']} {s['leaves']}")
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
