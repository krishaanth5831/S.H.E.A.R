#!/usr/bin/env python3
"""Step 5 of the SHEAR sim plan: a TRIAL batch of deleafing passes (cutter.run_pass), as videos for K7 to approve.

  python generate_cuts.py --trial 3                     3 passes into ~/.cache/shear_sim/cuts_trial
  python generate_cuts.py --trial 3 --out DIR --overwrite
  python generate_cuts.py --trial 1 --no-video          just plan, play and judge (fast)

One episode = one pass along a new row of three random dwarf plants: from the robot's start pose, every leaf the
rule cuts on p0, then p1, then p2, oldest first, then back to the start pose (K7 2026-11-03). Before the pass each
pot is turned once, to the angle that lets the arm cut the most of that plant's rule cuts (best_turns, K7's pick
2026-11-03), the way a person sets potted plants on the bench; the arm's base stays put (K7 2026-10-31: no sliding
base, SPINE comes last). ⚠️ Without the turn, 0 of 150 plants had a cut the fixed arm could plan (2026-11-01). A row where no cut
can be planned at all is skipped, and why is logged.

Written per episode: ep_KK.mp4, three views (the row from the side | birdseye | claw camera, the phase on top).
The birdseye and claw views carry the labels overlay: each leaf's box and cut point, coloured by what the rule
does with it (red cut, grey keep, blue held back, orange flag), the leaf being cut now in yellow. ⚠️ These are
the SIM'S OWN LABELS (labels.py), what the YOLO-pose model will learn. No model is trained yet (no GPU), so it is
not a detector's output. Plus summary.json and summary.txt.

⚠️ A trial only: at most TRIAL_MAX episodes, no LeRobot dataset, nothing pushed. Run in the lerobot env under the
9 GB cap.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")      # ⚠️ cameras must render through EGL (B.L.U.S.H trap)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cutter as ct  # noqa: E402
import labels as lb  # noqa: E402
import layout as lo  # noqa: E402

TRIAL_MAX = 12
DEFAULT_OUT = pathlib.Path.home() / ".cache" / "shear_sim" / "cuts_trial"
SEED_BASE = 700_000          # row attempt k uses seeds SEED_BASE + 3k .. +2
MAX_ATTEMPTS = 60            # rows tried for the batch
TURN_JITTER_DEG = 10.0       # the turned leaf points along the row within this (design choice)
VIEW_W, VIEW_H = 640, 480
LABEL_EVERY = 2              # overlay labels recomputed every this many frames (30 ms per camera)
DECISION_BGR = {"cut": (40, 40, 230), "flag": (0, 140, 255), "guard": (210, 90, 0), "cap": (210, 90, 0)}
KEEP_BGR, TARGET_BGR = (150, 150, 150), (0, 230, 255)


def pot_turns(seeds, rng):
    """Per plant, the turn (rad) that points its oldest leaf to cut across the arm's reach (square to the line from
    the arm's base to that stem, left or right); 0 if the rule cuts nothing. For p0, in front of the arm, that's
    along the row. ⚠️ Turned along the row instead, p1 and p2 (off to the side) never had a clean flush pose
    (2026-11-03)."""
    turns = []
    for k, s in enumerate(seeds):
        spec = ct.sg.make_plant(s, bend=True)
        sel = ct.sg.label(spec)
        if not sel.cuts:
            turns.append(0.0)
            continue
        i = {nd.id: k for k, nd in enumerate(spec.nodes)}[sel.nodes[sel.cuts[0]].id]
        reach_dir = np.arctan2(lo.ROW_Y - lo.AISLE_Y, lo.PLANT_X[k] - lo.ARM_X)     # base -> stem, seen from above
        side = reach_dir + rng.choice((np.pi / 2, -np.pi / 2))
        turns.append(float(side + np.radians(rng.uniform(-TURN_JITTER_DEG, TURN_JITTER_DEG)) - spec.azimuth[i]))
    return tuple(turns)


TURN_CANDIDATES_DEG = tuple(range(0, 360, 30))     # pot angles tried per plant (K7 2026-11-03: option 1)


def cuttable(scn, rng, k):
    """How many of the rule's cuts on plant k can be cut cleanly as the plant stands (no route: that's planned later)."""
    spec = scn.info["specs"][k]
    sel = ct.sg.label(spec)
    by_id = {nd.id: i for i, nd in enumerate(spec.nodes)}
    return sum(isinstance(ct.plan_cut(scn, rng, plant=k, only=by_id[sel.nodes[j].id], route=False), ct.Plan)
               for j in sel.cuts)


ACROSS_DEG = 25.0             # a leaf within this of square to the arm's reach counts in the quick score (cutpose twist)
TOP_TURNS = 3                 # candidate angles checked in full (settled plant, every cut planned)


def best_turns(seeds, rng):
    """Per plant, the pot angle (rad) that lets the arm cut the most of the rule's cuts on it cleanly, the other pots
    unturned. First a quick score over TURN_CANDIDATES_DEG: how many of the leaves to cut would point across the
    arm's reach (within ACROSS_DEG, left or right). Then the TOP_TURNS best are checked in full on the settled plant.
    ⚠️ Turning each pot only for its oldest leaf (pot_turns) left the plant's other leaves pointing anywhere: 3 of 18
    rule cuts in the first pass trial; a full search on unsettled plants picked angles that no longer worked once
    the leaves had drooped (2026-11-03)."""
    turns, found = [0.0, 0.0, 0.0], []
    for k in range(len(seeds)):
        spec = ct.sg.make_plant(seeds[k], bend=True)
        sel = ct.sg.label(spec)
        if not sel.cuts:
            found.append(0)
            continue
        by_id = {nd.id: i for i, nd in enumerate(spec.nodes)}
        az = np.array([spec.azimuth[by_id[sel.nodes[j].id]] for j in sel.cuts])
        reach_dir = np.arctan2(lo.ROW_Y - lo.AISLE_Y, lo.PLANT_X[k] - lo.ARM_X)
        quick = []
        for deg in TURN_CANDIDATES_DEG:
            off = (az + np.radians(deg) - reach_dir) % np.pi - np.pi / 2       # 0 = square to the reach, either side
            quick.append(int((np.abs(off) <= np.radians(ACROSS_DEG)).sum()))
        best_deg, best_n = 0, -1
        for c in np.argsort(quick)[::-1][:TOP_TURNS]:
            t = [0.0, 0.0, 0.0]
            t[k] = np.radians(TURN_CANDIDATES_DEG[c])
            n = cuttable(ct.Scene(seeds, turns=tuple(t)), rng, k)
            if n > best_n:
                best_deg, best_n = TURN_CANDIDATES_DEG[c], n
        turns[k] = float(np.radians(best_deg))
        found.append(best_n)
    return tuple(turns), found


def draw_overlay(img, objs, ep):
    import cv2
    img = np.ascontiguousarray(img[..., ::-1])
    for o in objs:
        if not o.cls.startswith("leaf") or (o.plant, o.node) in ep.done:
            continue
        target = ep.target == (o.plant, o.node)
        col = TARGET_BGR if target else DECISION_BGR.get(o.decision, KEEP_BGR)
        x0, y0, x1, y1 = o.box
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 3 if target else 1)
        (u, v, s), _ = o.kpts
        if s:
            cv2.circle(img, (int(u), int(v)), 5, col, -1 if s == 2 else 1, cv2.LINE_AA)
        if target:
            cv2.putText(img, "CUTTING", (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (VIEW_W, 22), (255, 255, 255), -1)
    cv2.putText(img, "sim labels (YOLO not trained yet): red cut, grey keep, blue held back, yellow = now",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    return img[..., ::-1]


class Video:
    def __init__(self, scn, path):
        import imageio
        from cameras import SimCameras
        self.scn, self.cams = scn, SimCameras(scn.m)
        self.r = mujoco.Renderer(scn.m, VIEW_H, VIEW_W)
        self.cam = mujoco.MjvCamera()
        # from the aisle, past the arm's shoulder, the whole row of three plants in view (cosmetic)
        self.cam.lookat[:] = (lo.PLANT_X[1], (lo.ROW_Y + lo.AISLE_Y) / 2, 0.22)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.85, 245, -22
        self.w = imageio.get_writer(str(path), fps=ct.FPS, codec="libx264", quality=7, macro_block_size=16)
        self.text, self.k, self.objs = "", 0, {}

    def frame(self, phase, ep):
        import cv2
        d = self.scn.d
        if self.k % LABEL_EVERY == 0:
            for cam in ("birdseye", "claw"):
                self.objs[cam] = lb.label_frame(self.scn.m, d, self.scn.info, self.cams, cam)
        self.k += 1
        self.r.update_scene(d, self.cam)
        tiles = [self.r.render()] + [draw_overlay(self.cams.rgb(d, c), self.objs[c], ep) for c in ("birdseye", "claw")]
        img = np.ascontiguousarray(np.concatenate(tiles, 1))
        bar = np.full((44, img.shape[1], 3), 255, np.uint8)
        now = ("" if ep.target is None else
               f"  leaf {self.scn.info['specs'][ep.target[0]].nodes[ep.target[1]].id[:3]} on plant {ep.target[0] + 1}")
        cv2.putText(bar, f"{phase:10s}{now}   {self.text}   cut so far: {len(ep.done)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        for k, label in enumerate(("side view", "birdseye camera", "claw camera")):
            cv2.putText(img, label, (10 + k * VIEW_W, VIEW_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                        cv2.LINE_AA)
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
    ap.add_argument("--start", type=int, default=0, help="first row attempt (to replay one)")
    a = ap.parse_args()
    if not 0 < a.trial <= TRIAL_MAX:
        sys.exit(f"--trial must be 1..{TRIAL_MAX}: the real batch waits for K7's go on the trial")
    ours = sorted(a.out.glob("ep_*.mp4")) + sorted(a.out.glob("summary.*")) if a.out.exists() else []
    if ours and not a.overwrite:
        sys.exit(f"{a.out} has episodes; pass --overwrite to replace them")
    for f in ours:            # only what this script writes: ⚠️ rmtree of the folder hit files it didn't own
        f.unlink()
    a.out.mkdir(parents=True, exist_ok=True)
    episodes, skipped = [], []
    for k in range(a.start, a.start + MAX_ATTEMPTS):
        if len(episodes) >= a.trial:
            break
        seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
        rng = np.random.default_rng(seeds[0])
        t0 = time.time()
        turns, found = best_turns(seeds, rng)
        if not any(found):
            skipped.append(dict(seeds=seeds, why="no pot angle lets the arm cut any of the rule's leaves cleanly"))
            print(f"row {seeds[0]}: skipped, no pot angle gives a clean cut [{time.time() - t0:.0f} s]", flush=True)
            continue
        scn = ct.Scene(seeds, turns=turns)
        # Is there anything to cut at all? (planned again during the pass, from wherever the arm is then)
        first = [ct.plan_cut(scn, rng, plant=p) for p in range(3)]
        if not any(isinstance(p, ct.Plan) for p in first):
            why = [p.why if not p.tries else p.tries[0][1] for p in first]
            skipped.append(dict(seeds=seeds, why=why))
            print(f"row {seeds[0]}: skipped, nothing plannable on any plant: {why}", flush=True)
            continue
        n = len(episodes)
        vid = None if a.no_video else Video(scn, a.out / f"ep_{n:02d}.mp4")
        if vid:
            vid.text = f"episode {n}"
        res = ct.run_pass(scn, rng, vid.frame if vid else None)
        if vid:
            vid.text = f"episode {n}: {res['n_cut']} of the rule's {res['n_rule']} cuts done, {res['verdict']}"
            ep_end = type("E", (), dict(done=set(), target=None))()
            for _ in range(2 * ct.FPS):                   # two seconds of the result at the end
                vid.frame("result", ep_end)
            vid.close()
        res.update(episode=n, seeds=seeds, pot_turns_deg=[round(float(np.degrees(t)) % 360, 1) for t in turns],
                   cuttable_at_setup=found,
                   seconds=round(time.time() - t0, 1))
        episodes.append(res)
        print(f"episode {n} (row {seeds[0]}): {res['verdict']}, {res['n_cut']}/{res['n_rule']} rule cuts done, "
              f"{len(res['skipped'])} skipped [{res['seconds']} s]", flush=True)
    (a.out / "summary.json").write_text(json.dumps(dict(episodes=episodes, skipped_rows=skipped), indent=1, default=str))
    lines = [f"{len(episodes)} episodes (passes over 3 plants); {len(skipped)} rows skipped (nothing plannable)", ""]
    for e in episodes:
        lines.append(f"ep_{e['episode']:02d}.mp4  {e['verdict']}  {e['n_cut']} of the rule's {e['n_rule']} cuts, "
                     f"{e['seconds_sim']} s of sim")
        for c in e["cuts"]:
            lines.append(f"    cut  plant {c['plant'] + 1} leaf {c['leaf']}: {c['verdict']}  stub {c['stub_mm']} mm  "
                         + "; ".join(c["why"] + c["notes"]))
        for s in e["skipped"]:
            lines.append(f"    skip plant {s['plant'] + 1} leaf {s['leaf']}: {s['why']}")
        lines.append(f"    back to start: {e['back_to_start']['route_points']} route points"
                     + (f", pressed {e['back_to_start']['pressed']}" if e['back_to_start']['pressed'] else ""))
    (a.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
