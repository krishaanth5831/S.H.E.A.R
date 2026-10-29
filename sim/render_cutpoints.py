#!/usr/bin/env python3
"""Step 4 of the SHEAR sim plan, part 2: a PILOT of labelled birdseye frames for the cut-point model.

  python render_cutpoints.py --pilot 6                     6 frames into ~/.cache/shear_sim/cutpoints_pilot
  python render_cutpoints.py --pilot 6 --out DIR --overwrite

Each frame is a row of three random dwarf plants (layout.py), settled, seen by the birdseye through the lens
model. Written per frame k: images/k.png, labels/k.txt (YOLO-pose rows, labels.py), truth/k.json (3D keypoints
and the rule's decisions). Plus data.yaml for ultralytics and preview/k.png with the labels drawn on.

⚠️ A pilot only: at most PILOT_MAX frames, no training, nothing pushed (K7 approves the pilot and the dataset
size first, SHEAR Index 2026-10-28: "no dataset yet"). No domain randomisation yet (light, colours, camera
pose): that comes with the real set. Claw-camera frames come with step 5, from the cutter's own approach poses.

Run in the lerobot env under the 9 GB cap. Writes only into --out, and refuses a non-empty --out without --overwrite.
"""
import argparse
import os
import pathlib
import shutil
import sys

os.environ.setdefault("MUJOCO_GL", "egl")      # ⚠️ cameras must render through EGL (B.L.U.S.H trap)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import check_labels as cl  # noqa: E402
import labels as lb  # noqa: E402
from cameras import SimCameras  # noqa: E402

PILOT_MAX = 24
DEFAULT_OUT = pathlib.Path.home() / ".cache" / "shear_sim" / "cutpoints_pilot"
SEED_BASE = 500_000          # frame k uses seeds SEED_BASE + 3k .. +2 (apart from every check's seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, required=True, help=f"frames to write (at most {PILOT_MAX})")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true", help="empty --out first if it has files")
    a = ap.parse_args()
    if not 0 < a.pilot <= PILOT_MAX:
        sys.exit(f"--pilot must be 1..{PILOT_MAX}: a bigger set waits for K7's go on the pilot")
    if a.out.exists() and any(a.out.iterdir()):
        if not a.overwrite:
            sys.exit(f"{a.out} has files; pass --overwrite to replace them")
        shutil.rmtree(a.out)
    for sub in ("images", "labels", "truth", "preview"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    (a.out / "data.yaml").write_text(lb.data_yaml(a.out))
    counts = {}
    for k in range(a.pilot):
        seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
        m, d, info = cl.settled_row(seeds)
        cams = SimCameras(m)
        rgb = cams.rgb(d, "birdseye")
        objs = lb.label_frame(m, d, info, cams, "birdseye")
        cams.close()
        Image.fromarray(rgb).save(a.out / "images" / f"{k:05d}.png")
        (a.out / "labels" / f"{k:05d}.txt").write_text("\n".join(lb.yolo_lines(objs)) + "\n")
        (a.out / "truth" / f"{k:05d}.json").write_text(lb.truth_json(objs, info, "birdseye"))
        Image.fromarray(cl.draw(rgb, objs)).save(a.out / "preview" / f"{k:05d}.png")
        for o in objs:
            counts[o.cls] = counts.get(o.cls, 0) + 1
        cuts = [o for o in objs if o.decision == "cut"]
        print(f"frame {k}: seeds {seeds}, {len(objs)} objects, {len(cuts)} rule cuts "
              f"({sum(o.kpts[0, 2] == 2 for o in cuts)} cut points seen)", flush=True)
    print(f"\nwrote {a.pilot} frames to {a.out}: " + ", ".join(f"{c} {counts.get(c, 0)}" for c in lb.CLASSES))
    print(f"look at {a.out / 'preview'} (labels drawn on, 2x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
