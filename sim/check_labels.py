#!/usr/bin/env python3
"""Step 4 of the SHEAR sim plan, part 1: the labels the cut-point model will learn from (labels.py), checked.

  python check_labels.py                       PASS/FAIL checks on four fixed rows (ROWS), exit 1 on a FAIL
  python check_labels.py --picture out.png     each plant zoomed in from the birdseye frame, leaves numbered, one
                                               legend line per leaf (draw_plants)
  python check_labels.py --windowed            the row in the viewer, every cut point marked by what the rule
                                               does with that leaf: red = cut, orange = flag, blue = held back
                                               (guard / cap), white = keep. [ and ] switch to the birdseye.

Checks:
- every leaf, truss, scar and stem that shows in the birdseye gets exactly one label, with its true class
- the depth render agrees with the projected points (the visibility test rests on it)
- a cut point behind the stem is marked hidden (the segmentation alone would call it seen); one on a petiole
  crossing the view isn't hidden by its own stem or petiole
- seen keypoints sit in or at their object's box; the YOLO rows read back to the same boxes and keypoints
- INFO: how many of the rule's cut points the birdseye actually sees (the rest is the claw camera's job)

The labels are on birdseye frames. Since the v1 rig (K7 2026-11-06) the dataset records the base camera + claw
instead, but the birdseye and its mast stay in the scene for these label tools (cameras.py).

Run in the lerobot env with MUJOCO_GL=egl, under the 9 GB cap. Writes only the --picture file.
"""
import argparse
import os
import sys
import time

if "--windowed" not in sys.argv:
    # ⚠️ Cameras must render through EGL (B.L.U.S.H trap), but forcing it kills the viewer: only off-window.
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import labels as lb  # noqa: E402
import layout as lo  # noqa: E402
import stemgen as sg  # noqa: E402

ROWS = ((11, 12, 13), (21, 22, 23), (31, 32, 33), (41, 42, 43))
SETTLE_S = 3.0          # check_layout.py's settle
BOX_TOL_PX = 4          # a seen keypoint may sit this far outside its box (the cut point is on the stem surface)
DECISION_RGBA = {"cut": (1, 0.1, 0.1, 1), "flag": (1, 0.6, 0, 1), "guard": (0.2, 0.4, 1, 1), "cap": (0.2, 0.4, 1, 1)}
CLASS_BGR = {"leaf": (60, 200, 60), "truss": (40, 40, 230), "scar": (40, 140, 200), "stem": (200, 200, 200)}


def settled_row(seeds):
    m, d, info = lo.build(seeds)
    for _ in range(round(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return m, d, info


def run_checks():
    from cameras import SimCameras
    fails = 0

    def report(ok, msg):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + msg, flush=True)

    n_obj, missing, dup, wrong_cls, depth_err = 0, [], 0, [], []
    behind, behind_hidden, facing, facing_self = 0, 0, 0, 0
    out_box, round_err, cut_total, cut_seen = 0, 0.0, 0, 0
    for seeds in ROWS:
        m, d, info = settled_row(seeds)
        cams = SimCameras(m)
        ids, depth = cams.ids(d, "birdseye"), cams.depth(d, "birdseye")
        objs = lb.label_frame(m, d, info, cams, "birdseye", ids, depth)
        n_obj += len(objs)
        keys = [(o.plant, o.node) for o in objs]
        dup += len(keys) - len(set(keys))
        # truth: what shows (own pixels) vs what was labelled
        owner, _ = lb.geom_owner(m)
        shown = {}
        for g in np.unique(ids[ids >= 0]):
            if owner[g] is not None:
                shown[owner[g]] = shown.get(owner[g], 0) + int((ids == g).sum())
        for k, spec in enumerate(info["specs"]):
            where_k = {i: w for (kk, i), w in info["where"].items() if kk == k}
            for i, (c, _, _) in lb.node_points(spec, info["stems"][k], where_k).items():
                need = lb.SCAR_MIN_PX if c == "scar" else lb.MIN_PX
                if shown.get((k, i), 0) >= need and (k, i) not in keys:
                    missing.append((seeds[k], i, c))
        for o in objs:
            spec = info["specs"][o.plant]
            n = spec.nodes[o.node] if o.node >= 0 else None
            true = ("stem" if n is None else "scar" if n.state == "scar" and n.kind == "leaf" else
                    f"leaf_{n.health}" if n.kind == "leaf" else f"truss_{n.stage}")
            if true != o.cls:
                wrong_cls.append((o.cls, true))
            x0, y0, x1, y1 = o.box
            for u, v, vis in o.kpts:
                if vis == 2 and not (x0 - BOX_TOL_PX <= u <= x1 + BOX_TOL_PX
                                     and y0 - BOX_TOL_PX <= v <= y1 + BOX_TOL_PX):
                    out_box += 1
            if o.cls == "stem":
                # depth check on the stem's own pixels near its middle: render vs the stem's front surface
                stem_g = [g for g in range(m.ngeom) if owner[g] == (o.plant, -1) and m.geom(g).name.endswith("stem")]
                vs = np.linspace(0.3, 0.7, 9)
                for t in vs:
                    P = o.pts[0] + t * (o.pts[1] - o.pts[0]) - np.array([0, spec.stem_r, 0])    # birdseye looks +y
                    u, v = cams.project(d, "birdseye", P[None])[0]
                    if np.isfinite(u) and ids[int(v), int(u)] in stem_g:
                        depth_err.append(abs(depth[int(v), int(u)] - cams.cam_depth(d, "birdseye", P[None])[0]))
            if o.cls.startswith("leaf"):
                a = info["where"][(o.plant, o.node)]["a"]
                toward = -a[1]                        # +1 = petiole pointing straight at the birdseye (it looks +y)
                if toward < -0.6:
                    behind += 1
                    behind_hidden += o.kpts[0, 2] == 1
                elif abs(toward) < 0.3:
                    # crossing the view: it may be hidden, but only by something in front, not its own stem
                    facing += 1
                    if o.kpts[0, 2] == 1:
                        u, v = o.kpts[0, :2].astype(int)
                        g = ids[v, u]
                        facing_self += g >= 0 and owner[g] in ((o.plant, -1), (o.plant, o.node))
                if o.decision == "cut":
                    cut_seen += o.kpts[0, 2] == 2
        for spec in info["specs"]:
            cut_total += len(sg.label(spec).cuts)
        back = lb.parse_yolo(lb.yolo_lines(objs))
        for o, (_c, box, kp) in zip(objs, back):
            round_err = max(round_err, float(np.abs(np.array(box) - np.array(o.box)).max()),
                            float(np.abs(kp[o.kpts[:, 2] > 0] - o.kpts[o.kpts[:, 2] > 0]).max(initial=0)))
        cams.close()

    report(not missing and not dup and not wrong_cls,
           f"{n_obj} objects labelled in {len(ROWS)} birdseye frames: every leaf, truss, scar and stem that shows "
           f"is there once with its true class ({len(missing)} missing, {dup} twice, {len(wrong_cls)} wrong class)"
           + (f" missing {missing[:4]}" if missing else "") + (f" wrong {wrong_cls[:4]}" if wrong_cls else ""))
    de = np.array(depth_err)
    report(len(de) > 20 and np.median(de) < 0.001, f"depth render vs projected depth on {len(de)} stem pixels: "
           f"median {np.median(de) * 1000:.2f} mm, max {de.max() * 1000:.1f} mm (the visibility test uses it)")
    report(behind and behind_hidden / behind >= 0.9,
           f"cut points behind the stem (petiole pointing away from the birdseye) marked hidden: "
           f"{behind_hidden}/{behind}")
    report(facing and facing_self <= max(1, facing // 10),
           f"cut points on a petiole crossing the view: {facing_self} of {facing} marked hidden behind their own stem "
           f"or petiole (the others marked hidden have another leaf or truss in front)")
    report(out_box == 0, f"seen keypoints inside their box (+{BOX_TOL_PX} px): {out_box} outside")
    report(round_err < 1.0, f"YOLO rows read back: worst error {round_err:.3f} px")
    print(f"INFO the birdseye sees {cut_seen} of the rule's {cut_total} cut points (the rest are hidden: the claw "
          f"camera has to find them)")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return fails


def draw(img, objs, scale=2):
    """The whole frame: every box, and the leaves' cut points coloured by the rule. Crowded; draw_plants is the
    readable one."""
    import cv2
    img = cv2.resize(np.ascontiguousarray(img[..., ::-1]), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for o in objs:
        col = CLASS_BGR[o.cls.split("_")[0]]
        x0, y0, x1, y1 = (np.array(o.box) * scale).astype(int)
        cv2.rectangle(img, (x0, y0), (x1 + scale - 1, y1 + scale - 1), col, 1)
        dec_bgr = {"cut": (30, 30, 255), "flag": (0, 150, 255), "guard": (255, 100, 50), "cap": (255, 100, 50)}
        if not o.cls.startswith("leaf"):
            continue          # only leaves get dots: a truss's dot read as a cut point (K7, 2026-10-31)
        kcol = dec_bgr.get(o.decision, (255, 255, 255))
        (u0, v0, s0), (u1, v1, s1) = o.kpts
        if s0 and s1:
            cv2.line(img, (int(u0 * scale), int(v0 * scale)), (int(u1 * scale), int(v1 * scale)), kcol, 1, cv2.LINE_AA)
        for u, v, s in o.kpts[:1]:
            if s:
                cv2.circle(img, (int(u * scale), int(v * scale)), 4, kcol, -1 if s == 2 else 1, cv2.LINE_AA)
        if o.cls.startswith("leaf"):
            cv2.putText(img, o.decision, (x0, max(y0 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, kcol, 1, cv2.LINE_AA)
    return img[..., ::-1]


DECISION_TEXT = {"cut": ("CUT", (0, 0, 230)), "flag": ("flag for a person", (0, 140, 255)),
                 "guard": ("held back: plant needs its leaves", (210, 90, 0)),
                 "cap": ("held back: max 3 per pass", (210, 90, 0)), "keep": ("keep", (120, 120, 120))}
PANEL_H = 760           # px, each plant's zoomed panel
LEGEND_W = 420


def draw_plants(img, objs, n_plants=3):
    """One zoomed panel per plant, leaves only: each leaf numbered from the bottom, its box, its cut point (filled
    = the camera sees it, ring = hidden behind something), the petiole's direction, and a legend line per number."""
    import cv2
    base = np.ascontiguousarray(img[..., ::-1])
    panels = []
    for k in range(n_plants):
        leaves = sorted((o for o in objs if o.plant == k and o.cls.startswith("leaf")), key=lambda o: o.node)
        stem = [o for o in objs if o.plant == k and o.cls == "stem"]
        boxes = np.array([o.box for o in leaves + stem]) if leaves or stem else None
        if boxes is None:
            continue
        x0, y0 = boxes[:, :2].min(0) - 12
        x1, y1 = boxes[:, 2:].max(0) + 12
        x0, y0 = max(int(x0), 0), max(int(y0), 0)
        x1, y1 = min(int(x1), base.shape[1] - 1), min(int(y1), base.shape[0] - 1)
        sc_ = PANEL_H / (y1 - y0 + 1)
        crop = cv2.resize(base[y0:y1 + 1, x0:x1 + 1], None, fx=sc_, fy=sc_, interpolation=cv2.INTER_LINEAR)
        crop = (0.55 * crop + 0.45 * 255).astype(np.uint8)          # faded, so the marks stand out
        canvas = np.full((PANEL_H + 50, crop.shape[1] + LEGEND_W, 3), 255, np.uint8)
        canvas[50:50 + crop.shape[0], :crop.shape[1]] = crop
        # image px -> this panel's canvas px (crop corner and scale bound now; P is only used in this panel)
        P = lambda u, v, x0=x0, y0=y0, sc_=sc_: (int((u - x0) * sc_), int((v - y0) * sc_) + 50)
        n_cut = sum(o.decision == "cut" for o in leaves)
        cv2.putText(canvas, f"plant {k + 1}: {len(leaves)} leaves in view, {n_cut} to cut", (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        rows, last_y = [], 0
        for n, o in enumerate(leaves, 1):
            word, col = DECISION_TEXT.get(o.decision, ("keep", (120, 120, 120)))
            bx0, by0 = P(o.box[0], o.box[1])
            bx1, by1 = P(o.box[2] + 1, o.box[3] + 1)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), col, 2 if o.decision == "cut" else 1)
            (u0, v0, s0), (u1, v1, s1) = o.kpts
            if not s0:
                continue
            p0 = P(u0, v0)
            if s0 == 2 and s1:
                p1 = np.array(P(u1, v1))
                tip = tuple((np.array(p0) + 2.5 * (p1 - np.array(p0))).astype(int))   # petiole direction, drawn long
                cv2.arrowedLine(canvas, p0, tip, col, 3, cv2.LINE_AA, tipLength=0.3)
            if s0 == 2:
                cv2.circle(canvas, p0, 8, col, -1, cv2.LINE_AA)
                cv2.circle(canvas, p0, 8, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, p0, 8, col, 2, cv2.LINE_AA)
            cv2.putText(canvas, str(n), (p0[0] - 5, p0[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                        cv2.LINE_AA)
            rows.append((p0, n, word + ("" if s0 == 2 else " (cut point hidden)"), col))
        # legend: one line per leaf, at the height of its cut point, pushed apart so they don't overlap
        lx = crop.shape[1] + 30
        for p0, n, text, col in sorted(rows, key=lambda r: r[0][1]):
            y = max(p0[1], last_y + 26, 70)
            last_y = y
            cv2.line(canvas, (p0[0] + 9, p0[1]), (lx - 6, y - 5), col, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{n}  {text}", (lx, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col,
                        1 if col == (120, 120, 120) else 2, cv2.LINE_AA)
        panels.append(canvas)
    sep = np.full((PANEL_H + 50, 6, 3), 200, np.uint8)
    body = np.concatenate([x for p in panels for x in (p, sep)][:-1], 1)
    head = np.full((70, body.shape[1], 3), 255, np.uint8)
    for i, line in enumerate(("Each plant zoomed in, leaves only (numbered from the bottom). Dot = where the leaf "
                              "stalk meets the stem = the cut point.",
                              "Filled dot = the overview camera sees it; ring = hidden behind fruit, stem or another "
                              "leaf. Arrow = the stalk's direction.")):
        cv2.putText(head, line, (10, 28 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return np.concatenate([head, body], 0)[..., ::-1]


def picture(path, rows=ROWS[:2]):
    from PIL import Image

    from cameras import SimCameras
    tiles = []
    for seeds in rows:
        m, d, info = settled_row(seeds)
        cams = SimCameras(m)
        objs = lb.label_frame(m, d, info, cams, "birdseye")
        tiles.append(draw_plants(cams.rgb(d, "birdseye"), objs))
        cams.close()
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 10), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
    Image.fromarray(np.concatenate(tiles, 0)).save(path)
    print(f"wrote {path}: each plant zoomed, leaves numbered, a legend line per leaf")


def windowed(seeds=ROWS[0]):
    import mujoco.viewer
    m, d, info = lo.build(seeds)
    marks = []
    for k, spec in enumerate(info["specs"]):
        sel = sg.label(spec)
        dec = {n.id: sel.decision.get(j, "") for j, n in enumerate(sel.nodes)}
        for (kk, i), w in info["where"].items():
            if kk == k:
                marks.append((w["base"], DECISION_RGBA.get(dec.get(spec.nodes[i].id, ""), (1, 1, 1, 1))))
    print(f"{len(marks)} cut points: red = the rule cuts it, orange = flagged for a person, blue = held back, "
          "white = keep.\nThe dots stay where the leaves were built; the leaves settle a few mm.")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (lo.PLANT_X[1], lo.ROW_Y, 0.25)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.8, -90, -10
        while v.is_running():
            t = time.time()
            mujoco.mj_step(m, d)
            with v.lock():
                v.user_scn.ngeom = 0
                for p, rgba in marks:
                    if v.user_scn.ngeom < v.user_scn.maxgeom:
                        mujoco.mjv_initGeom(v.user_scn.geoms[v.user_scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.004, 0, 0]), np.asarray(p, float), np.eye(3).ravel(),
                                            np.asarray(rgba, np.float32))
                        v.user_scn.ngeom += 1
            v.sync()
            time.sleep(max(0.0, m.opt.timestep - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picture", metavar="PNG")
    ap.add_argument("--windowed", action="store_true")
    a = ap.parse_args()
    if a.windowed:
        windowed()
        return 0
    if a.picture:
        picture(a.picture)
        return 0
    return run_checks()


if __name__ == "__main__":
    sys.exit(main())
