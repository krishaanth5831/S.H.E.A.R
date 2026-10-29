"""Labels for SHEAR's vision (sim step 4): what a camera frame shows of each plant, in YOLO-pose form.

    objs = label_frame(m, d, info, cams, "birdseye")     one Obj per leaf, truss, scar and stem in view
    lines = yolo_lines(objs)                            "cls cx cy w h  x0 y0 v0  x1 y1 v1" per object, normalised
    truth = truth_json(objs, info)                      the 3D points and the rule's decisions, for checks

What the rule needs from vision (rule/select_leaves.py takes the stem's nodes bottom-up: kind, leaf state and
health, truss stage), and what the cut needs (the cut point and the petiole's direction, cutpose.py):
- **leaf_green / leaf_yellow / leaf_diseased**: box around the leaf; keypoint 0 = the cut point (where the
  petiole meets the stem surface), keypoint 1 = KPT_ALONG further out along the petiole. Two points give the
  petiole's direction in the image; the claw camera (or a second view) gives depth.
- **truss_green / truss_turning / truss_ripe / truss_harvested**: box around the truss; keypoint 0 = where the
  rachis leaves the stem, 1 = KPT_ALONG along it. The node heights order the nodes for the rule.
- **scar**: an old flush cut, so the rule counts the node. Keypoint 0 = the scar; keypoint 1 is never labelled
  (v = 0): a scar has no direction, and a point out in the air would teach the model nothing.
- **stem**: box around the visible stem; keypoint 0 = its bottom (pot rim), 1 = its top. Groups nodes by plant.

Keypoint visibility (YOLO's v): 2 = in frame and seen, 1 = in frame but hidden, 0 = out of frame. "Seen" is a
depth test, not the segmentation id: ⚠️ a cut point on the stem's far side projects onto the stem's front, where
the segmentation says "stem", but it is hidden behind it. A point is seen when the depth render at its pixel is
no nearer than VIS_TOL in front of the point.

An object is labelled when at least MIN_PX of its own pixels are in the frame (segmentation).
"""
import json
from dataclasses import dataclass, field

import numpy as np

import stemgen as sg
from lens import LENSES
from rule.select_leaves import LEAF, PRESENT, SCAR

CLASSES = ("leaf_green", "leaf_yellow", "leaf_diseased",
           "truss_green", "truss_turning", "truss_ripe", "truss_harvested", "scar", "stem")
KPT_ALONG = 0.015        # m, keypoint 1 along the organ (a few px from keypoint 0 in the birdseye; design choice)
# m. The cut point sits on the stem surface; the petiole's own front is up to its radius (~1.8 mm) nearer the
# camera, and a stem in front is a stem diameter (6-9 mm) nearer. 4 mm separates the two. ⚠️ 3 mm marked a cut
# point on its own stem's silhouette hidden (3.3 mm: one pixel is ~1.1 mm at the birdseye's 0.53 m).
VIS_TOL = 0.004
MIN_PX = 15              # an object with fewer visible pixels isn't labelled (guess, unverified)
SCAR_MIN_PX = 3          # a scar is tiny: 1.5 mm proud, ~3 mm across, so a few px in the birdseye
TRUSS_PARTS = ("rachis", "pedicel", "calyx", "fruit")


@dataclass
class Obj:
    cls: str
    plant: int
    node: int                   # node index on the stem (-1 for the stem)
    box: tuple                  # x0, y0, x1, y1 (px, inclusive)
    px: int                     # visible pixels
    kpts: np.ndarray            # (2, 3): u, v, visibility 0/1/2
    pts: np.ndarray             # (2, 3): the keypoints in the world (m)
    decision: str = ""          # leaves: the rule's decision on sim truth (cut / keep / flag / guard / cap / ...)
    extra: dict = field(default_factory=dict)


def _dir(phi, elev):
    return np.array([np.cos(elev) * np.cos(phi), np.cos(elev) * np.sin(phi), np.sin(elev)])


def node_points(spec, xy, where):
    """Per built node of one plant: (class, P0, P1) in the world. `where`: plant_xml's dict for this plant."""
    out = {}
    for i, n in enumerate(spec.nodes):
        if not spec.in_band(i):
            continue
        S = np.array([xy[0], xy[1], spec.z[i]])
        if n.kind == LEAF and n.state == SCAR:
            a = _dir(spec.azimuth[i], 0.0)
            P0 = S + a * spec.stem_r
            out[i] = ("scar", P0, P0 + a * KPT_ALONG)
        elif n.kind == LEAF and n.state == PRESENT:
            w = where[i]
            out[i] = (f"leaf_{n.health}", w["base"], w["base"] + w["a"] * KPT_ALONG)
        elif n.kind != LEAF:
            _, _, down = spec.truss[i]
            a = _dir(spec.azimuth[i], -down)
            P0 = S + a * spec.stem_r
            out[i] = (f"truss_{n.stage}", P0, P0 + a * KPT_ALONG)
    return out


def geom_owner(m):
    """Per geom: (plant, node) for plant geoms (node -1 = the stem), None otherwise; and its seg class."""
    owner, cls = [], []
    for g in range(m.ngeom):
        name = m.geom(g).name or ""
        c = sg.seg_class(name)
        cls.append(c)
        if c is None or c == "pot":
            owner.append(None)
            continue
        node = sg.node_of(name)
        owner.append((sg.plant_of(name), -1 if node is None else node))
    return owner, cls


def label_frame(m, d, info, cams, cam, ids=None, depth=None):
    """Every leaf, truss, scar and stem of every plant in `info` that shows in this camera's frame."""
    lens = LENSES[cam]
    ids = cams.ids(d, cam) if ids is None else ids
    depth = cams.depth(d, cam) if depth is None else depth
    owner, gcls = geom_owner(m)
    # (plant, node) -> pixel mask, via a per-geom key table
    keys = sorted({o for o in owner if o is not None})
    kid = {k: n + 1 for n, k in enumerate(keys)}
    table = np.zeros(m.ngeom + 1, np.int32)          # index ids + 1 (ids -1 = nothing -> 0)
    for g, o in enumerate(owner):
        if o is not None:
            table[g + 1] = kid[o]
    pix = table[ids + 1]
    objs = []
    for k, (xy, spec) in enumerate(zip(info["stems"], info["specs"])):
        where = {i: w for (kk, i), w in info["where"].items() if kk == k}
        sel = sg.label(spec)
        dec = {n.id: sel.decision.get(j, "") for j, n in enumerate(sel.nodes)}
        items = [(i, c, P0, P1) for i, (c, P0, P1) in node_points(spec, xy, where).items()]
        z0 = spec.stem_z[0] + (0.012 if spec.profile.pot else 0.0)      # just above the pot rim
        items.append((-1, "stem", np.array([xy[0], xy[1], z0]), np.array([xy[0], xy[1], spec.stem_z[1]])))
        for i, c, P0, P1 in items:
            key = (k, i)
            mask = pix == kid[key] if key in kid else np.zeros(ids.shape, bool)
            n_px = int(mask.sum())
            if n_px < (SCAR_MIN_PX if c == "scar" else MIN_PX):
                continue
            ys, xs = np.nonzero(mask)
            P = np.array([P0, P1])
            uv = cams.project(d, cam, P)
            zc = cams.cam_depth(d, cam, P)
            kp = np.zeros((2, 3))
            for j in range(1 if c == "scar" else 2):
                u, v = uv[j]
                if not np.isfinite(u) or not (0 <= u < lens.w and 0 <= v < lens.h):
                    continue
                seen = depth[int(v), int(u)]
                kp[j] = (u, v, 2 if (seen == 0 or seen >= zc[j] - VIS_TOL) else 1)
            objs.append(Obj(cls=c, plant=k, node=i, box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                            px=n_px, kpts=kp, pts=P,
                            decision=dec.get(spec.nodes[i].id, "") if i >= 0 else ""))
    return objs


def yolo_lines(objs, w=None, h=None):
    """YOLO-pose rows: class cx cy bw bh, then x y v per keypoint, all normalised to the frame. A keypoint
    out of frame is 0 0 0 (ultralytics' convention)."""
    w = w or LENSES["birdseye"].w
    h = h or LENSES["birdseye"].h
    out = []
    for o in objs:
        x0, y0, x1, y1 = o.box
        row = [CLASSES.index(o.cls), (x0 + x1 + 1) / 2 / w, (y0 + y1 + 1) / 2 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h]
        for u, v, vis in o.kpts:
            row += [u / w, v / h, int(vis)] if vis else [0.0, 0.0, 0]
        out.append(" ".join(str(row[0]) if j == 0 else (str(x) if isinstance(x, int) else f"{x:.6f}")
                            for j, x in enumerate(row)))
    return out


def parse_yolo(lines, w=None, h=None):
    """Back from yolo_lines: [(class name, box px, kpts (2, 3) px)] (for the round-trip check)."""
    w = w or LENSES["birdseye"].w
    h = h or LENSES["birdseye"].h
    out = []
    for ln in lines:
        f = ln.split()
        c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
        kp = np.array(list(map(float, f[5:]))).reshape(2, 3) * (w, h, 1)
        out.append((CLASSES[c], (cx * w - bw * w / 2, cy * h - bh * h / 2, cx * w + bw * w / 2 - 1, cy * h + bh * h / 2 - 1), kp))
    return out


def truth_json(objs, info, cam):
    """Everything a check or a later rule-from-perception test needs, per object."""
    return json.dumps(dict(camera=cam, seeds=[s.seed for s in info["specs"]], profile=info["specs"][0].profile.name,
                           objects=[dict(cls=o.cls, plant=o.plant, node=o.node, box=o.box, px=o.px,
                                         kpts=np.round(o.kpts, 2).tolist(), pts=np.round(o.pts, 5).tolist(),
                                         decision=o.decision) for o in objs]), indent=1)


def data_yaml(root):
    """ultralytics dataset file for a folder with images/ and labels/."""
    names = "\n".join(f"  {k}: {c}" for k, c in enumerate(CLASSES))
    # flip_idx: the two keypoints aren't left/right mirrors of each other, so a flip keeps their order.
    return f"path: {root}\ntrain: images\nval: images\nkpt_shape: [2, 3]\nflip_idx: [0, 1]\nnames:\n{names}\n"
