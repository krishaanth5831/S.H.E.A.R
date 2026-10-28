#!/usr/bin/env python3
"""Where SHEAR's cameras and arm sit (layout.py), checked and pictured.

  python check_layout.py --picture out.png     one image: overview with each camera's view cone, the
                                               birdseye frame, the claw frame
  python check_layout.py --windowed            the scene in a window (the cameras are in the viewer's
                                               camera list: press [ and ] to switch to them)
  python check_layout.py                       PASS/FAIL checks, exit 1 on a FAIL

Checks:
- the birdseye sits on its mast, which never shows in its own frame
- the birdseye sees every built leaf's cut point and every stem's work band on all plants in the row
- the arm's base stands outside the birdseye frame (the camera looks at the plants, not the arm)
- the plants settle inside the greenhouse (leaves may rest on the frame, like real ones on wires)
- the claw camera looks along the jaw: the fingertips are near its image centre
- the arm's base is square to the row, right in front of its plant, and IK reaches the ready pose

Run in the lerobot env under the 9 GB cap. Writes only the --picture file. Needs ./fetch_so101.sh once.
"""
import argparse
import sys
import time

import mujoco
import numpy as np

import layout as lo
import stemgen as sg
from cameras import CAM_IN_MODEL, SimCameras
from lens import LENSES

SEEDS = (11, 12, 13)
SETTLE_S = 3.0         # ⚠️ 2 s left the bigger tomato leaflets ringing at 0.067 rad/s (p0_n08, arm or no arm)
SETTLED_HINGE = 0.05    # rad/s, as check_plant.py
MARGIN_PX = 8           # a point this close to the frame edge counts as cut off
CLAW_CENTRE_FRAC = 0.35  # fingertips within this share of the half-diagonal from the claw image centre


def in_frame(uv, lens, margin=MARGIN_PX):
    return (~np.isnan(uv).any(1) & (uv[:, 0] >= margin) & (uv[:, 0] < lens.w - margin)
            & (uv[:, 1] >= margin) & (uv[:, 1] < lens.h - margin))


def settle(m, d):
    for _ in range(round(SETTLE_S / m.opt.timestep)):
        mujoco.mj_step(m, d)


def fingertips(m, d):
    jg = lo.sc.jaw_geometry()
    g = m.body("arm/gripper").id
    return d.xpos[g] + d.xmat[g].reshape(3, 3) @ np.array([jg["x_meet"], jg["y_blade"], jg["tip_z"]])


def run_checks():
    fails = 0

    def report(ok, msg):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + msg, flush=True)

    m, d, info = lo.build(SEEDS)
    settle(m, d)
    cams = SimCameras(m)
    be = LENSES["birdseye"]
    print(f"birdseye at {lo.BIRDSEYE_POS}, tilt {lo.BIRDSEYE_TILT_DEG:.0f} deg, "
          f"{lo.ROW_Y - lo.BIRDSEYE_POS[1]:.2f} m from the row, lens {be.fov_deg['h']:.0f} x {be.fov_deg['v']:.0f} deg "
          f"({be.source})")
    pts = np.array([w["base"] for w in info["where"].values()])
    uv = cams.project(d, "birdseye", pts)
    inside = in_frame(uv, be)
    report(inside.all(), f"birdseye sees all {len(pts)} built leaves' cut points on {len(SEEDS)} plants "
           f"({int(inside.sum())} inside the frame)")
    zs = [[sp.z[i] for i in range(len(sp.nodes)) if sp.in_band(i)] for sp in info["specs"]]
    band = [(x, y, z) for (x, y), zz in zip(info["stems"], zs) for z in (min(zz), max(zz))]
    ok_band = in_frame(cams.project(d, "birdseye", np.array(band)), be)
    report(ok_band.all(), f"birdseye sees every plant from its lowest to its highest node, z {min(map(min, zs)):.2f}-{max(map(max, zs)):.2f} m "
           f"({int(ok_band.sum())}/{len(band)} band ends in frame)")
    base = info["arm_base"]
    arm_base_pts = np.array([base + (dx, dy, dz) for dx in (-0.05, 0.05) for dy in (-0.05, 0.05) for dz in (0.0, 0.10)])
    out = ~in_frame(cams.project(d, "birdseye", arm_base_pts), be, margin=0)
    report(out.all(), f"the arm's base ({base[0]:.2f}, {base[1]:.2f}) stands outside the birdseye frame "
           f"({int(out.sum())}/{len(out)} corners of a 10 cm box around it outside)")
    ids = cams.ids(d, "birdseye")
    mast = [g for g in range(m.ngeom) if m.geom(g).name.startswith("gh_mast_")]
    seen = int(np.isin(ids, mast).sum())
    lens_off = np.linalg.norm(d.cam_xpos[m.camera("birdseye").id] - np.array(lo.BIRDSEYE_POS))
    report(seen == 0 and lens_off < 1e-9 and len(mast) == 5, f"the birdseye sits on its mast at the lens front "
           f"({len(mast)} mast parts, {seen} of their pixels in the birdseye frame)")
    tip = fingertips(m, d)
    uv_t = cams.project(d, "claw", tip[None])[0]
    cl = LENSES["claw"]
    r = np.hypot(uv_t[0] - cl.w / 2, uv_t[1] - cl.h / 2) / np.hypot(cl.w / 2, cl.h / 2)
    report(np.isfinite(r) and r < CLAW_CENTRE_FRAC, f"claw camera looks along the jaw: fingertips at pixel "
           f"({uv_t[0]:.0f}, {uv_t[1]:.0f}), {r:.2f} of the half-diagonal from the centre (limit {CLAW_CENTRE_FRAC})")
    names = [m.geom(g).name for g in range(m.ngeom)]
    structure = {g for g, n in enumerate(names) if n == "floor" or n.startswith("gh_")}
    plant_g = {g for g, n in enumerate(names) if sg.seg_class(n) is not None}
    touch = sorted({names[c.geom1] if c.geom1 in plant_g else names[c.geom2] for c in d.contact[:d.ncon]
                    if (c.geom1 in plant_g and c.geom2 in structure) or (c.geom2 in plant_g and c.geom1 in structure)})
    # Leaves may rest on the frame, like real ones on wires and posts; they must settle there, not fight it.
    hinge = [m.jnt_dofadr[j] for j in range(m.njnt) if "midrib" in m.joint(j).name]
    w = max(abs(d.qvel[a]) for a in hinge)
    report(w < SETTLED_HINGE, f"the plants settle inside the greenhouse: midrib hinges {w:.4f} rad/s after "
           f"{SETTLE_S:.0f} s (limit {SETTLED_HINGE})")
    print(f"INFO {len(touch)} plant geoms rest on the greenhouse frame" + (f": {touch[:6]}" if touch else ""))
    fwd = np.array([np.cos(info["arm_yaw"]), np.sin(info["arm_yaw"])])
    row_normal = np.array([0.0, 1.0])                    # row R1 runs along x; the arm should point +y
    ang = np.degrees(np.arccos(np.clip(fwd @ row_normal, -1, 1)))
    sx, sy = info["stems"][info["target_plant"]]
    report(ang < 0.5 and abs(base[0] - sx) < 1e-6, f"the arm's base faces the row squarely ({ang:.1f} deg off its "
           f"normal) and stands right in front of plant p{info['target_plant']} (x {base[0]:.2f} = stem x {sx:.2f})")
    report(info["ready_err"] < 0.002, f"ready pose: fingertips {lo.READY_STANDOFF * 100:.0f} cm in front of the stem at "
           f"{lo.READY_Z:.2f} m, IK error {info['ready_err'] * 1000:.1f} mm "
           f"({', '.join(f'{k} {np.degrees(v):.0f}' for k, v in info['ready'].items())} deg)")
    cams.close()
    print("\nALL PASS" if not fails else f"\n{fails} FAIL")
    return 1 if fails else 0


def _frustum(m, d, cam, length):
    """World points: camera position and the four frame-corner rays out to `length`."""
    lens = LENSES[cam]
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_IN_MODEL[cam])
    R, c = d.cam_xmat[cid].reshape(3, 3), d.cam_xpos[cid]
    out = []
    for u, v in ((0, 0), (lens.w, 0), (lens.w, lens.h), (0, lens.h)):
        a, b = (u - lens.K[0, 2]) / lens.K[0, 0], (v - lens.K[1, 2]) / lens.K[1, 1]
        th = float(lens.theta(np.hypot(a, b)))
        n = np.hypot(a, b)
        ray_cv = np.array([np.sin(th) * a / n, np.sin(th) * b / n, np.cos(th)])   # OpenCV: x right, y down, z fwd
        ray = R @ np.array([ray_cv[0], -ray_cv[1], -ray_cv[2]])                    # MuJoCo camera: y up, looks -z
        out.append(c + ray * length)
    return c, out


def _add_line(scn, a, b, rgba, width=3.0):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, width, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def _add_dot(scn, c, rgba, r=0.008):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(c, float),
                        np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _label(img, text):
    import cv2
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 12, 28), (255, 255, 255), -1)
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def picture(path):
    from PIL import Image
    m, d, info = lo.build(SEEDS)
    settle(m, d)
    cams = SimCameras(m)
    be, cl = cams.rgb(d, "birdseye"), cams.rgb(d, "claw")
    r = mujoco.Renderer(m, 480, 640)
    views = []
    import greenhouse as gh
    for az, el, look, dist, roof in ((20, -18, (0.30, 0.18, 0.24), 1.25, False), (90, -88, (0.37, 0.12, 0.0), 1.35, False)):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = look, dist, az, el
        opt = mujoco.MjvOption()
        opt.geomgroup[gh.ROOF_GROUP] = roof
        r.update_scene(d, cam, opt)
        for name, rgba, L in (("birdseye", (0.1, 0.4, 1.0, 1), lo.ROW_Y - lo.BIRDSEYE_POS[1]), ("claw", (1.0, 0.2, 0.6, 1), 0.15)):
            c, corners = _frustum(m, d, name, L)
            _add_dot(r.scene, c, rgba)
            for k, p in enumerate(corners):
                _add_line(r.scene, c, p, rgba)
                _add_line(r.scene, p, corners[(k + 1) % 4], rgba)
        views.append(r.render())
    r.close()
    cams.close()
    top = np.concatenate([_label(views[0], "overview (roof hidden): birdseye blue, claw pink"),
                          _label(views[1], "from above (roof hidden): row R1 at top, aisle, birdseye")], 1)
    bottom = np.concatenate([_label(be, "birdseye (640x480, lens model)"), _label(cl, "claw (640x480, lens model)")], 1)
    Image.fromarray(np.concatenate([top, bottom], 0)).save(path)
    print(f"wrote {path}")


def windowed():
    import mujoco.viewer
    m, d, info = lo.build(SEEDS)
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = (-0.2, 0, 0.3)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, -150, -25
        while v.is_running():
            t = time.time()
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0.0, m.opt.timestep - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picture", metavar="PNG", help="write the layout picture here")
    ap.add_argument("--windowed", action="store_true")
    a = ap.parse_args()
    if a.picture:
        picture(a.picture)
        return 0
    if a.windowed:
        windowed()
        return 0
    return run_checks()


if __name__ == "__main__":
    sys.exit(main())
