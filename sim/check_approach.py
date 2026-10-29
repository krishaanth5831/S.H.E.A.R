#!/usr/bin/env python3
"""Step 3 of the SHEAR sim plan: does the cut still work across the ways the arm can meet a leaf?

  python check_approach.py                 PASS/FAIL checks + one line of numbers per case, exit 1 on a FAIL
  python check_approach.py --windowed --case az=120,el=30     watch one case (it loops)
  python check_approach.py --reach         the reach survey on the greenhouse's dwarf plants (kinematic, ~6 min)
  python check_approach.py --reach --free-roll    the same, as if the wrist rolled a full turn (a what-if)

Each case is step 1 with the arm moved: cutpose.CutIK puts the blades on one dwarf-size leaf, the arm
settles there on its servos (jaw open), the stem and leaf are built around where the blades really ended up,
and the jaw closes like the real servo. One stem, one leaf, nothing else in the way.

Cases (keys for --case; all default to az=-90, el=30, offset 2 mm, a 3 mm petiole, flush):
- az: the petiole's direction around the stem: 0 = pointing straight at the arm, -90 = along the row to the arm's
  left (the default: where a flush cut fits), 90 = to its right, 180 = straight away. The arm takes its best
  flush pose (cutpose.py): blade level and fingers level where it can, leaning or twisted where it can't.
- el: the petiole above level (dwarf leaves 25-45 deg green, 5-20 deg yellow; stemgen.ELEVATION_DEG).
- psi: the approach angle: the fingers turned this far about the hinge from the best pose (on the default
  leaf that's the fingers' pitch). The arm holds the fingers there and gives up twist, then lean, instead.
- mode: flush (default) or square (the blade square to the petiole, leaning as far as the petiole rises).
- offset: blade plane to stem surface (mm), the stub to leave. r: petiole radius (mm).
- shift: the petiole off the blades' centre along the fingers (mm, + toward the palm).
- gap: the petiole this far above the fixed edge when the jaw starts closing (mm): servo sag / IK error.
- jaw: how far the jaw is open when the cut starts (deg; -10 = shut). Default scene.JAW_OPEN_DEG.

Per case: the tilt (jaw hinge vs petiole; 0 = blade square to it), cut or not, the stub at the petiole's axis
and its near/far edges (a tilted blade cuts a slanted face), the peak force of any gripper part on the stem
(the costliest error), and the blade force. Trusses and neighbouring leaves are --reach's job (kinematic)
and step 5's (the scripted cutter in the full greenhouse).

Run in the lerobot env under the 9 GB cap. Writes nothing. Needs ./fetch_so101.sh once.
"""
import argparse
import dataclasses
import sys
import time

import mujoco
import numpy as np

import cutpose as cp
import plant as pl
import scene as sc
import stemgen as sg

# --- the rig: the arm's base at the origin facing +x, one dwarf stem in front of it ---
# The stem stands as far in front of the base as row R1 in the greenhouse (layout.ROW_Y - AISLE_Y, 0.185 m).
STEM_X = 0.185
Z_CUT = 0.27            # cut height: mid dwarf plant (stemgen dwarf leaves sit at 0.15-0.45 m)
DWARF = sg.PROFILES["dwarf_cherry"]
STEM_R = float(np.mean(DWARF.stem_r))
# A mid-size dwarf leaf (stemgen's dwarf ranges, middle values; all unverified like them).
_L = float(np.mean(DWARF.leaf_len))
LEAF = pl.LeafParams(length=_L, mass=sg.LEAF_MASS_REF[1] * (_L / sg.LEAF_MASS_REF[0]) ** 2, n_seg=3,
                     petiole_r=0.0015, petiole_len=DWARF.petiole_frac * _L, leaflet_l=float(np.mean(DWARF.leaflet_l)),
                     leaflet_w=float(np.mean(DWARF.leaflet_l) * np.mean(DWARF.leaflet_aspect)), droop_deg=10.0,
                     rgba=sg.LEAF_COLOURS["green"][0], leaflet_rgba=sg.LEAF_COLOURS["green"][1], style="tomato",
                     mid_r=DWARF.mid_r, bend=True)
# The default leaf points along the row to the arm's left (az -90): the direction a flush cut fits (cutpose.py).
DEFAULT = dict(az=-90.0, el=30.0, psi=None, offset=2.0, r=1.5, shift=0.0, gap=0.1, mode="flush", jaw=sc.JAW_OPEN_DEG)

# --- timing, as check_cut.py ---
SETTLE_S, CLOSE_S, HOLD_S = 1.0, 0.4, 1.1
STEM_TOUCH_N = 0.5      # gripper force on the stem that counts as touching it (check_cut.py, unverified)
STUB_TOL = 0.001        # m, stub at the axis vs the offset asked for (check_cut.py)
# Built-in overlap deeper than this between the arm and the plant (stem or leaf) means the pose can't be
# reached without going through the plant: reported as blocked, not run (MuJoCo would fling them apart).
BLOCK_PEN = 0.0005
SQUARE_DEG = 2.0        # tilt under this counts as square

AZ = (-90, -60, -30, 0, 30, 60, 90, 120, 150, 180)
EL = (10, 30, 45)
PSI = (-60, -45, -30, -15, 15, 30, 45, 60)
LEAN_EL = (0, 5, 10, 15, 20, 30)          # square cuts on a leaf along the row: the blade leans as far as el
LEAN_OFFSET = (2.0, 4.0, 6.0)


def petiole_dir(az, el):
    """az 0 = toward the arm's base (-x), + turning toward -y; el above level."""
    az, el = np.radians(az), np.radians(el)
    return np.array([-np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


_IK = None


def rig_ik():
    global _IK
    if _IK is None:
        _IK = cp.CutIK(base=(0.0, 0.0, 0.0), yaw=0.0)
    return _IK


def case_pose(c):
    """The arm pose for a case: CutIK's best pose, or the one holding the fingers psi off it (turned about
    the petiole's level direction for a flush cut, about the petiole for a square one)."""
    a = petiole_dir(c["az"], c["el"])
    r, off = c["r"] / 1000, c["offset"] / 1000
    S = np.array([STEM_X, 0.0, Z_CUT])
    sols = rig_ik().solve(S, a, STEM_R, off, r, mode=c["mode"])
    if not sols:
        return a, None
    if c["psi"]:
        f = _rot(cp.level(a) if c["mode"] == "flush" else a, c["psi"]) @ sols[0].fingers
        sols = rig_ik().solve(S, a, STEM_R, off, r, mode=c["mode"], fingers=f, x0=sols[0].q)
        sols = sorted(sols, key=lambda p: -p.fingers @ f)
        if not sols:
            return a, None
    return a, sols[0]


def _overlap(m, d, robot, plant):
    """Deepest built-in penetration between robot geoms and plant geoms (m, >= 0), and what overlaps there."""
    deepest, what = 0.0, ""
    for i in range(d.ncon):
        c = d.contact[i]
        if ((c.geom1 in robot and c.geom2 in plant) or (c.geom2 in robot and c.geom1 in plant)) and -c.dist > deepest:
            deepest = -c.dist
            what = " / ".join(sorted((m.geom(g).name or m.body(m.geom_bodyid[g]).name).replace("arm/", "")
                                     for g in (c.geom1, c.geom2)))
    return deepest, what


def build_case(c, torque=sc.GRIPPER_TORQUE):
    a, pose = case_pose(c)
    if pose is None:
        return None
    params = dataclasses.replace(LEAF, petiole_r=c["r"] / 1000)
    m, d, info = sc.build(offset=c["offset"] / 1000, shift=c["shift"] / 1000, gripper_torque=torque, pose=pose.q,
                          a=a, params=params, stem_r=STEM_R, clearance=c["gap"] / 1000, cut=c["mode"],
                          jaw_open_deg=c["jaw"])
    info.update(a=a, pose=pose, petiole_r=c["r"] / 1000)
    return m, d, info


def stub_range(S, a, petiole_r, plane_pt, n, stem_r=STEM_R, k=36):
    """Min and max distance from the stem surface to the cut face, square to the (vertical) stem: where each of
    k lines along the petiole's surface crosses the blade plane. A tilted blade leaves a slanted face."""
    e1 = np.cross(a, UP if abs(a[2]) < 0.9 else np.array([1.0, 0, 0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    h = []
    for th in np.linspace(0, 2 * np.pi, k, endpoint=False):
        u = petiole_r * (np.cos(th) * e1 + np.sin(th) * e2)
        t = ((plane_pt - S - u) @ n) / (a @ n)
        P = S + u + a * t
        h.append(np.hypot(*(P - S)[:2]) - stem_r)
    return min(h), max(h)


UP = np.array([0.0, 0.0, 1.0])


def run_case(c, torque=sc.GRIPPER_TORQUE):
    built = build_case(c, torque)
    if built is None:
        return dict(reach=False)
    m, d, info = built
    cut_force = pl.cut_force_for(info["petiole_r"])
    cutter = pl.Cutter(m, d, info, cut_force)
    a, n = info["a"], info["gmat"][:, 1]
    nl = cp.level(n)
    out = dict(reach=True, tilt=float(np.degrees(np.arccos(np.clip(n @ a, -1, 1)))),
               lean=float(np.degrees(np.arcsin(np.clip(n[2], -1, 1)))),
               twist=float(np.degrees(np.arccos(np.clip(nl @ cp.level(a), -1, 1)))),
               pitch=float(np.degrees(np.arcsin(np.clip(-info["gmat"][2, 2], -1, 1)))),
               pos_err=info["pose"].pos_err, fingers=info["pose"].fingers, cut_force=cut_force, mode=c["mode"])
    leaf = m.body("leaf").id
    # ⚠️ The rig is about the cut, not the approach: the leaf's own leaflets and midrib don't collide with the
    # arm here (a hand coming in pushes them aside; built into the hand they would fling the leaf off). Only
    # the stem and the petiole count. Step 5's scripted cutter runs in the full scene, where they all collide.
    soft = [g for g in np.flatnonzero(m.body_rootid[m.geom_bodyid] == leaf) if g != cutter.petiole]
    m.geom_contype[soft], m.geom_conaffinity[soft] = pl.FALLEN_BITS        # the floor only
    cutter.leaf_geoms = np.array([cutter.petiole])     # reset() / the cut then only touch the petiole's bits
    mujoco.mj_forward(m, d)
    pen, what = _overlap(m, d, cutter.robot_geoms, {cutter.petiole, cutter.stem})
    out["overlap"], out["overlap_what"] = pen, what
    if pen > BLOCK_PEN:
        out["blocked"] = True
        return out
    out["blocked"] = False
    jaw, G = m.actuator("arm/gripper").id, m.body("arm/gripper").id
    blades = [m.geom("arm/blade_fixed").id, m.geom("arm/blade_moving").id]
    n_steps = round((SETTLE_S + CLOSE_S + HOLD_S) / m.opt.timestep)
    tail, at_cut = [], None
    for k in range(n_steps):
        u = np.clip((d.time - SETTLE_S) / CLOSE_S, 0.0, 1.0)
        d.ctrl[jaw] = info["jaw_open"] + (sc.GRIPPER_RANGE_RAD[0] - info["jaw_open"]) * u
        mujoco.mj_step(m, d)
        cutter.step()
        if cutter.just_cut:
            at_cut = stub_range(d.xpos[leaf].copy(), d.xmat[leaf].reshape(3, 3)[:, 0].copy(), info["petiole_r"],
                                np.mean([d.geom_xpos[b] for b in blades], axis=0), d.xmat[G].reshape(3, 3)[:, 1].copy())
        if d.time > SETTLE_S + CLOSE_S + HOLD_S - 0.2:
            tail.append(cutter.squeeze)
    out.update(cut=cutter.cut_t is not None, stub=cutter.stub, stem_peak=cutter.peak["stem_force"],
               squeeze_peak=cutter.peak["squeeze"], squeeze_ss=float(np.median(tail)), blade_touch=cutter.blade_touch)
    if at_cut:
        out.update(stub_min=at_cut[0], stub_max=at_cut[1])
    return out


def fmt(c, r):
    key = ",".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in c.items()
                   if v is not None and v != DEFAULT[k]) or "default"
    if not r["reach"]:
        return f"{key:26s} no IK pose"
    blade = f"lean {r['lean']:+5.1f} twist {r['twist']:5.1f} pitch {r['pitch']:+5.1f}"
    head = f"{key:26s} {blade}  fingers ({r['fingers'][0]:+.2f} {r['fingers'][1]:+.2f} {r['fingers'][2]:+.2f})"
    if r["blocked"]:
        return f"{head}  BLOCKED: {r['overlap'] * 1000:.1f} mm overlap ({r['overlap_what']}) at the cut pose"
    if not r["cut"]:
        return (f"{head}  no cut   blades {r['squeeze_ss']:4.1f} N (need {r['cut_force']:.1f})  "
                f"stem {r['stem_peak']:5.2f} N")
    return (f"{head}  CUT  stub {r['stub_min'] * 1000:4.1f}..{r['stub_max'] * 1000:4.1f} mm"
            f"  stem {r['stem_peak']:5.2f} N")


def case(**kw):
    return DEFAULT | kw


GROUPS = {
    "direction (flush)": [case(az=az, el=el) for el in EL for az in AZ],
    "lean (square cuts on a leaf along the row)": [case(mode="square", el=el, offset=o) for el in LEAN_EL
                                                   for o in LEAN_OFFSET],
    "finger pitch (flush)": [case(psi=p) for p in PSI],
    "offset x petiole (flush)": [case(offset=o, r=r) for r in (1.2, 1.5, 1.8) for o in (1.0, 2.0, 4.0, 8.0)],
    "placement": [case(shift=s_) for s_ in (-2.0, 2.0)] + [case(gap=g) for g in (0.5, 1.0, 2.0)],
}


def run_all(table):
    res = {}
    for name, cases in GROUPS.items():
        if table:
            print(f"--- {name}")
        for c in cases:
            t = time.time()
            r = run_case(c)
            res[_key(c)] = r
            if table:
                print(f"  {fmt(c, r)}   [{time.time() - t:.1f} s]", flush=True)
        if table:
            print()
    return res


def _key(c):
    return tuple(sorted((k, v) for k, v in c.items() if v is not None))


def run_checks(table):
    fails = 0

    def report(ok, msg):
        nonlocal fails
        fails += not ok
        print(("PASS " if ok else "FAIL ") + msg, flush=True)

    ik = rig_ik()
    print(f"rig: dwarf stem {STEM_R * 2000:.1f} mm, {STEM_X:.3f} m in front of the arm's base, cut at z {Z_CUT:.2f} m; "
          f"leaf {LEAF.length * 100:.1f} cm / {LEAF.mass * 1000:.1f} g, bending petiole; joint limits from "
          f"{ik.limits_from}. Cut force scales with the petiole: {pl.cut_force_for(0.0015):.1f} N for 3 mm "
          f"(unverified).\n", flush=True)
    res = run_all(table)
    get = lambda **kw: res[_key(case(**kw))]
    clean = lambda r, off=0.002: (r["reach"] and not r["blocked"] and r["cut"] and r["stem_peak"] < STEM_TOUCH_N
                                  and r["stub_min"] >= off - STUB_TOL / 2 and r["stub_max"] <= off + STUB_TOL)
    verdict = lambda r: ("no IK" if not r["reach"] else f"blocked ({r['overlap_what']})" if r["blocked"] else
                         "no cut" if not r["cut"] else "stem hit" if r["stem_peak"] >= STEM_TOUCH_N else
                         f"stub {r['stub_min'] * 1000:.1f}-{r['stub_max'] * 1000:.1f} mm")

    rs = [get(el=el) for el in EL]
    report(all(clean(r) for r in rs), f"flush cut on a leaf along the row (az -90), petiole up {'/'.join(map(str, EL))} deg: "
           + ", ".join(verdict(r) for r in rs) + f" (2 mm asked, stem untouched)")

    sq = [(el, o) for el in LEAN_EL if el >= 5 for o in LEAN_OFFSET]
    bad = [(el, o) for el, o in sq if not clean(get(mode="square", el=el, offset=o), o / 1000)]
    report(len(bad) == len(sq), f"why flush: a blade square to a rising petiole leans into the stem. Leaning 5-30 deg, "
           f"no clean cut at a 2-6 mm stub ({len(bad)}/{len(sq)}: the moving pad or the blade meets the stem)")

    ok_pitch = [get(psi=p)["pitch"] for p in PSI if get(psi=p)["reach"] and clean(get(psi=p))]
    bad_pitch = [get(psi=p)["pitch"] for p in PSI if get(psi=p)["reach"] and not clean(get(psi=p))]
    report(ok_pitch and min(ok_pitch) < -15 and max(ok_pitch) > 5,
           f"fingers pitched {min(ok_pitch):+.0f}..{max(ok_pitch):+.0f} deg off level still cut clean"
           + (f"; {', '.join(f'{p:+.0f}' for p in bad_pitch)} deg don't (the finger runs along the stem)" if bad_pitch else ""))

    offs = [(o, r) for r in (1.2, 1.5, 1.8) for o in (2.0, 4.0, 8.0)]
    bad = [(o, r) for o, r in offs if not clean(get(offset=o, r=r), o / 1000)]
    report(not bad, f"flush stubs of 2 / 4 / 8 mm on 2.4-3.6 mm petioles: the stub follows the offset, stem untouched"
           + (f" (fails: {bad})" if bad else ""))
    one = [get(offset=1.0, r=r) for r in (1.2, 1.5, 1.8)]
    print(f"INFO a 1 mm stub: {', '.join(verdict(r) + f' / stem {r['stem_peak']:.0f} N' for r in one)}"
          f" -> 2 mm is the closest clean flush cut with the stock fingertips")

    for kw, what in (({"shift": -2.0}, "2 mm toward the tips"), ({"shift": 2.0}, "2 mm toward the palm"),
                     ({"gap": 0.5}, "0.5 mm above the fixed edge"), ({"gap": 1.0}, "1 mm above the fixed edge")):
        report(clean(get(**kw)), f"petiole {what} (aim / sag error): {verdict(get(**kw))}")
    print(f"INFO petiole 2 mm above the fixed edge: {verdict(get(gap=2.0))} (blades {get(gap=2.0)['squeeze_ss']:.1f} N: "
          f"part of the squeeze goes into bending the petiole)")

    print("\nINFO which leaves one base position cuts flush (az: 0 = petiole at the arm, -90 = along the row to its left):")
    for az in AZ:
        rr = [get(az=az, el=el) for el in EL]
        r0 = rr[1]
        blade = (f"lean {r0['lean']:+3.0f} twist {r0['twist']:3.0f}" if r0["reach"] else "no IK")
        print(f"     az {az:4d}: {blade}  ->  {' | '.join(verdict(r) for r in rr)}")
    print("     Right-pointing leaves need the wrist rolled ~183 deg, past the follower's calibrated 165 (cutpose.py):"
          "\n     the jaw can't flip over, so the blade leans 20 deg into the stem.")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAIL'}")
    return fails


def parse_case(text):
    c = dict(DEFAULT)
    for part in filter(None, (text or "").split(",")):
        k, v = part.split("=")
        c[k] = v if k == "mode" else float(v)
    return c


def windowed(c):
    import mujoco.viewer
    built = build_case(c)
    if built is None:
        print("no IK pose for this case")
        return
    m, d, info = built
    cutter = pl.Cutter(m, d, info, pl.cut_force_for(info["petiole_r"]))
    leaf = m.body("leaf").id
    soft = [g for g in np.flatnonzero(m.body_rootid[m.geom_bodyid] == leaf) if g != cutter.petiole]
    m.geom_contype[soft], m.geom_conaffinity[soft] = pl.FALLEN_BITS
    cutter.leaf_geoms = np.array([cutter.petiole])
    jaw = m.actuator("arm/gripper").id
    p = info["pose"]
    print(f"{c}\nIK pose: lean {p.lean_deg:+.1f}, twist {p.twist_deg:.1f}, finger pitch {p.pitch_deg:+.1f} deg. "
          f"Cut force {pl.cut_force_for(info['petiole_r']):.1f} N (unverified). Leaflets pass through the hand here.")
    cycle = SETTLE_S + CLOSE_S + HOLD_S + 1.5
    last = 0.0
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = info["B"]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.25, 200, -25
        while v.is_running():
            t_wall = time.time()
            with v.lock():
                u = np.clip((d.time - SETTLE_S) / CLOSE_S, 0.0, 1.0)
                d.ctrl[jaw] = info["jaw_open"] + (sc.GRIPPER_RANGE_RAD[0] - info["jaw_open"]) * u
                mujoco.mj_step(m, d)
                cutter.step()
                if cutter.just_cut:
                    print(f"\n>>> cut at {cutter.squeeze_at_cut:.1f} N: stub {cutter.stub * 1000:.1f} mm")
                if d.time >= cycle:
                    print(f"\n    peak gripper force on the stem {cutter.peak['stem_force']:.2f} N")
                    mujoco.mj_resetData(m, d)
                    d.qpos[:], d.ctrl[:] = info["q0"], info["ctrl0"]
                    cutter.reset()
                    mujoco.mj_forward(m, d)
            if t_wall - last > 0.2:
                last = t_wall
                print(f"\rblade force {cutter.squeeze:6.2f} N   stem {cutter.stem_force:6.2f} N   ", end="", flush=True)
            v.sync()
            wait = sc.TIMESTEP - (time.time() - t_wall)
            if wait > 0:
                time.sleep(wait)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windowed", action="store_true", help="watch one case in a window (it loops)")
    ap.add_argument("--case", default="", help="e.g. az=120,el=30 or psi=30 or offset=1,r=1.2 (see the docstring)")
    ap.add_argument("--table", action="store_true", help="also print one line per case")
    ap.add_argument("--reach", action="store_true", help="reach survey on the greenhouse plants (kinematic)")
    ap.add_argument("--free-roll", action="store_true", help="--reach what-if: the wrist rolls a full turn")
    a = ap.parse_args()
    if a.windowed:
        windowed(parse_case(a.case))
        return 0
    if a.reach:
        import reach
        return reach.survey(a.free_roll)
    return 1 if run_checks(a.table) else 0


if __name__ == "__main__":
    sys.exit(main())
