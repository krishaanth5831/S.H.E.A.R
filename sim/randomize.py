"""Domain randomisation for one SHEAR scene: lighting, colours and camera poses (K7 2026-11-03: "do varied lighting
and colors like domain randomization").

    params = apply(m, seed)       after the scene is built, before the first frame (cutter.Scene does it)

Adapted from B.L.U.S.H sim/randomize.py (commit 41da64d): its lights, surfaces and camera jitter, with SHEAR's own
geoms. Not its physics draws: SHEAR's cut judge is about stems and stubs, and K7 asked for the look only.
Everything is drawn from the scene seed on its own stream, so the plants are the same with or without it, and
the params are returned for the episode log.

What is randomised and why:
- Lights and the headlight ambient: the real greenhouse's light isn't known (sun, cloud, grow lamps).
- Greenhouse ply, floor, pots, mast: their real finish isn't known.
- Leaves, stems, fruit: jittered INSIDE their class (hue a little, saturation and brightness more), so a green
  leaf stays green and a yellow one yellow, and a ripe fruit stays red: the rule's and the labels' classes keep
  meaning the same thing (as B.L.U.S.H keeps fruit inside its ripeness class).
- Camera poses: the birdseye mast is placed by hand and the claw camera sits on a printed mount (B.L.U.S.H's
  ranges).
"""
import colorsys

import mujoco
import numpy as np

import stemgen as sg

# All ranges are guesses (unverified) around the nominal look; each says what it spans.
LIGHT_SCALE = (0.55, 1.45)        # x the nominal diffuse of each light (B.L.U.S.H 0.6-1.4, a little wider)
LIGHT_DIR_DEG = 20.0              # +- tilt of each light's direction (B.L.U.S.H 15)
AMBIENT = (0.15, 0.45)            # headlight ambient (nominal 0.3)
SURFACE_VAL = (0.80, 1.20)        # x brightness of the ply, floor, pots, mast
SURFACE_HUE = 0.02                # +- hue turn on those surfaces
# Per plant class: (hue +- in turns, saturation x, value x). 0.02 turns = ~7 deg: green stays green.
PLANT_HSV = {"leaf": (0.020, (0.85, 1.15), (0.80, 1.20)),
             "stem": (0.020, (0.85, 1.15), (0.85, 1.15)),
             "fruit": (0.012, (0.85, 1.10), (0.80, 1.15))}
BIRDSEYE_POS_M, BIRDSEYE_ROT_DEG = 0.010, 2.0   # mast placement by hand (B.L.U.S.H)
BASE_POS_M, BASE_ROT_DEG = 0.004, 2.0           # printed bracket off the base plate (guess, as the claw mount)
CLAW_POS_M, CLAW_ROT_DEG = 0.002, 1.5           # printed wrist mount (B.L.U.S.H)


def _rot(rng, max_deg):
    """Small random rotation as a quaternion (w, x, y, z). (B.L.U.S.H randomize._rot)"""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = np.radians(rng.uniform(-max_deg, max_deg))
    return np.r_[np.cos(a / 2), np.sin(a / 2) * axis]


def _qmul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _hsv(rgb, dh, ks, kv):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return np.clip(colorsys.hsv_to_rgb((h + dh) % 1.0, min(1, s * ks), min(1, v * kv)), 0, 1)


def _kind(name):
    c = sg.seg_class(name)
    if c in ("petiole", "midrib", "leaflet", "scar", "stub"):
        return "leaf"
    if c in ("stem", "rachis", "pedicel", "calyx"):
        return "stem"
    if c == "fruit":
        return "fruit"
    if name == "floor" or name.startswith("gh_") or c == "pot":
        return "surface"
    return None


def apply(m, seed):
    """Randomise model m's look in place for scene `seed`. Returns the drawn params."""
    rng = np.random.default_rng([int(seed), 0x5EA4])     # own stream: the plants' draws are untouched
    p = {}
    for i in range(m.nlight):
        k = float(rng.uniform(*LIGHT_SCALE))
        m.light_diffuse[i] *= k
        dvec = np.zeros(3)
        mujoco.mju_rotVecQuat(dvec, m.light_dir[i].copy(), _rot(rng, LIGHT_DIR_DEG))
        m.light_dir[i] = dvec
        p[f"light_{m.light(i).name}"] = {"diffuse_x": round(k, 3), "dir": np.round(dvec, 3).tolist()}
    amb = float(rng.uniform(*AMBIENT))
    m.vis.headlight.ambient[:] = amb
    p["ambient"] = round(amb, 3)
    # one draw per kind of thing (all leaves shift together, then each leaf a little on its own)
    draws = {k: (rng.uniform(-h, h), rng.uniform(*s), rng.uniform(*v)) for k, (h, s, v) in PLANT_HSV.items()}
    draws["surface"] = (rng.uniform(-SURFACE_HUE, SURFACE_HUE), 1.0, rng.uniform(*SURFACE_VAL))
    for i in range(m.ngeom):
        kind = _kind(m.geom(i).name or "")
        if kind is None:
            continue
        dh, ks, kv = draws[kind]
        if kind != "surface":                            # each organ a little different from its neighbours
            dh += rng.uniform(-0.005, 0.005)
            kv *= rng.uniform(0.95, 1.05)
        m.geom_rgba[i, :3] = _hsv(m.geom_rgba[i, :3], dh, ks, kv)
    p["colour"] = {k: [round(float(x), 3) for x in v] for k, v in draws.items()}
    for name, dp, dr in (("birdseye", BIRDSEYE_POS_M, BIRDSEYE_ROT_DEG), ("base", BASE_POS_M, BASE_ROT_DEG),
                         ("arm/claw", CLAW_POS_M, CLAW_ROT_DEG)):
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if c < 0:
            continue
        off = rng.uniform(-dp, dp, 3)
        m.cam_pos[c] += off
        m.cam_quat[c] = _qmul(m.cam_quat[c], _rot(rng, dr))
        p[f"cam_{name.split('/')[-1]}"] = {"pos_offset_mm": np.round(off * 1000, 1).tolist()}
    return p
