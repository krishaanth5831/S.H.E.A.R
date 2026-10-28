"""SHEAR's scene: the mini greenhouse, a row of random dwarf cherry tomatoes, the SO-101 in the aisle, the cameras.

    m, d, info = build(seeds=(11, 12, 13))     dwarf cherry tomatoes in pots along row R1; the arm faces p0

Seen from above (greenhouse frame, greenhouse.py):

      y ^  0.40  row R1 (crop rail above)   p0      p1      p2     <- plants hang from the crop rail
        |                                    ^       ^
        |  0.20  aisle                      arm      |            <- base square to the row, facing p0
        |  0.00  row R0 (no plants)
        | -0.15                                   birdseye        <- on its mast; level, straight at the row, through R0
        +-----------------------------------------------------> x  (end frames at 0 and 0.74)

- **Arm** (K7 2026-10-28): the base faces the row squarely (shoulder_pan 0 points straight at the row,
  +y), centred on the aisle, in front of the plant it works on.
- **Birdseye** (K7 2026-10-28): looks straight at the row, level, from the far side of the aisle, centred
  on the row. It does not look over the arm's shoulder like B.L.U.S.H's mast. The arm's base stands off
  its centre line and below its frame, so the camera sees the arm only when it reaches into the row.
- **Claw**: the SO-101's stock wrist camera mount, where the real camera sits; it looks along the jaw.

Row R0 stays empty: the birdseye looks through it. All positions are design guesses (unverified)
until the rig is built.
"""
import numpy as np
import mujoco

import greenhouse as gh
import ik
import plant as pl
import scene as sc
import stemgen as sg
from cameras import pinhole_size
from lens import LENSES

# --- the row ---
ROW_Y = gh.ROWS[0] - gh.KEYHOLE_OFF      # R1's keyholes, on the aisle side of the crop rail
PLANT_X = (0.22, 0.37, 0.52)             # along the row (unverified spacing: 0.15 m, scaled)
AISLE_Y = (gh.ROWS[0] + gh.ROWS[1]) / 2  # 0.20, the aisle between R1 and R0

# --- the arm ---
ARM_X = PLANT_X[0]                       # in front of the plant it works on
ARM_YAW = np.pi / 2                      # shoulder_pan 0 points +y, straight at the row (model x -> world +y)
# Where the arm waits: fingertips this far in front of the target plant's stem axis, at this height
# (unverified; step 3 sets the approach from here).
READY_STANDOFF, READY_Z = 0.10, 0.30

# --- birdseye: straight at the row ---
# Level, centred on the row, at the middle of the work band, behind the empty row R0: it looks through the
# gap between R0's crop rail (z 0.52) and anchor rail (z < 0.04). ⚠️ From the aisle edge (y 0.02, 0.36 m
# from the row) it framed only the middle of the plants (first greenhouse render, 2026-10-28). At ~0.53 m the
# 72 x 52 deg frame (lens.py, 640x480) covers ~0.77 m of row (most of the 0.74 m greenhouse) and z ~0.09-0.61.
# Guesses (unverified).
BIRDSEYE_POS = (PLANT_X[1], -0.15, 0.34)
BIRDSEYE_TILT_DEG = 0.0                  # + = looking down

# --- the birdseye's mast (K7 2026-10-28: the camera sits on a mast, not in the air) ---
# A weighted base plate on the floor, a pole up to the camera, a short bracket forward, and the camera
# board's housing with the lens at its front face, where the MuJoCo camera sits. Everything is behind the
# lens, so the birdseye never sees its own mast (check_layout.py checks its segmentation). Sizes are design
# guesses (unverified) until the mast is printed: B.L.U.S.H's printed mast had a 24 mm pole on a base plate.
MAST_POLE_R = 0.012
MAST_BACK = 0.045                        # pole axis behind the lens
MAST_BASE = (0.11, 0.11, 0.008)          # base plate x, y, thickness (screwed or weighed down)
CAM_BOARD = (0.032, 0.032, 0.020)        # InnoMaker 32 x 32 mm board in its housing: width, height, depth
LENS_R, LENS_LEN = 0.008, 0.012          # M12 lens barrel
MAST_RGBA = "0.60 0.60 0.62 1"
CAM_RGBA = "0.12 0.12 0.12 1"

# --- claw: on the stock wrist camera mount (so101_new_calib_camera.xml body "wrist_camera") ---
# Its body +z points out of the lens toward the jaw (check_layout.py checks the fingertips sit near the
# image centre). A MuJoCo camera looks down its own -z, so the camera's y flips: x = body x, y = -body y.
CLAW_POS_IN_BODY = (0.0, 0.0, 0.03)      # just outside the housing mesh (its z reaches 27 mm)
CLAW_XYAXES = (1, 0, 0, 0, -1, 0)


def birdseye_xml():
    t = np.radians(BIRDSEYE_TILT_DEG)
    # Looking +y: image right = +x, image up = (0, sin t, cos t).
    x, y, z = BIRDSEYE_POS
    return (f'<camera name="birdseye" pos="{x} {y} {z}" xyaxes="1 0 0 0 {np.sin(t):.5f} {np.cos(t):.5f}" '
            f'fovy="{LENSES["birdseye"].fovy:.4f}"/>')


def mast_xml():
    """The birdseye's mast, housing and lens. Named gh_mast_* so checks treat it as structure."""
    x, y, z = BIRDSEYE_POS
    bits = f'contype="{pl.FLOOR_BITS[0]}" conaffinity="{pl.FLOOR_BITS[1]}"'
    yp = y - MAST_BACK                                     # pole axis (the camera looks +y)
    bw, bh, bd = CAM_BOARD
    yb = y - LENS_LEN - bd / 2                             # housing centre: lens barrel in front of it
    g = [f'<geom name="gh_mast_base" type="box" pos="{x} {yp} {MAST_BASE[2] / 2}" '
         f'size="{MAST_BASE[0] / 2} {MAST_BASE[1] / 2} {MAST_BASE[2] / 2}" rgba="{MAST_RGBA}" {bits}/>',
         f'<geom name="gh_mast_pole" type="cylinder" fromto="{x} {yp} {MAST_BASE[2]} {x} {yp} {z}" '
         f'size="{MAST_POLE_R}" rgba="{MAST_RGBA}" {bits}/>',
         f'<geom name="gh_mast_bracket" type="box" pos="{x} {(yp + yb - bd / 2) / 2:.4f} {z}" '
         f'size="0.010 {abs(yb - bd / 2 - yp) / 2 + MAST_POLE_R:.4f} 0.006" rgba="{MAST_RGBA}" {bits}/>',
         f'<geom name="gh_mast_cam" type="box" pos="{x} {yb:.4f} {z}" size="{bw / 2} {bd / 2} {bh / 2}" '
         f'rgba="{CAM_RGBA}" {bits}/>',
         f'<geom name="gh_mast_lens" type="cylinder" fromto="{x} {y - LENS_LEN} {z} {x} {y - 0.001} {z}" '
         f'size="{LENS_R}" rgba="{CAM_RGBA}" {bits}/>']
    return "".join(g)


def build(seeds=(11, 12, 13), arm=True, profile=sg.DEFAULT_PROFILE, leaf_scale=1.0, target=0):
    specs, bodies, eqs, where, stems = [], [], [], {}, []
    for k, seed in enumerate(seeds):
        spec = sg.make_plant(seed, profile, leaf_scale)
        xy = (PLANT_X[k], ROW_Y)
        b, e, w = sg.plant_xml(spec, xy=xy, pp=f"p{k}_")
        specs.append(spec); bodies.append(b); eqs.append(e); stems.append(xy)
        where.update({(k, i): v for i, v in w.items()})
    world = mujoco.MjSpec.from_string(sc.world_xml(gh.greenhouse_xml() + mast_xml() + birdseye_xml() + "".join(bodies),
                                                   "".join(eqs)))
    # A pale sky, so the frames aren't black above the plants (cosmetic, unverified colours).
    world.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=(0.85, 0.90, 0.95), rgb2=(0.55, 0.65, 0.75), width=256, height=1536)
    base = np.array([ARM_X, AISLE_Y, 0.0])
    if arm:
        s = sc.arm_spec()
        cam = s.body("wrist_camera").add_camera(name="claw", pos=CLAW_POS_IN_BODY, fovy=LENSES["claw"].fovy)
        cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
        cam.alt.xyaxes = CLAW_XYAXES
        frame = world.worldbody.add_frame(pos=base, quat=(np.cos(ARM_YAW / 2), 0, 0, np.sin(ARM_YAW / 2)))
        world.attach(s, frame=frame, prefix="arm/")
        # ⚠️ Same blade pair as scene.build: parent/child bodies skip contacts without it.
        p = world.add_pair(geomname1="arm/blade_fixed", geomname2="arm/blade_moving")
        p.condim = 3
        p.solref, p.solimp = pl.CONTACT_SOLREF, pl.CONTACT_SOLIMP
        p.friction = (0.3, 0.3, 0.005, 0.0001, 0.0001)
    m = world.compile()
    # The offscreen buffer must fit the pinhole renders, not just 640x480 (lens.py).
    m.vis.global_.offwidth, m.vis.global_.offheight = pinhole_size()
    d = mujoco.MjData(m)
    ready = ready_err = None
    if arm:
        q = sc.settled_arm()[0]            # wrist roll that levels the jaw hinge, jaw open
        for j, v in q.items():
            d.qpos[m.joint("arm/" + j).qposadr[0]] = v
        sx, sy = stems[target]
        ready, ready_err = ik.reach(m, d, (sx, sy - READY_STANDOFF, READY_Z))
        for j, v in (q | ready).items():
            d.qpos[m.joint("arm/" + j).qposadr[0]] = v
            d.ctrl[m.actuator("arm/" + j).id] = v
    mujoco.mj_forward(m, d)
    info = dict(specs=specs, where=where, stems=stems, target_plant=target, arm_base=base, arm_yaw=ARM_YAW,
                ready=ready, ready_err=ready_err)
    return m, d, info
