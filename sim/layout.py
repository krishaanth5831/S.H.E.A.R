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
- **Base camera** (K7 2026-11-06): low, on a bracket off the base plate, left of the base, looking across at p0's
  stem. The dataset records base + claw since the v1 rig; the birdseye and its mast stay in the scene for the
  label tools, but no policy sees it.

Row R0 stays empty: the birdseye looks through it. All positions are design guesses (unverified)
until the rig is built.
"""
import mujoco
import numpy as np

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
# K7 2026-11-06: the base moved back from the aisle centreline, away from the row. The arm was cramped against p0:
# with the base 5 cm further back the pot-turn search found MORE plannable cuts, not fewer (3 rows: 1 -> 3 of 12),
# and the new base camera gets room. Unverified beyond those rows.
ARM_SETBACK = 0.05
ARM_Y = AISLE_Y - ARM_SETBACK
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

# --- base camera (K7 2026-11-06): replaces the birdseye in the dataset ---
# The birdseye on its mast looks along the arm's own line of reach, so at the cut the arm hides the cut point; the
# claw camera looks along the jaw and can't see how far the blade still is from the petiole. This camera sits low on
# a bracket off the arm's base plate, to the LEFT of the base (away from p1 and p2, which clutter the view from the
# right), and looks across at p0's stem, so it sees the blade and the petiole side-on. Straight in front of the base
# it was useless: at the cut pose the lower arm filled its whole frame (preview 2026-11-06). ⚠️ At left 12 / forward
# 5 / up 6 cm it sat right under the gripper of the folded start pose (START_NORM folds the arm to the left: the
# gripper rests 13 cm left of the base, 12 cm up), and a return route pressed it with 364 N (trial ep_01). Now well
# forward and low, out of the fold's sweep; from there it still sees the cuts on p0 and p1. Offsets in the base's own
# frame (forward = toward the row, left, up), so it moves with the base. Design guesses, unverified on the rig.
BASE_CAM_FWD, BASE_CAM_LEFT, BASE_CAM_Z = 0.15, 0.14, 0.04
BASE_CAM_LOOK_Z = 0.22                   # aim: p0's stem at the middle of the leaves the rule cuts
BASE_POST_R = 0.006


def _base_cam_frame(base, arm_yaw):
    """Camera position and (right, up, forward) unit axes in the world, looking at p0's stem."""
    fwd = np.array([np.cos(arm_yaw), np.sin(arm_yaw), 0.0])      # model +x = toward the row
    left = np.array([-np.sin(arm_yaw), np.cos(arm_yaw), 0.0])
    pos = base + BASE_CAM_FWD * fwd + BASE_CAM_LEFT * left + np.array([0, 0, BASE_CAM_Z])
    tgt = np.array([PLANT_X[0], ROW_Y, BASE_CAM_LOOK_Z])
    f = tgt - pos
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return pos, right, up, f


def base_cam_xml(base, arm_yaw):
    """The base camera plus its bracket: a flat bar on the floor from the base plate, a post, the housing and lens.
    Named gh_basecam_* so the cutter treats it as structure (hard: the arm never touches it)."""
    pos, right, up, f = _base_cam_frame(base, arm_yaw)
    bits = f'contype="{pl.FLOOR_BITS[0]}" conaffinity="{pl.FLOOR_BITS[1]}"'
    v = lambda a: " ".join(f"{x:.5f}" for x in a)
    foot = np.array([pos[0], pos[1], 0.0])
    bw, bh, bd = CAM_BOARD
    centre = pos - f * (LENS_LEN + bd / 2)
    g = [f'<geom name="gh_basecam_bar" type="cylinder" fromto="{v(base + [0, 0, 0.006])} {v(foot + [0, 0, 0.006])}" '
         f'size="{BASE_POST_R}" rgba="{MAST_RGBA}" {bits}/>',
         f'<geom name="gh_basecam_post" type="cylinder" fromto="{v(foot)} {v(centre * [1, 1, 0] + [0, 0, pos[2]])}" '
         f'size="{BASE_POST_R}" rgba="{MAST_RGBA}" {bits}/>',
         f'<geom name="gh_basecam_cam" type="box" pos="{v(centre)}" xyaxes="{v(right)} {v(up)}" '
         f'size="{bw / 2} {bh / 2} {bd / 2}" rgba="{CAM_RGBA}" {bits}/>',
         f'<geom name="gh_basecam_lens" type="cylinder" fromto="{v(pos - f * LENS_LEN)} {v(pos - f * 0.001)}" '
         f'size="{LENS_R}" rgba="{CAM_RGBA}" {bits}/>',
         f'<camera name="base" pos="{v(pos)}" xyaxes="{v(right)} {v(up)}" fovy="{LENSES["base"].fovy:.4f}"/>']
    return "".join(g)


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


def build(seeds=(11, 12, 13), profile=sg.DEFAULT_PROFILE, leaf_scale=1.0, target=0, arm_xy=None, arm_yaw=ARM_YAW,
          bend=False, turns=None):
    """arm_xy / arm_yaw: put the arm's base elsewhere (reach.py slides it along the aisle; default: in front of p0).
    bend: petioles that bend at the stem (stemgen.make_plant). turns: per plant, the pot turned about its stem (rad).
    target: the plant the arm's ready pose (info["ready"]) waits in front of."""
    specs, bodies, eqs, where, stems = [], [], [], {}, []
    for k, seed in enumerate(seeds):
        spec = sg.make_plant(seed, profile, leaf_scale, bend, turns[k] if turns else 0.0)
        xy = (PLANT_X[k], ROW_Y)
        b, e, w = sg.plant_xml(spec, xy=xy, pp=f"p{k}_")
        specs.append(spec)
        bodies.append(b)
        eqs.append(e)
        stems.append(xy)
        where.update({(k, i): v for i, v in w.items()})
    base = np.array([*(arm_xy or (ARM_X, ARM_Y)), 0.0])
    world = mujoco.MjSpec.from_string(sc.world_xml(gh.greenhouse_xml() + mast_xml() + birdseye_xml()
                                                   + base_cam_xml(base, arm_yaw) + "".join(bodies), "".join(eqs)))
    # A pale sky, so the frames aren't black above the plants (cosmetic, unverified colours).
    world.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                      rgb1=(0.85, 0.90, 0.95), rgb2=(0.55, 0.65, 0.75), width=256, height=1536)
    s = sc.arm_spec()
    cam = s.body("wrist_camera").add_camera(name="claw", pos=CLAW_POS_IN_BODY, fovy=LENSES["claw"].fovy)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = CLAW_XYAXES
    frame = world.worldbody.add_frame(pos=base, quat=(np.cos(arm_yaw / 2), 0, 0, np.sin(arm_yaw / 2)))
    world.attach(s, frame=frame, prefix="arm/")
    sc.add_blade_pairs(world)              # ⚠️ parent/child bodies skip contacts without the pairs
    m = world.compile()
    # The offscreen buffer must fit the pinhole renders, not just 640x480 (lens.py).
    m.vis.global_.offwidth, m.vis.global_.offheight = pinhole_size()
    d = mujoco.MjData(m)
    q = sc.settled_arm()[0]                # wrist roll that levels the jaw hinge, jaw open
    for j, v in q.items():
        d.qpos[m.joint("arm/" + j).qposadr[0]] = v
    sx, sy = stems[target]
    ready, ready_err = ik.reach(m, d, (sx, sy - READY_STANDOFF, READY_Z))
    for j, v in (q | ready).items():
        d.qpos[m.joint("arm/" + j).qposadr[0]] = v
        d.ctrl[m.actuator("arm/" + j).id] = v
    mujoco.mj_forward(m, d)
    info = dict(specs=specs, where=where, stems=stems, target_plant=target, arm_base=base, arm_yaw=arm_yaw,
                ready=ready, ready_err=ready_err)
    return m, d, info
