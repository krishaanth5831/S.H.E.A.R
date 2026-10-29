"""SHEAR sim scene, step 1: the SO-101 with its stock rigid jaws and a blade on each fingertip, one stem,
and one leaf whose petiole crosses the blades.

    m, d, info = build()                 blade plane 2 mm off the stem surface (the stub a cut should leave)
    m, d, info = build(offset=0.008)     8 mm off
    m, d, info = build(shift=0.0085)     the petiole sits 8.5 mm deeper in the jaw, behind the blades

The arm holds one pose: step 1's ARM_POSE, or (step 3, check_approach.py) a cut pose from cutpose.CutIK. A first
pass settles it there and reads where the blades ended up, and the second pass builds the stem and the leaf
around them.

⚠️ The arm model needs the SO-ARM100 files: run ./fetch_so101.sh once (sim/assets/so101 is gitignored).
"""
import functools
import pathlib

import mujoco
import numpy as np

import plant as pl

HERE = pathlib.Path(__file__).resolve().parent
SO101_XML = HERE / "assets" / "so101" / "so101_new_calib_camera.xml"

# --- copied from B.L.U.S.H sim/scene.py (commit e91ef0c) ---
TIMESTEP = 0.002                # 500 Hz, B.L.U.S.H's step
# ⚠️ Squeezing needs elliptic friction cones + impratio > 1. With MuJoCo's default pyramidal cones a
# squeezed fruit crept out of B.L.U.S.H's fingers; the blades squeeze the petiole the same way.
CONE, IMPRATIO = "elliptic", 10
# Max = the 12 V STS3215's stall torque, 30 kg.cm (SO-ARM100 README; K7's follower runs on the 12 V PSU),
# which is the MJCF's own sts3215 class default. ⚠️ The MJCF actuators override it with 3.35 N.m (source
# unknown), and B.L.U.S.H e91ef0c uses that, so its jaw forces read 14 % high. ⚠️ The PSU sags under
# load, so the real jaw gives less than this: every blade force here is an upper bound.
GRIPPER_MAX_TORQUE = 30 * 0.0980665
# lerobot so_follower configure(): gripper Max_Torque_Limit 500 = 50 % of max.
GRIPPER_TORQUE = 0.5 * GRIPPER_MAX_TORQUE
# ⚠️ The servo drops to Overload_Torque 25 % once it has been stalled long enough, and a blade pressing
# a stalk it can't cut is a stall. check_cut.py measures the blade force at both levels.
GRIPPER_TORQUE_STALLED = 0.25 * GRIPPER_MAX_TORQUE
# follower_white gripper range 2030..3481 ticks = 127.5 deg of travel from the -10 deg closed stop.
GRIPPER_RANGE_RAD = (np.radians(-10.0), np.radians(-10.0 + (3481 - 2030) * 360 / 4095))
# ---

# The arm holds one pose in step 1: every joint at zero. That's the SO-101 straight out, fingers
# horizontal along +x, the jaw hinge along +y, the fixed jaw underneath and the moving jaw closing down
# onto it (checked on the model: gripper x = up, y = +y, z = -x at q = 0).
# ⚠️ Except the wrist roll: at q = 0 the stock jaw hinge tilts 2.8 deg (gripper y = (0, 0.9988, 0.0486)),
# so the blade plane leans into a vertical stem and the moving blade rubbed it with 31 N. settled_arm()
# rolls the wrist until the hinge is level.
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ARM_POSE = (0.0, 0.0, 0.0, 0.0, 0.0)
JAW_OPEN_DEG = 25.0             # while the plant settles; the fingertips are ~45 mm apart there
SETTLE_ARM_S = 1.0              # arm sags onto its position servos before the plant is placed

# --- the blades. ⚠️ Design guesses until K7 picks and mounts a real blade ---
# A side cutter at the tips: one thin plate on each fingertip, on the finger's stem side (-y), its edge
# facing the other jaw. The edges meet as the jaw closes, like the two edges of a flush cutter, so the
# cut is a plane across the petiole right next to the stem. Edges that slide past each other
# (secateurs) may cut cleaner; that choice is open (SHEAR Sim Plan, "Measure first").
BLADE_T = 0.0005                # real blade thickness: a snap-off utility blade, guess (unverified)
# ⚠️ Collision thickness. A 0.5 mm box can tunnel through a stalk at the 2 ms step (HOIST's thin hook
# on a thin wire did, 2026-10-27). The cut plane is the box's mid-plane = the real blade's.
BLADE_T_COL = 0.002
# Along the finger, from the tip: the flat part of the stock fingertip (measured on the mesh: the fixed
# jaw's inner face stays at x = -7.9 mm for its last 8 mm, then falls back).
BLADE_LEN = 0.008
BLADE_DEPTH = 0.006             # edge to back of the plate, inside the finger outline, guess (unverified)
# The edges meet 1 deg before the -10 deg closed stop, so the servo presses edge on edge instead of
# resting on the joint limit. That makes each edge stand ~1.6 mm proud of its fingertip (design choice).
JAW_MEET_DEG = -9.0
# The petiole is built this far above the fixed edge. ⚠️ The weld is near-rigid, so a petiole the moving
# blade has to push down onto the fixed one sends part of the squeeze into the weld instead of the cut:
# 6 of 22 N at a 0.5 mm gap. A real petiole bends there; modelling that is for step 3.
PETIOLE_CLEARANCE = 0.0001
BLADE_RGBA = (0.78, 0.80, 0.84, 1)
# Pads: the stock fingers collide through boxes, one per 8 mm slab of each finger mesh. ⚠️ The stock
# meshes stay visual only: their convex hulls would fill the jaw opening (B.L.U.S.H scene.py).
PAD_SLAB = 0.008
FIXED_FINGER_Z_END = -0.048     # gripper frame: where the wrist part starts (mesh: the y extent doubles by z -0.032)
MOVING_FINGER_Y_END = -0.028    # jaw frame: where the hinge boss starts (mesh: the z extent jumps by y -0.023)


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    return q


def _mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _mesh_verts(m, d, meshname, body):
    """Vertices of the colliding geom that uses `meshname`, in `body`'s frame."""
    b = m.body(body).id
    for g in range(m.ngeom):
        if (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g]
                and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) == meshname):
            a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
            w = d.geom_xpos[g] + m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T
            return (w - d.xpos[b]) @ d.xmat[b].reshape(3, 3)
    raise KeyError(meshname)


def _slabs(V, axis, start, end, step):
    """Bounding box (lo, hi) of the mesh in each slab along `axis`, from start toward end."""
    out, t, sgn = [], start, np.sign(end - start)
    while (end - t) * sgn > 1e-9:
        t1 = t + sgn * min(step, abs(end - t))
        lo_t, hi_t = min(t, t1), max(t, t1)
        s = V[(V[:, axis] >= lo_t) & (V[:, axis] <= hi_t)]
        if len(s):
            lo, hi = s.min(0), s.max(0)
            lo[axis], hi[axis] = lo_t, hi_t
            out.append((lo, hi))
        t = t1
    return out


@functools.lru_cache(maxsize=None)
def jaw_geometry():
    """Measured once on the stock meshes. Gripper body frame unless it says jaw frame.

    tip_z: the fixed fingertip. fixed / moving: finger slabs (gripper / jaw frame). x_meet: where the
    edges meet. y_blade: the blade plane (the fingertips' stem side minus half a blade). z_blade: the
    blades' centre along the fingers. p_j, R_j: the jaw body's pose in the gripper frame at q = 0.
    """
    if not SO101_XML.exists():
        raise FileNotFoundError(f"{SO101_XML} missing: run ./fetch_so101.sh in sim/ first")
    m = mujoco.MjModel.from_xml_path(str(SO101_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    F = _mesh_verts(m, d, "wrist_roll_follower_so101_v1", "gripper")
    M = _mesh_verts(m, d, "moving_jaw_so101_v1", "moving_jaw_so101_v1")
    tip_z = float(F[:, 2].min())
    fixed = _slabs(F, 2, tip_z, FIXED_FINGER_Z_END, PAD_SLAB)
    moving = _slabs(M, 1, float(M[:, 1].min()), MOVING_FINGER_Y_END, PAD_SLAB)
    jb = m.body("moving_jaw_so101_v1").id
    p_j, R_j = m.body_pos[jb].copy(), _mat(m.body_quat[jb])
    q_meet = np.radians(JAW_MEET_DEG)
    lo, hi = moving[0]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    tip_moving = p_j + corners @ (R_j @ _rotz(q_meet)).T      # moving fingertip at q_meet, gripper frame
    x_face_fixed = float(fixed[0][1][0])                      # fixed fingertip's inner face (faces +x)
    x_face_moving = float(tip_moving[:, 0].min())
    x_meet = 0.5 * (x_face_fixed + x_face_moving)
    y_side = min(float(fixed[0][0][1]), float(tip_moving[:, 1].min()))
    return dict(tip_z=tip_z, fixed=fixed, moving=moving, p_j=p_j, R_j=R_j, q_meet=q_meet, x_meet=x_meet,
                protrude=x_meet - x_face_fixed, y_blade=y_side - BLADE_T / 2, z_blade=tip_z + BLADE_LEN / 2)


def _collider(g):
    g.contype, g.conaffinity = pl.ROBOT_BITS
    g.group, g.priority, g.condim = 3, 1, 3
    g.solref, g.solimp = pl.CONTACT_SOLREF, pl.CONTACT_SOLIMP
    g.friction = (0.5, 0.005, 0.0001)      # PLA and steel on a stalk, guess (unverified)


def arm_spec(gripper_torque=GRIPPER_TORQUE):
    jg = jaw_geometry()
    s = mujoco.MjSpec.from_file(str(SO101_XML))
    for body, mesh in (("gripper", "wrist_roll_follower_so101_v1"), ("moving_jaw_so101_v1", "moving_jaw_so101_v1")):
        for g in s.body(body).geoms:
            if g.meshname == mesh:
                g.contype = g.conaffinity = 0
    box = mujoco.mjtGeom.mjGEOM_BOX
    for body, part, slabs in (("gripper", "fixed", jg["fixed"]), ("moving_jaw_so101_v1", "moving", jg["moving"])):
        for k, (lo, hi) in enumerate(slabs):
            _collider(s.body(body).add_geom(name=f"pad_{part}{k}", type=box, pos=(lo + hi) / 2,
                                            size=(hi - lo) / 2, rgba=(1, 0.5, 0, 0.3)))
    # Both blades are built in the gripper frame as mirror images about x_meet. The moving one is then
    # carried into the jaw frame at q_meet, so the two edges lie flat on each other exactly when they meet.
    fixed_c = np.array([jg["x_meet"] - BLADE_DEPTH / 2, jg["y_blade"], jg["z_blade"]])
    moving_c = np.array([jg["x_meet"] + BLADE_DEPTH / 2, jg["y_blade"], jg["z_blade"]])
    to_jaw = (jg["R_j"] @ _rotz(jg["q_meet"])).T
    for part, body, c, q in (("fixed", "gripper", fixed_c, (1, 0, 0, 0)),
                             ("moving", "moving_jaw_so101_v1", to_jaw @ (moving_c - jg["p_j"]), _quat(to_jaw))):
        b = s.body(body)
        _collider(b.add_geom(name=f"blade_{part}", type=box, pos=c, quat=q, rgba=(1, 0, 0, 0.3),
                             size=(BLADE_DEPTH / 2, BLADE_T_COL / 2, BLADE_LEN / 2)))
        v = b.add_geom(name=f"blade_{part}_vis", type=box, pos=c, quat=q, rgba=BLADE_RGBA,
                       size=(BLADE_DEPTH / 2, BLADE_T / 2, BLADE_LEN / 2))
        v.contype = v.conaffinity = 0
        v.group = 2
    s.joint("gripper").range = GRIPPER_RANGE_RAD
    a = s.actuator("gripper")
    a.ctrlrange = GRIPPER_RANGE_RAD
    a.forcerange = (-gripper_torque, gripper_torque)
    return s


def world_xml(plant_body="", plant_eq=""):
    return f"""<mujoco model="shear_step1">
  <compiler angle="radian"/>
  <option timestep="{TIMESTEP}" integrator="implicitfast" cone="{CONE}" impratio="{IMPRATIO}"/>
  <visual>
    <global azimuth="150" elevation="-20"/>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <worldbody>
    <light name="top" pos="0.3 0 1.4" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="1 1 0.01" rgba="0.75 0.75 0.72 1"
          contype="{pl.FLOOR_BITS[0]}" conaffinity="{pl.FLOOR_BITS[1]}"/>
    {plant_body}
  </worldbody>
  <equality>{plant_eq}</equality>
</mujoco>"""


def _pose_arm(m, d, q):
    for j, v in q.items():
        d.qpos[m.joint("arm/" + j).qposadr[0]] = v
        d.ctrl[m.actuator("arm/" + j).id] = v


@functools.lru_cache(maxsize=None)
def settled_arm(pose=None, jaw_open_deg=JAW_OPEN_DEG):
    """The arm settled on its servos with the jaw open (no plant). Returns joint values by name and the
    gripper body's world position and rotation.

    pose None: step 1's ARM_POSE, with the wrist rolled until the jaw hinge is level. pose = 5 joint values
    (a tuple, from cutpose.CutIK): held as given, so the plant can be built around where the blades really
    end up after the servos sag (step 3)."""
    world = mujoco.MjSpec.from_string(world_xml())
    world.attach(arm_spec(), frame=world.worldbody.add_frame(), prefix="arm/")
    m = world.compile()
    d = mujoco.MjData(m)
    gb = m.body("arm/gripper").id
    target = dict(zip(ARM_JOINTS, ARM_POSE if pose is None else pose)) | {"gripper": np.radians(jaw_open_deg)}
    for k in range(2 if pose is None else 1):     # settle, level the hinge with the wrist roll, settle again
        if k:
            target["wrist_roll"] += np.arctan2(R[2, 1], R[2, 0])    # turns gripper y's lean into gripper x
        _pose_arm(m, d, target)
        for _ in range(round(SETTLE_ARM_S / TIMESTEP)):
            mujoco.mj_step(m, d)
        R = d.xmat[gb].reshape(3, 3)
    q = {j: float(d.qpos[m.joint("arm/" + j).qposadr[0]]) for j in ARM_JOINTS + ("gripper",)}
    return q, d.xpos[gb].copy(), d.xmat[gb].reshape(3, 3).copy()


def build(offset=0.002, shift=0.0, gripper_torque=GRIPPER_TORQUE, leaf_len=pl.LEAF_LEN, pose=None, a=None,
          params=None, stem_r=pl.STEM_R, jaw_open_deg=JAW_OPEN_DEG, clearance=PETIOLE_CLEARANCE, cut="square"):
    """offset: blade plane to stem surface along the petiole, i.e. the stub a cut should leave.
    shift: the plant moved along the fingers, + deeper into the jaw (toward the palm), - out past the tips.

    Step 3 (check_approach.py): pose = 5 joint values to hold the arm at (cutpose.CutIK), a = the petiole's
    world direction (None: along the jaw hinge, step 1), params = the leaf's plant.LeafParams, stem_r = the
    stem's radius. The stem stays vertical; with a off the hinge the petiole crosses the blades at a slant.
    clearance: petiole above the fixed edge (the arm's sag / IK error shows up here on a real approach).
    cut: "square" puts the stem `offset` from the blade plane along the petiole (step 1); "flush" puts it
    `offset` from the blade plane square to the plane (cutpose.py's flush cut)."""
    jg = jaw_geometry()
    P = params or pl.LeafParams(length=leaf_len)
    q, gpos, gmat = settled_arm(None if pose is None else tuple(float(v) for v in pose), jaw_open_deg)
    # The petiole rests just above the fixed edge, centred on the blades, and runs along the jaw hinge
    # (gripper +y) away from the stem, which stands on the blades' -y side.
    a = gmat[:, 1] if a is None else np.asarray(a, float) / np.linalg.norm(a)
    # ⚠️ A petiole crossing the 2 mm collision blade at a slant climbs toward the closing side inside it: lift
    # it by that much, or the fixed blade starts out cut into it (0.6 mm at a 30 deg slant).
    slant = BLADE_T_COL / 2 * abs(a @ gmat[:, 0]) / max(abs(a @ gmat[:, 1]), 1e-6)
    B_g = np.array([jg["x_meet"] + P.petiole_r + clearance + slant, jg["y_blade"], jg["z_blade"] + shift])
    B = gpos + gmat @ B_g
    n = gmat[:, 1]
    if cut == "flush":
        S = B - a * (stem_r + offset) / (a @ n)       # S's distance to the blade plane, along its normal
    else:
        S = B - a * (offset + pl.surface_dist(a, stem_r))
    body, eq, where = pl.plant_xml(S, a, params=P, stem_r=stem_r)
    world = mujoco.MjSpec.from_string(world_xml(body, eq))
    world.attach(arm_spec(gripper_torque), frame=world.worldbody.add_frame(), prefix="arm/")
    # ⚠️ The blades sit on a parent and a child body (gripper, moving jaw), and MuJoCo skips contacts
    # between those, so without an explicit pair the edges pass straight through each other.
    p = world.add_pair(geomname1="arm/blade_fixed", geomname2="arm/blade_moving")
    p.condim = 3
    p.solref, p.solimp = pl.CONTACT_SOLREF, pl.CONTACT_SOLIMP
    p.friction = (0.3, 0.3, 0.005, 0.0001, 0.0001)   # steel on steel, guess (unverified)
    m = world.compile()
    d = mujoco.MjData(m)
    _pose_arm(m, d, q)
    mujoco.mj_forward(m, d)
    info = dict(where, B=B, offset=offset, shift=shift, jaw_open=q["gripper"], q0=d.qpos.copy(),
                ctrl0=d.ctrl.copy(), jaw=jg, gmat=gmat)
    return m, d, info
