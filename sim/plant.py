"""One tomato stem and one leaf for SHEAR sim step 1, and the cut.

    body_xml, eq_xml, where = plant_xml(S, a)   stem vertical through S, leaf from S along a
    cutter = Cutter(m, d, where, cut_force)     call cutter.step() after every mj_step

The stem is a static capsule. The leaf is one free body welded to the stem: a rigid petiole, then a
midrib of hinged capsules that droops under its own weight, with a pair of leaflet plates per segment
and one at the tip. Only the Cutter switches the weld off, so the leaf never snaps or tears off however
hard it is pulled (SHEAR cuts and never snaps, K7 2026-10-27). A hard pull shows up as weld force.

Cut = both blade edges press the petiole, each with at least the cut force, for CUT_DWELL_S. Then the
weld goes off, the leaf stops colliding with anything but the floor (so it falls clear of the jaws),
and the stub left on the stem is drawn. Stub length = distance along the petiole from the stem surface
to the blade plane at the moment of the cut, so it isn't rounded to a segment length.
"""
import mujoco
import numpy as np

G = 9.81

# --- plant sizes. ⚠️ All unverified guesses until measured on real plants (SHEAR Sim Plan, "Measure first") ---
STEM_R = 0.006            # 12 mm stem in the deleafing band, guess (unverified)
STEM_Z = (0.02, 0.45)     # the stretch of stem in the scene, metres above the floor
PETIOLE_R = 0.0035        # 7 mm petiole where it meets the stem, guess (unverified). It sets the cut force.
PETIOLE_LEN = 0.050       # stem surface to the first leaflet pair, guess (unverified)
# Stem surface to leaf tip. ⚠️ Real leaves run ~0.45 m (misina-lab's sim uses that); the SO-101 test
# greenhouse is 0.52 m tall, so real-size or scaled leaves is K7's call before step 2. --leaf-len overrides.
LEAF_LEN = 0.30
LEAF_MASS = 0.025         # whole leaf, guess (unverified)
PETIOLE_MASS_FRAC = 0.1   # share of the leaf mass in the petiole, guess (unverified)
N_MID = 5                 # midrib segments
MID_R = 0.0025            # midrib radius, guess (unverified)
LEAFLET_L, LEAFLET_W, LEAFLET_T = 0.060, 0.030, 0.001   # leaflet plate, guess (unverified)
# Each midrib hinge bends this far under the leaf's own weight with the leaf held straight out: picked
# so the leaf arches like the ones in photos, not measured (unverified). The leaf is built already
# bent this far, so it starts close to hanging still instead of falling from straight.
DROOP_DEG = 8.0
DROOP_ZETA = 0.5          # damping ratio per hinge. ⚠️ 1.0 went unstable (the tip hinge spun up to 16 rad/s)
# ⚠️ Numerical: without it the light tip hinge kept ringing at 0.25 rad/s after 3 s. 1e-5 kg.m^2 is
# about the tip segment's own inertia about its hinge.
MID_ARMATURE = 1e-5

# --- the cut ---
# Force each blade edge has to press the petiole with before it parts. ⚠️ unverified placeholder: measure
# the real petiole with the real blade. What the literature brackets:
#   - tomato and eggplant seedling stems: 0.2-5.5 N per mm of stem diameter, razor blades the lowest
#     (AMSR, abstract only; seedling stems, not petioles), so 1.4-38 N for a 7 mm petiole
#   - a spiral pull-and-cut end-effector on a tomato "lateral branch": 66.3 N cutting plus 32.5 N
#     traction (Sun et al. 2024, Agriculture 14(12):2281; not a blade)
CUT_FORCE_N = 15.0
# The force has to hold this long, like a real stroke: the edges travel through a 7 mm stalk in about
# 60 ms at the jaw's ramped closing speed (unverified). ⚠️ It also keeps the impact spike of the blade
# landing from counting as a cut: at 25 % torque that spike stayed above the cut force for 28 ms.
CUT_DWELL_S = 0.05
# A blade contact counts as the edge only when it pushes within this angle of the jaw's closing axis.
# The flat of the blade rubbing the stalk doesn't cut (unverified).
EDGE_CONE_DEG = 45.0

# Collision bits (contype, conaffinity). Plant parts never collide with each other (the petiole starts
# inside the stem, and leaflets would snag on each other). The robot touches all of them. A cut leaf
# touches only the floor, so it falls clear of the jaws instead of staying clamped.
B_ROBOT, B_PLANT, B_FLOOR, B_FALLEN = 1, 2, 4, 8
ROBOT_BITS = (B_ROBOT, B_ROBOT | B_PLANT | B_FLOOR)
PLANT_BITS = (B_PLANT, B_ROBOT | B_FLOOR)
FLOOR_BITS = (B_FLOOR, B_ROBOT | B_PLANT | B_FALLEN)
FALLEN_BITS = (B_FALLEN, B_FLOOR)
# ⚠️ Near-hard contacts and weld. MuJoCo's default softness scales with the light body's mass: with it
# a 15 N squeeze sinks the blades centimetres into a 25 g leaf, and a pull stretches the weld by
# centimetres (the same trap B.L.U.S.H hit with its 10 g fruit). SOLREF and the weld's SOLIMP come
# from B.L.U.S.H sim/scene.py (commit e91ef0c: FINGER_SOLREF, FINRAY_LOOP_SOLIMP). The contacts get the
# near-hard SOLIMP too: B.L.U.S.H's softer FINGER_SOLIMP let the edges sink 1.1 mm into the petiole.
CONTACT_SOLREF = (0.004, 1.0)            # timeconst = 2 x the 2 ms step, MuJoCo's stability floor
CONTACT_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2)
WELD_SOLREF = (0.004, 1.0)
WELD_SOLIMP = (0.99, 0.999, 0.0001, 0.5, 2)
PLANT_FRICTION = (0.8, 0.005, 0.0001)    # stalk on steel and PLA, guess (unverified)
STEM_RGBA, LEAF_RGBA, LEAFLET_RGBA = "0.35 0.55 0.25 1", "0.30 0.60 0.25 1", "0.25 0.55 0.20 1"
STUB_RGBA = "0.55 0.45 0.25 1"           # cut face colour, so the stub stands out


def _f(v):
    return " ".join(f"{x:.6f}" for x in np.ravel(v))


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    return q


def leaf_frame(a):
    """Leaf axes in the world (columns): x along the petiole, z up, y = z x x (the hinge axis)."""
    x = np.asarray(a, float) / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0]) - x[2] * x
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def surface_dist(a):
    """Distance along the petiole axis from the (vertical) stem axis to the stem surface."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    return STEM_R / np.sqrt(1.0 - a[2] ** 2)


def _plant_geom(rgba):
    return (f'contype="{PLANT_BITS[0]}" conaffinity="{PLANT_BITS[1]}" condim="3" rgba="{rgba}" '
            f'friction="{_f(PLANT_FRICTION)}" solref="{_f(CONTACT_SOLREF)}" solimp="{_f(CONTACT_SOLIMP)}"')


def plant_xml(S, a, leaf_len=LEAF_LEN):
    """S: point on the stem axis where the petiole axis meets it. a: petiole direction, away from the stem."""
    R = leaf_frame(a)
    a = R[:, 0]
    r_s = surface_dist(a)
    stem_pos = np.array([S[0], S[1], 0.5 * (STEM_Z[0] + STEM_Z[1])])
    S_loc = np.asarray(S, float) - stem_pos
    # The stub: hidden (group 5) until a cut, then stretched from the stem axis to the cut plane.
    Rs = np.column_stack([np.cross(a, R[:, 1]), R[:, 1], a])    # capsule axis (local z) along a
    stem = (f'<body name="stem" pos="{_f(stem_pos)}">'
            f'<geom name="stem" type="capsule" fromto="0 0 {STEM_Z[0] - stem_pos[2]:.6f} 0 0 {STEM_Z[1] - stem_pos[2]:.6f}" '
            f'size="{STEM_R}" {_plant_geom(STEM_RGBA)}/>'
            f'<geom name="stub" type="capsule" size="{PETIOLE_R * 1.02:.6f} {r_s / 2:.6f}" pos="{_f(S_loc + a * r_s / 2)}" '
            f'quat="{_f(_quat(Rs))}" contype="0" conaffinity="0" group="5" rgba="{STUB_RGBA}"/></body>')

    seg = (leaf_len - PETIOLE_LEN) / N_MID
    m_pet = PETIOLE_MASS_FRAC * LEAF_MASS
    m_seg = (1.0 - PETIOLE_MASS_FRAC) * LEAF_MASS / N_MID
    # Hinge i carries segments i..N-1. Spring: their weight held straight out bends it DROOP_DEG.
    lever = lambda i: (np.arange(i, N_MID) - i + 0.5) * seg
    inner = ""
    for i in reversed(range(N_MID)):
        moment, inertia = G * m_seg * lever(i).sum(), m_seg * (lever(i) ** 2).sum()
        k = moment / np.radians(DROOP_DEG)
        damping = 2 * DROOP_ZETA * np.sqrt(k * inertia)
        last = i == N_MID - 1
        parts = 4 if last else 3                       # midrib + 2 leaflets (+ the tip leaflet)
        pos = (r_s + PETIOLE_LEN, 0, 0) if i == 0 else (seg, 0, 0)
        # Built bent by DROOP_DEG: ref = that angle, so qpos0 is the bent pose, and the spring
        # (springref 0) still rests at straight.
        bend = np.radians(DROOP_DEG)
        quat = (np.cos(bend / 2), 0.0, np.sin(bend / 2), 0.0)
        y = MID_R + LEAFLET_L / 2
        leaflets = "".join(
            f'<geom name="leaflet{i}{s}" type="box" size="{LEAFLET_W / 2} {LEAFLET_L / 2} {LEAFLET_T / 2}" '
            f'pos="{seg / 2:.6f} {sign * y:.6f} 0" mass="{m_seg / parts:.6g}" {_plant_geom(LEAFLET_RGBA)}/>'
            for s, sign in (("l", 1), ("r", -1)))
        if last:
            leaflets += (f'<geom name="leaflet_tip" type="box" size="{LEAFLET_L / 2} {LEAFLET_W / 2} {LEAFLET_T / 2}" '
                         f'pos="{seg + LEAFLET_L / 2:.6f} 0 0" mass="{m_seg / parts:.6g}" {_plant_geom(LEAFLET_RGBA)}/>')
        inner = (f'<body name="midrib{i}" pos="{_f(pos)}" quat="{_f(quat)}">'
                 f'<joint name="midrib{i}" type="hinge" axis="0 1 0" stiffness="{k:.6g}" damping="{damping:.6g}" '
                 f'armature="{MID_ARMATURE}" ref="{bend:.6f}" springref="0"/>'
                 f'<geom name="midrib{i}" type="capsule" fromto="0 0 0 {seg:.6f} 0 0" size="{MID_R}" '
                 f'mass="{m_seg / parts:.6g}" {_plant_geom(LEAF_RGBA)}/>{leaflets}{inner}</body>')
    leaf = (f'<body name="leaf" pos="{_f(S)}" quat="{_f(_quat(R))}"><freejoint name="leaf"/>'
            f'<geom name="petiole" type="capsule" fromto="0 0 0 {r_s + PETIOLE_LEN:.6f} 0 0" size="{PETIOLE_R}" '
            f'mass="{m_pet:.6g}" {_plant_geom(LEAF_RGBA)}/>{inner}</body>')
    # No relpose: the weld holds the leaf where it was built (MuJoCo uses the qpos0 pose).
    eq = f'<weld name="petiole" body1="stem" body2="leaf" solref="{_f(WELD_SOLREF)}" solimp="{_f(WELD_SOLIMP)}"/>'
    return stem + leaf, eq, dict(S=np.asarray(S, float), a=a, S_loc=S_loc, r_s=r_s)


class Cutter:
    """Watches the blade contacts after every mj_step and cuts the petiole when both edges press hard enough.

    Per step it also keeps: squeeze (the weaker blade's edge force on the petiole), pad_squeeze (same for
    the finger pads), stem_force (any robot part pressing the stem) and weld_force (what the leaf pulls
    on the stem with), plus their peaks. ⚠️ It edits the model when it cuts (collision bits, the stub
    geom); reset() puts the model back.
    """

    def __init__(self, m, d, where, cut_force=CUT_FORCE_N):
        self.m, self.d, self.where, self.cut_force = m, d, where, cut_force
        geom = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.weld = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "petiole")
        self.petiole, self.stem, self.stub_geom = geom("petiole"), geom("stem"), geom("stub")
        self.blades = (geom("arm/blade_fixed"), geom("arm/blade_moving"))
        self.leaf = m.body("leaf").id
        self.gripper = m.body("arm/gripper").id
        root = m.body_rootid[m.geom_bodyid]
        self.leaf_geoms = np.flatnonzero(root == self.leaf)
        self.robot_geoms = set(np.flatnonzero(root == m.body("arm/base").id))
        jaw = m.body("arm/moving_jaw_so101_v1").id
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self.pads = {g: int(m.geom_bodyid[g] == jaw) for g in range(m.ngeom) if names[g].startswith("arm/pad_")}
        self.stub_size0, self.stub_pos0 = m.geom_size[self.stub_geom].copy(), m.geom_pos[self.stub_geom].copy()
        self.cos_edge = np.cos(np.radians(EDGE_CONE_DEG))
        self._f6 = np.zeros(6)
        self.reset()

    def reset(self):
        m, d = self.m, self.d
        m.geom_contype[self.leaf_geoms], m.geom_conaffinity[self.leaf_geoms] = PLANT_BITS
        m.geom_size[self.stub_geom], m.geom_pos[self.stub_geom] = self.stub_size0, self.stub_pos0
        m.geom_group[self.stub_geom] = 5
        d.eq_active[self.weld] = 1
        self.cut_t = self.stub = self.squeeze_at_cut = self._over_t = None
        self.just_cut = False
        self.squeeze = self.pad_squeeze = self.stem_force = self.weld_force = 0.0
        self.peak = dict(squeeze=0.0, pad_squeeze=0.0, stem_force=0.0, weld_force=0.0)
        self.blade_touch = False

    @property
    def attached(self):
        return bool(self.d.eq_active[self.weld])

    def step(self):
        m, d = self.m, self.d
        axis = d.xmat[self.gripper].reshape(3, 3)[:, 0]          # the jaw closes along gripper x
        edge, pad, stem = [0.0, 0.0], [0.0, 0.0], 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            pair = {g1, g2}
            hit_petiole = self.petiole in pair
            hit_stem = self.stem in pair and bool(pair & self.robot_geoms)
            other = g2 if g1 == self.petiole else g1
            if not (hit_stem or (hit_petiole and (other in self.blades or other in self.pads))):
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            fn = self._f6[0]
            if hit_stem:
                stem += fn
            elif other in self.blades:
                self.blade_touch = True
                if abs(c.frame[:3] @ axis) >= self.cos_edge:
                    edge[self.blades.index(other)] += fn
            else:
                pad[self.pads[other]] += fn
        self.squeeze, self.pad_squeeze, self.stem_force = min(edge), min(pad), stem
        self.weld_force = 0.0
        if self.attached:
            rows = np.flatnonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                  & (d.efc_id[:d.nefc] == self.weld))
            self.weld_force = float(np.linalg.norm(d.efc_force[rows[:3]]))
        for k in self.peak:
            self.peak[k] = max(self.peak[k], getattr(self, k))
        self.just_cut = False
        if self.attached and self.squeeze >= self.cut_force:
            self._over_t = d.time if self._over_t is None else self._over_t
            if d.time - self._over_t >= CUT_DWELL_S - 1e-9:
                self._cut()
        else:
            self._over_t = None

    def _cut(self):
        m, d, w = self.m, self.d, self.where
        S, a = d.xpos[self.leaf], d.xmat[self.leaf].reshape(3, 3)[:, 0]
        plane = np.mean([d.geom_xpos[b] for b in self.blades], axis=0)   # both blades share one plane
        self.stub = float((plane - S) @ a - surface_dist(a))
        self.cut_t, self.squeeze_at_cut, self.just_cut = d.time, self.squeeze, True
        d.eq_active[self.weld] = 0
        m.geom_contype[self.leaf_geoms], m.geom_conaffinity[self.leaf_geoms] = FALLEN_BITS
        length = w["r_s"] + max(self.stub, 0.0)
        m.geom_size[self.stub_geom][1] = length / 2
        m.geom_pos[self.stub_geom] = w["S_loc"] + w["a"] * length / 2
        m.geom_group[self.stub_geom] = 2
