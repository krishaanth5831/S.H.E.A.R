"""Cut poses: where the gripper has to be to cut one leaf, and the 5-joint IK that puts it there.

    ik = CutIK(base, yaw)                       the arm's base in the world (layout.py's by default)
    sols = ik.solve(S, a, stem_r, offset=0.002, petiole_r=0.0015)     every distinct pose found, best first
    s = sols[0]; s.q (5 joint angles, rad), s.lean_deg, s.pitch_deg, s.twist_deg, s.fingers (world unit vector)

S: the point on the stem axis where the petiole axis meets it, a: the petiole's direction away from the stem
(both in stemgen's `where[i]`). The stem is vertical (the sim's stems are).

The blades sit as in step 1 (scene.build), on both faces of the fingertips (scene.BLADE_FACES): the pair next to
the stem cuts, so the jaw hinge (gripper +y) points away from the stem for the -y pair (face +1) and toward it
for the +y pair (face -1). solve() tries both. The petiole axis passes just above
the fixed edge, at the blades' centre along the fingers. Two ways to hold the blade (`mode`):

- **"flush"** (default): the blade plane parallel to the stem, `offset` off its surface. That's the grower's
  flush cut ("tot tegen de stam", research note), and the only way to cut close: a blade square to a petiole
  that rises 30 deg leans 30 deg into the stem, and its far end hits the stem before a 2 mm stub (sim step 3,
  2026-10-29: check_approach.py's lean group). Flush asks for the hinge level (lean 0) and the fingers level,
  so the jaw closes along the stem as in step 1: the blade is flush with the fingertips only, and further back
  the stock finger stands proud of it. Turning the hinge about the vertical (twist, off the petiole's own
  direction) only slants the cut face, so it's traded first.
- **"square"**: the blade plane square to the petiole (step 1's cut, where the petiole was level).

⚠️ The SO-101 has 5 arm joints and its fingers lie in the pan plane. Hinge level and fingers level means the
fingers point straight out from the base and the hinge is square to that, so a flush cut fits a petiole that
points across the arm's reach (along the row, seen from in front of the plant). A petiole pointing at the arm or
away from it needs a lean or a twist, or the base moved along the aisle (a SPINE move) until the petiole points
across its reach. The fingers can't point straight down or up at the work band instead: at 0.27 m high and
0.18 m out the follower's joint ranges don't reach it (FK scan 2026-10-29: fingers down only below z 0.10,
up only from z ~0.30). "square" uses all 5 joints (cut point + hinge = 5), so the approach is fixed there.

The LM loop and the sampled warm starts are copied from B.L.U.S.H sim/expert.py ArmIK (commit cec86fb); the
residuals are SHEAR's own. Joint limits are the real follower's calibrated ranges (arm_map.py), not the MJCF's.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

import arm_map
import layout as lo
import plant as pl
import scene as sc

ARM5 = sc.ARM_JOINTS        # shoulder_pan .. wrist_roll
UP = np.array([0.0, 0.0, 1.0])
# Residual weights, m per unit. Position is 1. W_LEAN on the hinge's vertical part (flush: the blade leaning
# into the stem), W_DIR on the hinge vs the petiole's direction (flush: its level part, so twist; square: the
# whole tilt). 0.01: a 1 mm miss costs as much as ~6 deg of twist, so the solver keeps the cut point and gives
# up twist first; 0.05 holds lean 5x harder than twist (design choices).
W_LEAN, W_DIR = 0.05, 0.01
# Flush only: the fingers level (gripper -z square to the stem). ⚠️ The blade is flush with the fingertips only:
# further back the stock fixed finger stands proud of the blade plane, so fingers running along the stem put the
# whole finger against it (sim step 3: fingers 62 deg up pressed fixed pads 3-7 up to 3.6 mm into the stem).
# Held harder than twist, softer than lean (design choice).
W_LEVEL = 0.02
# solve(fingers=...): the commanded finger direction, between the two, so the arm holds the asked-for approach
# angle, keeps the blade level and gives up twist (how check_approach.py sweeps the approach; design choice).
W_FINGERS = 0.03
POS_OK = 0.002              # m, a pose counts if its cut point lands this close (B.L.U.S.H ArmIK's 2 mm)
# The petiole sits this far above the fixed edge when the jaw starts closing. Step 1 builds it 0.1 mm off
# (PETIOLE_CLEARANCE); a moving arm sags and overshoots by more, so an approach leaves a margin (unverified).
APPROACH_CLEARANCE = 0.001
SAMPLES = 8000              # FK samples for warm starts (B.L.U.S.H ArmIK)
STARTS = 24                 # nearest samples tried per solve (12 missed the fingers-vertical flush poses)
SAME_POSE_DEG = 5.0         # two solutions whose fingers point within this are the same pose


@dataclass
class CutPose:
    q: np.ndarray           # 5 arm joints, rad
    pos_err: float          # m: cut point off the petiole axis, plus the blade plane off its asked-for distance
    lean_deg: float         # hinge above level (flush wants 0: the blade plane leaning into the stem)
    pitch_deg: float        # fingers above level (flush wants ~0: the fingers crossing the stem, not along it)
    twist_deg: float        # hinge's level part vs the petiole's level direction (slants the cut face)
    tilt_deg: float         # hinge vs the petiole (0 = blade square to the petiole)
    fingers: np.ndarray     # world direction the fingers point (gripper -z)
    R: np.ndarray           # gripper rotation (world)
    mode: str
    face: int = 1           # which blade pair is next to the stem (scene.BLADE_PAIRS)

    def q_dict(self):
        return dict(zip(ARM5, map(float, self.q)))


def tool_point(petiole_r, clearance=APPROACH_CLEARANCE, face=1):
    """Gripper-frame point the petiole axis passes through at the cut: scene.build's B_g."""
    jg = sc.jaw_geometry()
    return np.array([jg["x_meet"] + petiole_r + clearance, sc.blade_y(face), jg["z_blade"]])


def level(v):
    v = np.array([v[0], v[1], 0.0])
    return v / np.linalg.norm(v)


def follower_limits():
    """Real follower's joint ranges (rad) for the 5 arm joints. ⚠️ Falls back to the MJCF's if the follower
    calibration file isn't on this machine (it lives in ~/.cache/huggingface, from lerobot calibrate)."""
    try:
        lo_, hi_ = arm_map.FollowerMap().rad_limits()
        return lo_[:5], hi_[:5], "follower_white.json"
    except FileNotFoundError:
        m = mujoco.MjModel.from_xml_path(str(sc.SO101_XML))
        r = np.array([m.joint(j).range for j in ARM5])
        return r[:, 0], r[:, 1], "MJCF ranges (follower calibration missing)"


class CutIK:
    def __init__(self, base=None, yaw=lo.ARM_YAW, seed=0):
        base = np.array([lo.ARM_X, lo.AISLE_Y, 0.0]) if base is None else np.asarray(base, float)
        spec = mujoco.MjSpec()
        spec.attach(sc.arm_spec(), frame=spec.worldbody.add_frame(
            pos=base, quat=(np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))), prefix="arm/")
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.adr = np.array([self.m.joint("arm/" + j).qposadr[0] for j in ARM5])
        self.G = self.m.body("arm/gripper").id
        self.lo, self.hi, self.limits_from = follower_limits()
        self.base, self.yaw = base, yaw
        rng = np.random.default_rng(seed)
        self.Q = rng.uniform(self.lo, self.hi, (SAMPLES, 5))
        P, R = zip(*(self.fk(x) for x in self.Q))
        self.P, self.R = np.array(P), np.array(R)

    def fk(self, x):
        self.d.qpos[self.adr] = x
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.G].copy(), self.d.xmat[self.G].reshape(3, 3).copy()

    def _res(self, x, g):
        """g: the goal (dict from _goal)."""
        o, R = self.fk(x)
        d = o + R @ g["tool"] - g["S"]
        along = d @ g["a"]
        n = g["face"] * R[:, 1]                  # blade plane normal, away from the stem
        if g["mode"] == "flush":
            r = np.r_[d - along * g["a"], d @ n - g["dist"], W_LEAN * n[2], W_LEVEL * R[2, 2], W_DIR * (n - g["a_lvl"])]
        else:
            # W_LEVEL breaks the tie on a petiole across the reach, where square leaves the pitch free
            r = np.r_[d - along * g["a"], along - g["dist"], W_DIR * (n - g["a"]), 0.1 * W_LEVEL * R[2, 2]]
        return r if g["f"] is None else np.r_[r, W_FINGERS * (-R[:, 2] - g["f"])]

    def _lm(self, x, g, iters=100):
        """Levenberg-Marquardt inside the joint limits (B.L.U.S.H expert.ArmIK._lm, commit cec86fb)."""
        e = self._res(x, g)
        lam = 1e-3
        for _ in range(iters):
            if np.linalg.norm(e) < 1e-5:
                break
            J = np.empty((len(e), 5))
            for k in range(5):
                dx = np.zeros(5)
                dx[k] = 1e-6
                J[:, k] = (self._res(x + dx, g) - e) / 1e-6
            H, gr = J.T @ J, J.T @ e
            xn = np.clip(x - np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), gr), self.lo, self.hi)
            en = self._res(xn, g)
            if np.linalg.norm(en) < np.linalg.norm(e):
                x, e, lam = xn, en, max(lam / 3, 1e-9)
            else:
                lam *= 4
                if lam > 1e6:
                    break
        return x

    def pose(self, x, g):
        o, R = self.fk(x)
        d = o + R @ g["tool"] - g["S"]
        along = d @ g["a"]
        n = g["face"] * R[:, 1]
        miss = np.linalg.norm(d - along * g["a"])
        miss += abs((d @ n if g["mode"] == "flush" else along) - g["dist"])
        nl = np.array([n[0], n[1], 0.0])
        twist = np.degrees(np.arccos(np.clip(nl @ g["a_lvl"] / max(np.linalg.norm(nl), 1e-9), -1, 1)))
        return CutPose(q=np.asarray(x, float), pos_err=float(miss), lean_deg=float(np.degrees(np.arcsin(np.clip(n[2], -1, 1)))),
                       pitch_deg=float(np.degrees(np.arcsin(np.clip(-R[2, 2], -1, 1)))),
                       twist_deg=float(twist), tilt_deg=float(np.degrees(np.arccos(np.clip(n @ g["a"], -1, 1)))),
                       fingers=-R[:, 2], R=R, mode=g["mode"], face=g["face"])

    def solve(self, S, a, stem_r, offset=0.002, petiole_r=pl.PETIOLE_R, mode="flush", clearance=APPROACH_CLEARANCE,
              x0=None, fingers=None):
        """Every distinct pose (by finger direction) within POS_OK, the best blade first (flush: least lean,
        then least twist; square: least tilt).

        offset: the blade plane off the stem surface (flush: square to the stem; square: along the petiole), i.e.
        the stub to leave. fingers: a world direction to hold the fingers along (an approach angle)."""
        a = np.asarray(a, float) / np.linalg.norm(a)
        S = np.asarray(S, float)
        dist = stem_r + offset if mode == "flush" else pl.surface_dist(a, stem_r) + offset
        rad = np.radians
        key = ((lambda p: W_LEAN * abs(rad(p.lean_deg)) + W_LEVEL * abs(rad(p.pitch_deg)) + W_DIR * rad(p.twist_deg))
               if mode == "flush" else (lambda p: p.tilt_deg))
        out = []
        f = None if fingers is None else np.asarray(fingers, float) / np.linalg.norm(fingers)
        for face in list(sc.BLADE_PAIRS)[:sc.BLADE_FACES]:      # each blade pair, as the one next to the stem
            g = dict(mode=mode, S=S, a=a, a_lvl=level(a), dist=dist, tool=tool_point(petiole_r, clearance, face),
                     f=f, face=face)
            T = S + a * (pl.surface_dist(a, stem_r) + offset)          # rough cut point, for the warm starts
            hinge = face * (g["a_lvl"] if mode == "flush" else a)
            tips = self.P + np.einsum("nij,j->ni", self.R, g["tool"])
            cost = np.linalg.norm(tips - T, axis=1) + W_DIR * np.linalg.norm(self.R[:, :, 1] - hinge, axis=1)
            if f is not None:
                cost += W_FINGERS * np.linalg.norm(-self.R[:, :, 2] - f, axis=1)
            starts = ([np.asarray(x0, float)] if x0 is not None else []) + [self.Q[i] for i in np.argsort(cost)[:STARTS]]
            mine = []
            for x in starts:
                p = self.pose(self._lm(x.copy(), g), g)
                # ⚠️ The petiole's axis is a whole line: the flush residual is also met with the cut point on its
                # far side, through the stem, with the blades facing away from the stem. Never a cut.
                if p.pos_err > POS_OK or face * p.R[:, 1] @ (a if mode == "square" else g["a_lvl"]) <= 0:
                    continue
                same = [k for k, o in enumerate(mine)
                        if np.degrees(np.arccos(np.clip(o.fingers @ p.fingers, -1, 1))) < SAME_POSE_DEG]
                if not same:
                    mine.append(p)
                elif key(p) < key(mine[same[0]]):
                    mine[same[0]] = p
            out += mine
        return sorted(out, key=key)
