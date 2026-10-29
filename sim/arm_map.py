"""lerobot units (use_degrees=false) <-> MuJoCo joint angles, the way the real follower does it.

Copied unchanged from B.L.U.S.H sim/arm_map.py (commit 5c12eea). SHEAR uses it for the real follower's joint
limits (rad_limits): the cut-pose IK (cutpose.py) must stay inside what the real arm reaches, not the MJCF ranges.

Real teleop with use_degrees=false: the leader sends range-normalised values (arm joints -100..100,
gripper 0..100). The follower turns them into servo ticks through ITS OWN calibration
(follower_white.json) and drives there. The sim copies that chain, so a sim episode and a real
episode with the same leader motion carry the same numbers:

    action (leader norm) --follower calib--> ticks --> joint angle (rad) --> MuJoCo position ctrl
    MuJoCo qpos (rad) --> ticks --follower calib--> observation.state (norm)

Ticks -> angle uses lerobot's own DEGREES formula (FeetechMotorsBus._normalize, use_degrees=true):
deg = (ticks - mid) * 360 / 4095 with mid = (range_min + range_max) / 2. That puts angle 0 at the
middle of the calibrated range, the same zero as so101_new_calib.xml ("virtual zero = middle of
the joint range"). so101-il-dev-toolkit drives a MuJoCo follower with this formula and sign +1 on
all five arm joints.

SIGN = +1 on every joint and these zeros were checked on our arms by K7 (2026-10-20): with
`sim/scripts/teleop_sim.sh` every leader joint moved the sim joint the same way, and 3 poses
matched the real follower by eye. (The real follower's wrist_flex trails the leader by a few ticks,
its own calibration; accepted.)
"""
import hashlib
import json
import pathlib

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FOLLOWER_CALIB = (pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
                  / "follower_white.json")
MAX_RES = 4095                  # STS3215 resolution - 1, as in lerobot's DEGREES formula
SIGN = np.ones(6)               # checked with the leader 2026-10-20, see module docstring

# The gripper is range-normalised 0 (closed) .. 100 (open). Its closed end is the jaw angle where
# the fingers meet: -10 deg in so101_new_calib (the URDF lower limit; stock tips touch there,
# measured in hardware/end_effector/finray). The fin-ray fingers keep the stock outline, so their
# tips stop ~3 mm apart at the same angle.
GRIPPER_CLOSED_DEG = -10.0


class FollowerMap:
    def __init__(self, calib_path=FOLLOWER_CALIB):
        self.path = pathlib.Path(calib_path)
        raw = self.path.read_bytes()
        # Stored in every dataset's metadata. ⚠️ Recalibrating overwrites follower_white.json
        # without a prompt, which would silently move sim data away from real data.
        self.sha256 = hashlib.sha256(raw).hexdigest()
        cal = json.loads(raw)
        self.lo = np.array([cal[j]["range_min"] for j in JOINTS], float)
        self.hi = np.array([cal[j]["range_max"] for j in JOINTS], float)
        self.mid = (self.lo + self.hi) / 2

    def meta(self):
        return {"follower_calibration": str(self.path), "follower_calibration_sha256": self.sha256}

    def norm_to_ticks(self, norm):
        n = np.asarray(norm, float)
        frac = np.empty(6)
        frac[:5] = (np.clip(n[:5], -100, 100) + 100) / 200   # lerobot _unnormalize clamps to the range
        frac[5] = np.clip(n[5], 0, 100) / 100
        return self.lo + frac * (self.hi - self.lo)

    def ticks_to_norm(self, ticks):
        t = np.clip(np.asarray(ticks, float), self.lo, self.hi)   # lerobot _normalize clamps raw ticks
        frac = (t - self.lo) / (self.hi - self.lo)
        out = frac * 200 - 100
        out[5] = frac[5] * 100
        return out

    def ticks_to_rad(self, ticks):
        t = np.asarray(ticks, float)
        deg = (t - self.mid) * 360 / MAX_RES
        deg[5] = GRIPPER_CLOSED_DEG + (t[5] - self.lo[5]) * 360 / MAX_RES
        return SIGN * np.radians(deg)

    def rad_to_ticks(self, q):
        deg = np.degrees(np.asarray(q, float) * SIGN)
        t = self.mid + deg * MAX_RES / 360
        t[5] = self.lo[5] + (deg[5] - GRIPPER_CLOSED_DEG) * MAX_RES / 360
        return t

    def norm_to_rad(self, norm):
        return self.ticks_to_rad(self.norm_to_ticks(norm))

    def rad_to_norm(self, q):
        return self.ticks_to_norm(self.rad_to_ticks(q))

    def rad_limits(self):
        """Joint angles (rad) at the calibrated range ends: what the real follower can reach."""
        a, b = self.ticks_to_rad(self.lo), self.ticks_to_rad(self.hi)
        return np.minimum(a, b), np.maximum(a, b)
