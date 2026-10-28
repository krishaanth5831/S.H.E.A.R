"""Position IK for the SO-101's fingertips (damped least squares on the model's own kinematics).

    q, err = reach(m, d, target)           joint values (by name) that put the fingertips at target

Moves shoulder_pan, shoulder_lift, elbow_flex and wrist_flex, inside the model's joint ranges. The wrist
roll and the jaw stay where they are. It works on a copy of d, so the scene isn't touched.
Step 3 adds the jaw's orientation across the petiole; this is the position part.
"""
import mujoco
import numpy as np

import scene as sc

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
DAMPING = 1e-3          # DLS lambda^2 (m^2), standard small value (unverified for this arm)
STEP_MAX = 0.2          # rad per iteration
ITERS = 300
TOL = 5e-4              # m


def tip_local():
    """Fingertip point (where the blade edges meet, at the tips) in the gripper body frame."""
    jg = sc.jaw_geometry()
    return np.array([jg["x_meet"], jg["y_blade"], jg["tip_z"]])


def tip_world(m, d, prefix="arm/"):
    g = m.body(prefix + "gripper").id
    return d.xpos[g] + d.xmat[g].reshape(3, 3) @ tip_local()


def reach(m, d, target, prefix="arm/", q0=None):
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos if q0 is None else q0
    adr = [m.joint(prefix + j).qposadr[0] for j in JOINTS]
    dof = [m.joint(prefix + j).dofadr[0] for j in JOINTS]
    lo = np.array([m.joint(prefix + j).range[0] for j in JOINTS])
    hi = np.array([m.joint(prefix + j).range[1] for j in JOINTS])
    g = m.body(prefix + "gripper").id
    jac = np.zeros((3, m.nv))
    target = np.asarray(target, float)
    for _ in range(ITERS):
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_comPos(m, d2)
        p = tip_world(m, d2, prefix)
        e = target - p
        if np.linalg.norm(e) < TOL:
            break
        mujoco.mj_jac(m, d2, jac, None, p, g)
        J = jac[:, dof]
        dq = J.T @ np.linalg.solve(J @ J.T + DAMPING * np.eye(3), e)
        dq *= min(1.0, STEP_MAX / max(np.abs(dq).max(), 1e-12))
        d2.qpos[adr] = np.clip(d2.qpos[adr] + dq, lo, hi)
    mujoco.mj_kinematics(m, d2)
    err = float(np.linalg.norm(target - tip_world(m, d2, prefix)))
    return {j: float(d2.qpos[a]) for j, a in zip(JOINTS, adr)}, err
