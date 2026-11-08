"""RRT-Connect route planning for the SO-101 arm, with a list of what it must not touch (K7 2026-11-01: plan every
arm move with RRT, so it always steers around what it shouldn't damage).

    chk = Checker(m, d, hard=geoms, soft=geoms)        a kinematic copy of the scene as it stands now
    path = plan(chk, q_start, q_goal, rng)             joint configs (5 arm joints, rad) from start to goal, or None
    chk.hits(q), chk.path_ok(path)                     what the arm touches at q / along a path

What RRT does and what IK does: IK (cutpose.py) finds the joint angles that put the blades on the leaf stalk, the
goal. RRT finds a route there through joint space that keeps the arm clear of everything on the list: it grows two
trees of collision-free arm poses, one from the start and one from the goal, until they meet, then the route is
shortened by cutting corners wherever the straight joint-space line between two of its points is clear too.

What the arm keeps clear of:
- **hard**: never touched, never closer than the clearance (stems, pots, the greenhouse frame, the floor). Damage
  there costs a plant.
- **firm**: fruit and truss stalks. Always avoided on routes (SOFT_CLEARANCE); a finger pad may rest on them only
  where a caller asks for firm=False (SHEAR's short slide onto the petiole). ⚠️ When the fallback route could brush
  them too, it pushed a truss stalk with 8 N on the way to a leaf (2026-11-03).
- **soft**: the stiff parts of the leaves the rule keeps. Avoided when a route exists that clears them; when none
  does, plan() can be asked again with soft=False and the route may brush them (a hand reaching into a canopy does).
- **blade_guard**: things the BLADES never come near, even with soft=False (truss stalks: a blade on one can cut
  the truss off; a finger pad resting on it doesn't).

The checker moves only the arm: the plant stands where it was when the Checker was made (settled), kinematically.
Leaves pushed in the physics move afterwards; that's the dry run's job to judge.
"""
import copy

import mujoco
import numpy as np

import cutpose as cp

CLEARANCE = 0.004        # m kept from hard obstacles on a route (B.L.U.S.H expert.CLEARANCE_M)
BLADE_CLEARANCE = 0.002  # m the blades keep from blade_guard things (guess, unverified)
SOFT_CLEARANCE = 0.002   # m kept from kept leaves (guess, unverified)
STEP = np.radians(3.0)   # joint-space resolution of collision checks along a segment (largest joint)
EXTEND = np.radians(12.0)    # RRT step: how far a tree grows toward a sample (design choice)
MAX_ITERS = 4000         # per plan() call
SHORTCUT_TRIES = 150


class Checker:
    def __init__(self, m, d, hard=(), soft=(), ignore=(), blade_guard=(), firm=(), prefix="arm/"):
        self.m = copy.deepcopy(m)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[:] = d.qpos
        self.adr = np.array([self.m.joint(prefix + j).qposadr[0] for j in cp.ARM5])
        body_arm = np.array([self.m.body(b).name.startswith(prefix) for b in range(self.m.nbody)])
        self.is_arm = body_arm[self.m.geom_bodyid]
        for g in np.flatnonzero(self.is_arm & (self.m.geom_contype > 0)):
            self.m.geom_margin[g] = CLEARANCE           # report contacts this close, not only touching
        self.hard, self.soft, self.ignore, self.guard = set(hard), set(soft), set(ignore), set(blade_guard)
        self.firm = set(firm)
        self.blades = {g for g in range(self.m.ngeom) if self.is_arm[g] and (self.m.geom(g).name or "").startswith(
            prefix + "blade_") and self.m.geom_contype[g]}

    def hits(self, q, soft=True, clearance=CLEARANCE, soft_clearance=SOFT_CLEARANCE, ignore=(), firm=True,
             firm_clearance=SOFT_CLEARANCE):
        """Names of what the arm comes within the clearance of at joint config q (5, rad)."""
        self.d.qpos[self.adr] = q
        mujoco.mj_fwdPosition(self.m, self.d)
        out = set()
        for c in self.d.contact[:self.d.ncon]:
            a, b = self.is_arm[c.geom1], self.is_arm[c.geom2]
            if a == b:
                continue
            other = c.geom2 if a else c.geom1
            if other in self.ignore or other in ignore:
                continue
            mine = c.geom1 if a else c.geom2
            if other in self.guard and mine in self.blades and c.dist < BLADE_CLEARANCE:
                out.add("blade on " + (self.m.geom(other).name or f"geom{other}"))
            elif other in self.hard and c.dist < clearance:
                out.add(self.m.geom(other).name or f"geom{other}")
            elif firm and other in self.firm and c.dist < firm_clearance:
                out.add(self.m.geom(other).name or f"geom{other}")
            elif soft and other in self.soft and c.dist < soft_clearance:
                out.add(self.m.geom(other).name or f"geom{other}")
        return out

    def free(self, q, **kw):
        return not self.hits(q, **kw)

    def segment_free(self, q0, q1, **kw):
        n = max(2, int(np.ceil(np.abs(np.asarray(q1) - q0).max() / STEP)) + 1)
        return all(self.free(q0 + (q1 - q0) * s, **kw) for s in np.linspace(0, 1, n)[1:])

    def path_ok(self, path, **kw):
        return all(self.segment_free(a, b, **kw) for a, b in zip(path[:-1], path[1:]))


def _steer(q_from, q_to, step=EXTEND):
    d = q_to - q_from
    n = np.abs(d).max()
    return q_to.copy() if n <= step else q_from + d * (step / n)


def _connect(chk, tree, parents, q, kw):
    """Grow `tree` toward q until it reaches q (returns 'reached') or hits something ('trapped')."""
    while True:
        i = int(np.argmin(np.abs(np.array(tree) - q).max(1)))
        q_new = _steer(tree[i], q)
        if not chk.segment_free(tree[i], q_new, **kw):
            return "trapped", None
        tree.append(q_new)
        parents.append(i)
        if np.abs(q_new - q).max() < 1e-9:
            return "reached", len(tree) - 1


def _trace(tree, parents, i):
    out = []
    while i is not None:
        out.append(tree[i])
        i = parents[i]
    return out[::-1]


def plan(chk, q_start, q_goal, rng, lo=None, hi=None, iters=MAX_ITERS, **kw):
    """RRT-Connect from q_start to q_goal (5 joints, rad) inside [lo, hi]. Returns the shortcut path, or None.
    kw goes to Checker.hits (soft=False lets the route brush kept leaves)."""
    q_start, q_goal = np.asarray(q_start, float), np.asarray(q_goal, float)
    lo = cp.follower_limits()[0] if lo is None else lo
    hi = cp.follower_limits()[1] if hi is None else hi
    if not chk.free(q_start, **kw) or not chk.free(q_goal, **kw):
        return None
    if chk.segment_free(q_start, q_goal, **kw):
        return [q_start, q_goal]
    A, pa, B, pb = [q_start], [None], [q_goal], [None]
    for _ in range(iters):
        q_rand = rng.uniform(lo, hi)
        i = int(np.argmin(np.abs(np.array(A) - q_rand).max(1)))
        q_new = _steer(A[i], q_rand)
        if chk.segment_free(A[i], q_new, **kw):
            A.append(q_new)
            pa.append(i)
            status, j = _connect(chk, B, pb, q_new, kw)
            if status == "reached":
                pathA, pathB = _trace(A, pa, len(A) - 1), _trace(B, pb, j)
                path = pathA + pathB[::-1][1:]
                if A[0] is not q_start:                  # trees were swapped an odd number of times
                    path = path[::-1]
                return shortcut(chk, path, rng, **kw)
        A, pa, B, pb = B, pb, A, pa                      # grow the other tree next time
    return None


def shortcut(chk, path, rng, tries=SHORTCUT_TRIES, **kw):
    """Cut corners: join two random points of the path straight when that segment is clear."""
    path = [np.asarray(q, float) for q in path]
    for _ in range(tries):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if chk.segment_free(path[i], path[j], **kw):
            path = path[:i + 1] + path[j:]
    return path
