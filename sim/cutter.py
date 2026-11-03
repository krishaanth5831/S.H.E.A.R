"""Step 5 of the SHEAR sim plan: the scripted cutter. One episode is one deleafing pass along the row.

    scn = Scene(seeds, turns)                   the row (layout.py, bending petioles), settled with the arm at START
    out = run_pass(scn, rng, frame_cb)          start pose -> every leaf the rule cuts, plant by plant -> start pose
    out["cuts"], out["skipped"], out["verdict"] the judge, per cut and for the pass

The pass (K7 2026-11-03): it starts and ends at the robot's start pose (B.L.U.S.H's START_NORM, folded beside
its base), and works the three plants in front of it in order, p0, p1, p2. On each plant it cuts the leaves the
rule picks (rule/select_leaves.py), oldest first. A leaf it can't plan a clean cut for is skipped and logged.
Skipping is the cheap mistake: a person follows up (research note).

One cut:
1. **to_leaf**: an RRT route (rrt.py) from wherever the arm is to the pre-cut pose, SLIDE_M short of the cut along
   the fingers. K7 2026-11-01: every arm move is planned with RRT, around what it mustn't damage. It keeps
   CLEARANCE from stems, pots and the frame (never touched) and SOFT_CLEARANCE from fruit and truss stalks (always
   avoided on a route), and from the stiff parts (petiole, midrib) of the leaves the rule keeps. Leaflets give way
   and may be pushed. If no route clears the kept leaves, one that may brush them is taken, and the cut says so. The blades never come within
   2 mm of a truss stalk: a blade there can cut the truss off; a finger pad resting on it doesn't.
   ⚠️ What each of those choices fixed (2026-11-01): with leaflets as obstacles, no route on a dense dwarf
   cleared them; with truss stalks never-touch for the whole arm, a truss drooping across the cut point
   blocked most cuts; with fruit never-touch, no plant in 40 had a slide in that missed every fruit. A fruit
   nudged on its pedicel isn't damaged (B.L.U.S.H lets the fingers push a neighbour 15 mm).
2. **slide_in**: straight along the fingers onto the petiole, the gripper held in the cut pose (cutpose.py,
   flush, 2 mm stub). The petiole slides between the open blades from the tips.
3. **close**: the jaw closes like the real servo; the blades cut the petiole and the leaf drops.
4. **slide_out**: straight back out, the jaw shut.
Then the next cut's route starts from there; after the last, an RRT route back to START.

IK and RRT do different jobs: IK finds where the joints must end up (the cut pose and the slide's poses); RRT
finds a route there that steers around things.

The judge, per cut: the right leaf came off, no other leaf did, the stub is flush (STUB_OK), the arm never
pressed a stem, a truss stalk or the frame (FORCE_LIMIT), and no blade touched a truss stalk. Fruit pushed
(FRUIT_PUSH_N) and kept leaves brushed are logged.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

import arm_map
import cutpose as cp
import layout as lo
import plant as pl
import reach
import rrt
import scene as sc
import stemgen as sg

FPS = 30                          # config/hardware.env: every dataset records at 30 fps (as B.L.U.S.H)
# The robot's start pose, the same as B.L.U.S.H's episodes (sim/scene.py START_NORM, commit 41da64d), in lerobot
# units: folded low beside its base, turned away from the row. Clear of everything in this greenhouse too (checked
# 2026-11-03). Every pass starts and ends here (K7 2026-11-03).
START_NORM = (-90.0, -40.0, 90.0, -40.0, 0.0, 50.0)
# ⚠️ Until 2026-11-03 the episodes started from a raised pose above the aisle (0, -90, -5, 90 deg): K7 saw they
# didn't start from the robot's start position.
OFFSET = 0.002                    # flush stub asked for (check_approach.py: 2 mm is the closest clean cut)
# The petiole's axis this far above the near fixed blade's edge plus its radius, at the cut pose. ⚠️ At 1 mm
# (cutpose.APPROACH_CLEARANCE) the far blade pair pushed the petiole down first; it pivots at the stem, so it hardly
# moved at the near pair, the near fixed edge never touched it, and the far pair cut a 7-11 mm stub (2026-11-01).
# The arm now lands within ~0.2 mm (the sag correction), so it aims the petiole just onto the near edge.
CUT_CLEARANCE = 0.0002
JAW_APPROACH_DEG = 0.0            # 12.4 mm between the blade edges (measured on the model): a 3.6 mm petiole + aim
JAW_CLOSED = sc.GRIPPER_RANGE_RAD[0]
SLIDE_M = 0.04                    # pre-cut pose this far back along the fingers (B.L.U.S.H pre-grasp 4-7 cm)
SLIDE_STEPS = 9
SLIDE_DRIFT = (0.0015, 4.5)       # extra m / deg the slide may drift off its line at its start (exact at the cut)
SLIDE_SPEED = 0.02                # m/s along the fingers, in and out (slow and careful; guess, unverified)
JOINT_SPEED = np.radians(40.0)    # average speed on the largest joint along an RRT route (guess, unverified)
HOLD_S, CLOSE_S, CUT_WAIT_S = 0.5, 0.4, 1.2
SLIDE_CLEARANCE = 0.0005          # m from stems / stalks on the slide: the blade stands 2 mm off the stem at the cut
# Integral correction of the servo sag, per frame (B.L.U.S.H expert.SAG_KI 0.15; SAG_MAX_NORM 6 ~ 10 deg).
# ⚠️ Capped at 4.6 deg it ran out: the shoulder sagged ~6.8 deg reaching a low leaf, 4 mm at the blades (2026-11-01).
SAG_KI, SAG_MAX = 0.15, np.radians(10.0)
NEAR_LEAF_M = 0.08                # leaves this close to the cut get a Cutter (a wrong-leaf cut shows)
FORCE_LIMIT = {"stem": 0.5, "truss stalk": 1.0, "frame": 1.0, "pot": 1.0}   # N, peak; over = damage (guesses)
FRUIT_PUSH_N = 2.0                # N on a fruit counts as a knock, logged (guess)
STUB_OK = (0.0015, 0.0035)        # m, flush stub accepted
SETTLE_S = 3.0

HARD_CLASSES = {"stem": "stem", "pot": "pot", "rachis": "truss stalk", "pedicel": "truss stalk",
                "calyx": "truss stalk", "fruit": "fruit"}


class Scene:
    def __init__(self, seeds, turns=None, settle=True):
        """settle=False skips the 3 s settle (generate_cuts.best_turns' quick search; the leaves then hang as built)."""
        self.seeds, self.turns = seeds, turns
        self.m, self.d, self.info = lo.build(seeds, bend=True, turns=turns)
        m, d = self.m, self.d
        self.arm_adr = np.array([m.joint("arm/" + j).qposadr[0] for j in cp.ARM5])
        self.act = np.array([m.actuator("arm/" + j).id for j in cp.ARM5])
        self.jaw_adr, self.jaw_act = m.joint("arm/gripper").qposadr[0], m.actuator("arm/gripper").id
        self.start = arm_map.FollowerMap().norm_to_rad(START_NORM)[:5]
        self.home = self.start                    # where every pass starts and ends
        d.qpos[self.arm_adr], d.ctrl[self.act] = self.home, self.home
        d.qpos[self.jaw_adr] = d.ctrl[self.jaw_act] = JAW_CLOSED
        mujoco.mj_forward(m, d)
        for _ in range(round(SETTLE_S / m.opt.timestep) if settle else 0):
            mujoco.mj_step(m, d)
        self.ik = cp.CutIK(base=self.info["arm_base"], yaw=self.info["arm_yaw"])
        names = [m.geom(g).name or "" for g in range(m.ngeom)]
        self.names = names
        is_arm = np.array([m.body(m.geom_bodyid[g]).name.startswith("arm/") for g in range(m.ngeom)])
        self.arm_geoms = set(np.flatnonzero(is_arm & (m.geom_contype > 0)))
        self.cls = {}
        for g, n in enumerate(names):
            if is_arm[g]:
                continue
            c = sg.seg_class(n)
            if n.startswith("gh_"):
                self.cls[g] = "frame"
            elif c in HARD_CLASSES:
                self.cls[g] = HARD_CLASSES[c]
            elif c in ("petiole", "midrib"):
                self.cls[g] = "leaf"
            elif c == "leaflet":
                self.cls[g] = "leaflet"
        # never touched by any part of the arm; truss stalks are soft for the arm but guarded from the blades
        self.hard = {g for g, c in self.cls.items() if c in ("stem", "pot", "frame")}
        self.stalks = {g for g, c in self.cls.items() if c == "truss stalk"}

    def leaf_geoms(self, prefix):
        return {g for g, c in self.cls.items() if c in ("leaf", "leaflet") and self.names[g].startswith(prefix)}

    def checker(self, target_prefix):
        soft = {g for g, c in self.cls.items() if c == "leaf"}
        firm = {g for g, c in self.cls.items() if c in ("fruit", "truss stalk")}
        chk = rrt.Checker(self.m, self.d, hard=self.hard, soft=soft, firm=firm, blade_guard=self.stalks)
        chk.d.qpos[self.jaw_adr] = np.radians(JAW_APPROACH_DEG)
        return chk, self.leaf_geoms(target_prefix)


@dataclass
class Plan:
    plant: int
    node: int
    prefix: str
    pose: object                      # cutpose.CutPose
    slide: list                       # joint configs, pre-cut -> cut
    to_leaf: list                     # RRT route from where the arm was -> pre-cut
    brushes_leaves: bool              # True if no route cleared the kept leaves
    petiole_r: float
    tries: list = field(default_factory=list)   # (node, why not) for the leaves it skipped


@dataclass
class Skip:
    why: str
    tries: list = field(default_factory=list)


def _slide(scn, pose, petiole_r):
    """Joint configs along the straight slide, pre-cut (SLIDE_M back along the fingers) to the cut pose."""
    tool = cp.tool_point(petiole_r, CUT_CLEARANCE, face=pose.face)
    o, R = scn.ik.fk(pose.q)
    T_cut, f = o + R @ tool, -R[:, 2]
    qs, x = [], pose.q
    for s in np.linspace(0.0, SLIDE_M, SLIDE_STEPS):      # from the cut backwards, warm-starting each
        q, err, ang = scn.ik.reach_pose(T_cut - f * s, R, tool, x)
        # exact at the cut; further back the 5-joint arm may drift a little off the line (guess, unverified)
        if err > 0.0005 + SLIDE_DRIFT[0] * s / SLIDE_M or ang > 0.5 + SLIDE_DRIFT[1] * s / SLIDE_M:
            return None
        qs.append(q)
        x = q
    return qs[::-1]


def _jaw_catches(scn, chk, q, target):
    """What else the jaw would squeeze if it closed at q: anything but the target leaf inside the closing jaw.
    ⚠️ The first dry runs caught a truss stalk (15.7 N) or a kept leaf between the fingers, which took the
    squeeze away from the blades: the target wasn't cut (2026-11-01)."""
    out = set()
    for deg in np.linspace(JAW_APPROACH_DEG, np.degrees(JAW_CLOSED), 6):
        chk.d.qpos[scn.jaw_adr] = np.radians(deg)
        out |= chk.hits(q, soft=True, clearance=SLIDE_CLEARANCE, soft_clearance=0.0, ignore=target, firm_clearance=0.0)
    chk.d.qpos[scn.jaw_adr] = np.radians(JAW_APPROACH_DEG)
    return sorted(out)


def plan_cut(scn, rng, plant=0, only=None, q_from=None, route=True):
    """The oldest of the rule's cuts on this plant that has a clean flush pose, a clear slide and an RRT route from
    q_from (default the start pose), planned on the plant as it stands now (only: just that node). route=False skips
    the RRT route (to_leaf None): only whether the leaf itself can be cut cleanly."""
    q_from = scn.home if q_from is None else np.asarray(q_from, float)
    spec = scn.info["specs"][plant]
    sel = sg.label(spec)
    by_id = {n.id: i for i, n in enumerate(spec.nodes)}
    tries = []
    if not sel.cuts:
        return Skip("the rule cuts nothing on this plant", tries)
    for j in sel.cuts:
        i = by_id[sel.nodes[j].id]
        if only is not None and i != only:
            continue
        prefix = f"p{plant}_{spec.nodes[i].id}"
        lp, _ = spec.leaf[i]
        body = scn.m.body(prefix + ("petiole" if lp.bend else "leaf")).id
        leaf = scn.m.body(prefix + "leaf").id
        S = scn.d.xpos[leaf].copy()                       # on the stem axis (the welded base body)
        a = scn.d.xmat[body].reshape(3, 3)[:, 0].copy()   # the petiole's direction as it hangs now
        chk, target = scn.checker(prefix)
        why = "no clean flush pose (lean, pitch or twist)"
        for pose in scn.ik.solve(S, a, spec.stem_r, OFFSET, lp.petiole_r, clearance=CUT_CLEARANCE):
            if reach.classify(pose) != "ok":
                continue
            slide = _slide(scn, pose, lp.petiole_r)
            if slide is None:
                why = "the straight slide in isn't reachable"
                continue
            # on the short slide a finger pad may rest on a fruit or a truss stalk (firm=False); blades never do
            if not chk.path_ok(slide, soft=False, firm=False, clearance=SLIDE_CLEARANCE, ignore=target):
                why = f"the slide in would hit {sorted(chk.hits(slide[-1], soft=False, firm=False, clearance=SLIDE_CLEARANCE, ignore=target))[:3]}"
                continue
            caught = _jaw_catches(scn, chk, slide[-1], target)
            if caught:
                why = f"the closing jaw would also squeeze {caught[:2]}"
                continue
            if not route:
                return Plan(plant, i, prefix, pose, slide, None, False, lp.petiole_r, tries)
            route = rrt.plan(chk, q_from, slide[0], rng)
            brushes = False
            if route is None:
                route = rrt.plan(chk, q_from, slide[0], rng, soft=False)
                brushes = route is not None
            if route is None:
                why = "RRT found no route clear of stems, pots, the frame, fruit and truss stalks"
                continue
            return Plan(plant, i, prefix, pose, slide, route, brushes, lp.petiole_r, tries)
        tries.append((spec.nodes[i].id, why))
    return Skip("none of the rule's cuts on this plant can be planned", tries)


def _timed(path, speed=JOINT_SPEED, min_s=0.8):
    """A joint path played as one minimum-jerk move: returns (duration, q(t))."""
    P = np.array(path)
    seg = np.abs(np.diff(P, axis=0)).max(1)
    L = np.r_[0.0, np.cumsum(seg)]
    T = max(min_s, 1.875 * L[-1] / speed if L[-1] > 0 else min_s)     # min-jerk peaks at 1.875x the average

    def q(t):
        u = np.clip(t / T, 0.0, 1.0)
        s = (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5) * L[-1]
        k = min(int(np.searchsorted(L, s, side="right")) - 1, len(P) - 2)
        w = 0.0 if seg[k] == 0 else (s - L[k]) / seg[k]
        return P[k] + (P[k + 1] - P[k]) * np.clip(w, 0, 1)
    return T, q


class Monitor:
    """Peak force of any arm part on each kind of thing it mustn't damage, plus kept leaves touched."""

    def __init__(self, scn, target_prefix=""):
        self.scn = scn
        self.phase = ""
        self.reset(target_prefix)
        self.blades = {g for g in scn.arm_geoms if scn.m.geom(g).name.startswith("arm/blade_")}
        self._f6 = np.zeros(6)

    def reset(self, target_prefix=""):
        """New tallies (each cut is judged on its own)."""
        self.target = self.scn.leaf_geoms(target_prefix) if target_prefix else set()
        self.peak = {k: 0.0 for k in ("stem", "truss stalk", "fruit", "pot", "frame")}
        self.where = {}                   # kind -> the phase its peak force came in
        self.leaves_touched = set()
        self.blade_on_stalk = 0.0

    def step(self):
        m, d, scn = self.scn.m, self.scn.d, self.scn
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in scn.arm_geoms) == (g2 in scn.arm_geoms):
                continue
            other = g2 if g1 in scn.arm_geoms else g1
            kind = scn.cls.get(other)
            if kind is None:
                continue
            if kind in ("leaf", "leaflet"):
                if kind == "leaf" and other not in self.target:      # a kept leaf's stalk or midrib
                    self.leaves_touched.add("_".join(scn.names[other].split("_")[:2]))
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            if self._f6[0] > self.peak[kind]:
                self.peak[kind], self.where[kind] = float(self._f6[0]), self.phase
            if kind == "truss stalk" and (g1 if g1 in scn.arm_geoms else g2) in self.blades:
                self.blade_on_stalk = max(self.blade_on_stalk, float(self._f6[0]))


class Episode:
    """Plays cuts one after another in the physics at 500 Hz, the servos following a 30 Hz reference with an
    integral sag correction. frame_cb(phase, episode) is called at FPS (video)."""

    def __init__(self, scn, rng, frame_cb=None):
        self.scn, self.rng, self.cb = scn, rng, frame_cb
        m, d = scn.m, scn.d
        # ⚠️ One Cutter per leaf for the whole pass: a Cutter made later would reset() a leaf already cut back
        # onto the stem. Only the ones near the current cut are stepped.
        self.cutters = {key: pl.Cutter(m, d, w, pl.cut_force_for(scn.info["specs"][key[0]].leaf[key[1]][0].petiole_r))
                        for key, w in scn.info["where"].items()}
        self.mon = Monitor(scn)
        self.I = np.zeros(5)
        self.steps_per_frame = round(1.0 / FPS / m.opt.timestep)
        self.jaw = float(d.qpos[scn.jaw_adr])
        self.target = None                # (plant, node) being cut now, for the video
        self.done = set()                 # (plant, node) cut so far
        self.t = 0.0

    def q_now(self):
        return self.scn.d.qpos[self.scn.arm_adr].copy()

    def play(self, name, dur, ref, jaw, watch=()):
        scn, m, d = self.scn, self.scn.m, self.scn.d
        self.mon.phase = name
        for f in range(max(1, round(dur * FPS))):
            t = (f + 1) / FPS
            q_ref = ref(t)
            self.I = np.clip(self.I + SAG_KI * (q_ref - self.q_now()), -SAG_MAX, SAG_MAX)
            d.ctrl[scn.act] = q_ref + self.I
            self.jaw = jaw(t)
            d.ctrl[scn.jaw_act] = self.jaw
            for k in range(self.steps_per_frame):
                mujoco.mj_step(m, d)
                for c in watch:
                    c.step()
                if k % 5 == 0:
                    self.mon.step()
            self.t += 1.0 / FPS
            if self.cb:
                self.cb(name, self)

    def hold(self, name, dur, jaw_to=None):
        j0 = self.jaw
        jaw_to = j0 if jaw_to is None else jaw_to
        q_hold = self.scn.d.ctrl[self.scn.act] - self.I        # hold the last reference
        self.play(name, dur, lambda t: q_hold, lambda t: j0 + (jaw_to - j0) * min(1.0, t / max(dur, 1e-6)))

    def route(self, name, path, jaw_to):
        T, q = _timed(path)
        j0 = self.jaw
        self.play(name, T, q, lambda t: j0 + (jaw_to - j0) * min(1.0, t / 0.5))

    def cut(self, plan):
        """One cut, judged on its own."""
        scn, d = self.scn, self.scn.d
        key = (plan.plant, plan.node)
        self.target = key
        self.mon.reset(plan.prefix)
        cut_pt = scn.info["where"][key]["base"]
        near = [c for k, c in self.cutters.items() if k not in self.done
                and np.linalg.norm(scn.info["where"][k]["base"] - cut_pt) < NEAR_LEAF_M]
        jaw_open = np.radians(JAW_APPROACH_DEG)
        self.route("to_leaf", plan.to_leaf, jaw_open)
        T_sl = SLIDE_M / SLIDE_SPEED
        _, q_sl = _timed(plan.slide, min_s=T_sl)
        self.play("slide_in", T_sl, q_sl, lambda t: jaw_open)
        err = float(np.linalg.norm(scn.ik.fk(self.q_now())[0] - scn.ik.fk(plan.slide[-1])[0]))
        joint_err = np.round(np.degrees(self.q_now() - plan.slide[-1]), 2).tolist()
        self.play("close", CLOSE_S + CUT_WAIT_S, lambda t: plan.slide[-1],
                  lambda t: jaw_open + (JAW_CLOSED - jaw_open) * min(1, t / CLOSE_S), watch=near)
        _, q_back = _timed(plan.slide[::-1], min_s=T_sl)
        self.play("slide_out", T_sl, q_back, lambda t: JAW_CLOSED, watch=near)
        tgt = self.cutters[key]
        others = [k for k, c in self.cutters.items() if k != key and k not in self.done and c.cut_t is not None]
        for k in others + ([key] if tgt.cut_t is not None else []):
            self.done.add(k)
        self.target = None
        mon = self.mon
        why = []
        if tgt.cut_t is None:
            why.append(f"the target leaf wasn't cut (blades {tgt.peak['squeeze']:.1f} N, need {tgt.cut_force:.1f})")
        elif not STUB_OK[0] <= tgt.stub <= STUB_OK[1]:
            why.append(f"stub {tgt.stub * 1000:.1f} mm")
        if others:
            why.append(f"cut the wrong leaf too: {[scn.info['specs'][k].nodes[i].id for k, i in others]}")
        for k, lim in FORCE_LIMIT.items():
            if mon.peak[k] > lim:
                why.append(f"pressed the {k} with {mon.peak[k]:.1f} N ({mon.where.get(k)})")
        if mon.blade_on_stalk > 0:
            why.append(f"a blade touched a truss stalk ({mon.blade_on_stalk:.1f} N)")
        notes = []
        if mon.peak["fruit"] > FRUIT_PUSH_N:
            notes.append(f"pushed fruit with {mon.peak['fruit']:.1f} N")
        if mon.leaves_touched:
            notes.append(f"brushed {len(mon.leaves_touched)} kept leaves")
        if plan.brushes_leaves:
            notes.append("no route cleared every kept leaf")
        return dict(verdict="PASS" if not why else "FAIL", why=why, notes=notes, plant=plan.plant,
                    leaf=scn.info["specs"][plan.plant].nodes[plan.node].id,
                    stub_mm=None if tgt.stub is None else round(tgt.stub * 1000, 2), second_cut=bool(tgt.second_cut),
                    cut_force_n=round(tgt.cut_force, 1), blade_peak_n=round(tgt.peak["squeeze"], 1),
                    pair_peaks_n=[round(x, 1) for x in tgt.peak_pairs], aim_error_mm=round(err * 1000, 1),
                    joint_err_deg_at_cut=joint_err, peaks_n={k: round(v, 2) for k, v in mon.peak.items()},
                    pose=dict(lean=round(plan.pose.lean_deg, 1), twist=round(plan.pose.twist_deg, 1),
                              pitch=round(plan.pose.pitch_deg, 1), blade_pair=plan.pose.face),
                    route_points=len(plan.to_leaf))

    def go_start(self):
        """An RRT route back to the start pose, planned on the plant as it stands now."""
        chk, _ = self.scn.checker("none_")
        q0 = self.q_now()
        path = rrt.plan(chk, q0, self.scn.start, self.rng) or rrt.plan(chk, q0, self.scn.start, self.rng, soft=False)
        if path is None:
            path = [q0, self.scn.start]
        self.mon.reset()
        self.route("to_start", path, JAW_CLOSED)
        return dict(route_points=len(path), peaks_n={k: round(v, 2) for k, v in self.mon.peak.items()},
                    pressed=[k for k, lim in FORCE_LIMIT.items() if self.mon.peak[k] > lim])


def run_pass(scn, rng, frame_cb=None, plants=(0, 1, 2)):
    """One deleafing pass: start pose -> the rule's cuts on each plant in turn, oldest first -> start pose."""
    ep = Episode(scn, rng, frame_cb)
    ep.hold("start", HOLD_S)
    cuts, skipped = [], []
    for k in plants:
        spec = scn.info["specs"][k]
        sel = sg.label(spec)
        by_id = {n.id: i for i, n in enumerate(spec.nodes)}
        if not sel.cuts:
            skipped.append(dict(plant=k, leaf=None, why="the rule cuts nothing on this plant"))
        for j in sel.cuts:
            i = by_id[sel.nodes[j].id]
            if (k, i) in ep.done:
                continue
            p = plan_cut(scn, rng, plant=k, only=i, q_from=ep.q_now())
            if isinstance(p, Skip):
                skipped.append(dict(plant=k, leaf=spec.nodes[i].id, why=p.tries[0][1] if p.tries else p.why))
                continue
            cuts.append(ep.cut(p))
    end = ep.go_start()
    ep.hold("end", HOLD_S)
    ok = bool(cuts) and all(c["verdict"] == "PASS" for c in cuts) and not end["pressed"]
    return dict(verdict="PASS" if ok else "FAIL", cuts=cuts, skipped=skipped, back_to_start=end,
                n_cut=sum(c["verdict"] == "PASS" for c in cuts), n_rule=sum(len(sg.label(scn.info["specs"][k]).cuts) for k in plants),
                seconds_sim=round(ep.t, 1))
