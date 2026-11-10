"""Teacher v3 for deleafing: ONE flush cut per episode, as a closed-loop teacher (BLUSH + SHEAR post-mortem §6.1,
K7 approved 2026-11-09). Copied in design from B.L.U.S.H sim/truss_teacher.py (commit b25ea3b): the Leg pure pursuit,
the fixed priority lists, the one-detour rule.

    scn = cutter.Scene(seeds, turns)                       the row, settled at the start pose (no start jitter)
    T = ShearTeacher(scn).plan()                           the target leaf and every leg; PlanError if none
    out = run(scn, T, frame_cb)                            plays it in the physics, judged; frame_cb as cutter's

What changed from cutter.run_pass (the v1 expert, which recorded shear_sim_scripted_v1):
- One cut per episode, from the start pose and back (post-mortem R8): v1 passes held one cut in 165 of 201 kept
  episodes anyway, but WHICH of the rule's leaves came out of a reachability search the camera can't see. The
  target is now the first plannable leaf in a fixed order (plant p0, p1, p2; the rule's cuts oldest first; the
  cut poses in cutpose's order) and its cut point and petiole direction go into meta (cue_*), for the recorder to
  give the policy as numbers (observation.environment_state).
- No coin-flips: no RRT route (free moves are a straight joint line, else one detour over a fixed via-pose, else
  the leaf is dropped), no start jitter, no start hold (identical frames at home froze the n10 ACT run).
- Closed loop: act() works from the measured joints every frame (pure pursuit along the leg's joint path), so a push
  is steered back from and DAgger can ask it from any state. Pure pursuit waits for the arm by itself: the command
  is never more than one lead ahead of where the arm IS (v1 needed LAG_MAX for that, cutter.py 2026-11-06).

The motion: start -> standoff (STANDOFF_M behind the pre-cut pose along the fingers) -> pre-cut pose (cutter.SLIDE_M
back) -> slide in at cutter.SLIDE_SPEED -> close (cutter.CLOSE_S) + wait (cutter.CUT_WAIT_S) -> slide out -> start ->
short hold. The cut pose, the slide and their checks are cutter.plan_cut's (route=False), unchanged.
"""
import mujoco
import numpy as np

import arm_map
import cutter as ct

FPS = ct.FPS


class PlanError(Exception):
    pass


# --- fixed priority lists (first that works wins; design choices, 2026-11-09) ---
PLANTS = (0, 1, 2)                         # v1's pass order
STANDOFF_M = (0.04, 0.02, 0.0)             # behind the pre-cut pose along the fingers (as B.L.U.S.H's teacher)
STANDOFF_STEPS = 6                         # IK way-points standoff -> pre-cut (guess)
# Clearance levels for the free moves, in order: v1's RRT tried the default first, then through the kept leaves
# (soft=False); home first with a wide margin (cutter.HOME_CLEARANCE, see the 300 N pot press there).
TO_LEAF_KW = ({}, {"soft": False})
HOME_KW = ({"clearance": ct.HOME_CLEARANCE}, {}, {"soft": False})
# The one detour: over a fixed via-pose, nearest the move's goal first (joint space). Joint angles in norm units
# (use_degrees=false), copied from v1 RRT routes whose straight line was blocked; both went high over the aisle
# first (rows 910003 and 910009, 2026-11-09; gripper origin in the comment).
VIA_NORM = ((-36.0, -44.0, -27.0, -28.0, 1.0),     # (0.22, 0.18, 0.41): high over the aisle, left (row 910003)
            (-18.0, -50.0, -3.0, 15.0, -63.0),     # (0.19, 0.26, 0.36): high, left of p0 (row 910009)
            # Added from a search (2026-11-09): v1 RRT routes on rows 920000-920267 (apart from the test rows), the
            # route way-points pooled, and the one that unblocks most rows the line and the vias above can't reach
            # kept. 9 of 90 rows were cuttable but blocked; this pose unblocks 2, no other pose 2 more.
            (12.0, -69.0, 12.0, -18.0, -12.0))     # (0.22, 0.20, 0.36): high over the aisle, centre
# --- timing: v1's fixed values ---
HOLD_END_S = 0.3                           # v1 0.5; short, see the docstring
CLOSE_S, CUT_WAIT_S = ct.CLOSE_S, ct.CUT_WAIT_S
# --- closed loop (rad) ---
FREE_LEAD = 1.875 * ct.JOINT_SPEED / FPS   # v1's min-jerk peak speed on the largest joint, per frame
STANDOFF_M_S = 2 * ct.SLIDE_SPEED          # standoff -> pre-cut: nothing in the jaw yet, twice the slide's pace (guess)
END_TOL = {"servo": np.radians(0.3), "free": np.radians(2.0)}   # from a leg's end to call it reached (guess)
STALL_RAD, STALL_FRAMES = np.radians(0.03), 8   # ...or the arm moved less than this per frame for this many
SEARCH_BACK, SEARCH_AHEAD = 0.5, 6.0       # projection window in leads (B.L.U.S.H's teacher)
END_START_TOL = np.radians(3.0)            # back at the start pose within this on every joint (guess)
MAX_S = 40.0                               # done after this long whatever happens


# ⚠️ Self-collision (2026-11-09): rrt.Checker skips every contact between two arm parts, so neither the v1 expert's
# RRT nor this teacher saw the arm fold into itself. Leaving the pre-cut pose of rows 910009 and 910486 the shoulder
# touched the lower arm, the elbow servo sat at its full 3.35 N.m and the run stalled until the time limit (B.L.U.S.H's
# teacher had the same with its wrist camera mount). Every checker the teacher uses now also reports arm parts closer
# than SELF_CLEARANCE, except pairs inside the hand (HAND_PARTS touch by design).
SELF_CLEARANCE = 0.002                     # m (the planner's soft clearance, rrt.SOFT_CLEARANCE; guess, unverified)
HAND_PARTS = ("gripper", "moving_jaw", "finray", "pad_", "wrist_camera", "blade")


def add_self_check(chk, prefix="arm/"):
    """Make chk.hits (and so free / segment_free / path_ok) also report self-collisions of the arm. Returns chk."""
    m = chk.m
    body = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
    arm = np.array([b.startswith(prefix) for b in body])
    hand = np.array([any(h in b for h in HAND_PARTS) for b in body])
    orig = chk.hits

    def hits(q, **kw):
        out = orig(q, **kw)                       # leaves chk.d at q, contacts current
        for c in chk.d.contact[:chk.d.ncon]:
            a, b = c.geom1, c.geom2
            if (arm[a] and arm[b] and m.geom_bodyid[a] != m.geom_bodyid[b] and not (hand[a] and hand[b])
                    and c.dist < SELF_CLEARANCE):
                out.add(f"self: {body[a][len(prefix):]} / {body[b][len(prefix):]}")
        return out
    chk.hits = hits
    return chk


class Leg:
    """B.L.U.S.H truss_teacher.Leg (b25ea3b), in rad: a joint path ((n, 5)) or a hold, the jaw, the pace."""

    def __init__(self, name, W=None, hold=None, jaw=None, jaw_to=None, lead=FREE_LEAD, servo=False, frames=None):
        self.name, self.lead, self.servo, self.frames = name, lead, servo, frames
        self.W = None if W is None else np.asarray(W, float)
        self.hold = None if hold is None else np.asarray(hold, float)
        self.jaw, self.jaw_to = jaw, jaw_to
        if self.W is not None:
            seg = np.linalg.norm(np.diff(self.W, axis=0), axis=1)
            self.cum = np.r_[0.0, np.cumsum(seg)]
            self.L = float(self.cum[-1])

    def point(self, s):
        s = float(np.clip(s, 0.0, self.L))
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.W) - 2))
        seg = self.cum[i + 1] - self.cum[i]
        u = 0.0 if seg < 1e-12 else (s - self.cum[i]) / seg
        return self.W[i] + (self.W[i + 1] - self.W[i]) * u

    def project(self, q, s_lo, s_hi):
        best, best_s = np.inf, s_lo
        for i in range(len(self.W) - 1):
            a, b = self.W[i], self.W[i + 1]
            c0, c1 = self.cum[i], self.cum[i + 1]
            if c1 < s_lo or c0 > s_hi:
                continue
            ab = b - a
            n2 = float(ab @ ab)
            u = 0.0 if n2 < 1e-12 else float(np.clip((q - a) @ ab / n2, 0, 1))
            s = float(np.clip(c0 + (c1 - c0) * u, s_lo, s_hi))
            dist = np.linalg.norm(q - self.point(s))
            if dist < best:
                best, best_s = dist, s
        return best_s


class ShearTeacher:
    def __init__(self, scn):
        self.scn = scn
        self.fm = arm_map.FollowerMap()
        self.vias = [self.fm.norm_to_rad(np.r_[v, 0.0])[:5] for v in VIA_NORM]
        self.legs, self.phases, self.meta = [], [], {}

    # --- planning (once, before the first frame) ---
    def plan(self):
        scn = self.scn
        tried = []
        for k in PLANTS:
            spec = scn.info["specs"][k]
            sel = ct.sg.label(spec)
            by_id = {n.id: i for i, n in enumerate(spec.nodes)}
            for j in sel.cuts:
                i = by_id[sel.nodes[j].id]
                p = ct.plan_cut(scn, None, plant=k, only=i, route=False)   # route=False: no RRT, no rng
                if isinstance(p, ct.Skip):
                    tried.append(f"p{k} leaf {spec.nodes[i].id}: {p.tries[0][1] if p.tries else p.why}")
                    continue
                why = self._legs_for(p)
                if why is None:
                    self.meta.update(plant=k, leaf=spec.nodes[i].id, n_rule=len(sel.cuts), tried=tried)
                    return self
                tried.append(f"p{k} leaf {spec.nodes[i].id}: {why}")
        self.meta = dict(tried=tried)
        raise PlanError("; ".join(tried) or "the rule cuts nothing on any plant")

    def _standoffs(self, p):
        """Joint way-points standoff -> pre-cut for each STANDOFF_M that is reachable (cutter._slide's IK and drift
        limits, extended back), in order."""
        scn = self.scn
        tool = ct.cp.tool_point(p.petiole_r, ct.CUT_CLEARANCE, face=p.pose.face)
        o, R = scn.ik.fk(p.pose.q)
        T_cut, f = o + R @ tool, -R[:, 2]
        out = []
        for so in STANDOFF_M:
            if so == 0.0:
                out.append(([p.slide[0]], 0.0))
                continue
            qs, x, ok = [], p.slide[0], True
            for s in np.linspace(ct.SLIDE_M, ct.SLIDE_M + so, STANDOFF_STEPS):
                q, err, ang = scn.ik.reach_pose(T_cut - f * s, R, tool, x)
                if err > 0.0005 + ct.SLIDE_DRIFT[0] * s / ct.SLIDE_M or ang > 0.5 + ct.SLIDE_DRIFT[1] * s / ct.SLIDE_M:
                    ok = False
                    break
                qs.append(q)
                x = q
            if ok:
                out.append((qs[::-1], so))
        return out

    def free_move(self, chk, xa, xb, levels, **extra):
        """For each clearance level in order: the straight joint line, else over a fixed via-pose (nearest xb first),
        else two vias chained. Returns (way-points, how) or (None, None); how = (level, 0 for the line, the via's
        number from 1, or 10 * first + second for a chain)."""
        V = self.vias
        dist = lambda u, v: float(np.abs(u - v).max())   # noqa: E731
        order = sorted(range(len(V)), key=lambda j: dist(V[j], xb))
        # last fallback, as B.L.U.S.H's teacher: two vias chained (up out of the start, then over), fixed order
        pairs = sorted(((i, j) for i in range(len(V)) for j in range(len(V)) if i != j),
                       key=lambda ij: dist(xa, V[ij[0]]) + dist(V[ij[0]], V[ij[1]]) + dist(V[ij[1]], xb))
        for li, kw in enumerate(levels):
            kw = {**kw, **extra}
            if chk.path_ok([xa, xb], **kw):
                return [xa, xb], (li, 0)
            for j in order:
                if chk.path_ok([xa, self.vias[j], xb], **kw):
                    return [xa, self.vias[j], xb], (li, j + 1)
            for i, j in pairs:
                if chk.path_ok([xa, self.vias[i], self.vias[j], xb], **kw):
                    return [xa, self.vias[i], self.vias[j], xb], (li, 10 * (i + 1) + j + 1)
        return None, None

    def _legs_for(self, p):
        scn = self.scn
        chk, target = scn.checker(p.prefix)
        add_self_check(chk)
        q0 = scn.d.qpos[scn.arm_adr].copy()
        # plan_cut checked the slide without the arm's own parts; again with them
        if not chk.path_ok(p.slide, soft=False, firm=False, clearance=ct.SLIDE_CLEARANCE, ignore=target):
            return "the slide folds the arm into itself"
        to_leaf, std = None, None
        for xs, so in self._standoffs(p):
            if len(xs) > 1 and not chk.path_ok(xs):
                continue
            path, how = self.free_move(chk, q0, xs[0], TO_LEAF_KW)
            if path is not None:
                to_leaf, std, how_leaf = path, (xs, so), how
                break
        if to_leaf is None:
            return "start -> standoff blocked, via-poses too"
        # home from the pre-cut pose with the leaf gone (its geoms ignored), jaw shut as v1's go_start
        chk.d.qpos[scn.jaw_adr] = ct.JAW_CLOSED
        home, how_home = self.free_move(chk, p.slide[0], scn.start, HOME_KW, ignore=target)
        chk.d.qpos[scn.jaw_adr] = np.radians(ct.JAW_APPROACH_DEG)
        if home is None:
            return "pre-cut -> start blocked, via-poses too"
        jaw_open, jaw_shut = np.radians(ct.JAW_APPROACH_DEG), ct.JAW_CLOSED
        jaw_now = float(scn.d.qpos[scn.jaw_adr])

        def cart_lead(xs, metres, speed):
            L = float(np.linalg.norm(np.diff(np.asarray(xs), axis=0), axis=1).sum())
            return max(L / max(metres / speed * FPS, 1.0), 1e-5)

        f = lambda s: max(1, int(round(s * FPS)))   # noqa: E731
        cut = np.asarray(p.slide[-1])
        L = [Leg("to_leaf", to_leaf, jaw_to=(jaw_now, jaw_open))]
        if len(std[0]) > 1:
            L.append(Leg("standoff_in", std[0], jaw=jaw_open, servo=True,
                         lead=cart_lead(std[0], std[1], STANDOFF_M_S)))
        L.append(Leg("slide_in", p.slide, jaw=jaw_open, servo=True,
                     lead=cart_lead(p.slide, ct.SLIDE_M, ct.SLIDE_SPEED)))
        L.append(Leg("close", hold=cut, jaw_to=(jaw_open, jaw_shut), servo=True, frames=f(CLOSE_S)))
        L.append(Leg("cut_wait", hold=cut, jaw=jaw_shut, servo=True, frames=f(CUT_WAIT_S)))
        L.append(Leg("slide_out", p.slide[::-1], jaw=jaw_shut, servo=True,
                     lead=cart_lead(p.slide, ct.SLIDE_M, ct.SLIDE_SPEED)))
        L.append(Leg("to_start", home, jaw=jaw_shut))
        L.append(Leg("hold_end", hold=scn.start, jaw=jaw_shut, frames=f(HOLD_END_S)))
        self.legs, self.plan_cut = L, p
        where = scn.info["where"][(p.plant, p.node)]
        lp, _ = scn.info["specs"][p.plant].leaf[p.node]
        body = scn.m.body(p.prefix + ("petiole" if lp.bend else "leaf")).id
        self.meta = dict(teacher="v3", standoff_mm=round(std[1] * 1000), to_leaf=dict(level=how_leaf[0],
                         via=how_leaf[1]), home=dict(level=how_home[0], via=how_home[1]),
                         cue_cut_point=np.round(where["base"], 4).tolist(),
                         cue_petiole_dir=np.round(scn.d.xmat[body].reshape(3, 3)[:, 0], 4).tolist(),
                         pose=dict(lean=round(p.pose.lean_deg, 1), twist=round(p.pose.twist_deg, 1),
                                   pitch=round(p.pose.pitch_deg, 1), blade_pair=p.pose.face))
        self.reset()
        return None

    # --- running (every frame) ---
    def reset(self):
        self.i, self.n_in_leg, self.n_total = 0, 0, 0
        self.s, self.I, self.stall, self.q_prev = 0.0, np.zeros(5), 0, None
        self.phases = []

    @property
    def done(self):
        return self.i >= len(self.legs) or self.n_total >= MAX_S * FPS

    @property
    def leg(self):
        return self.legs[min(self.i, len(self.legs) - 1)].name

    def _next_leg(self):
        self.phases.append((self.legs[self.i].name, self.n_in_leg))
        self.i += 1
        self.n_in_leg, self.s, self.stall, self.q_prev = 0, 0.0, 0, None

    def act(self, q):
        """(arm command (5, rad), jaw command (rad)) for this frame from the measured arm joints q (5, rad)."""
        q = np.asarray(q, float)
        g = self.legs[self.i]
        if g.W is not None:
            self.s = g.project(q, self.s - SEARCH_BACK * g.lead, self.s + SEARCH_AHEAD * g.lead)
            target, ref = g.point(self.s + g.lead), g.point(self.s)
            prog = 1.0 if g.L < 1e-12 else min(1.0, (self.s + g.lead) / g.L)
        else:
            target = ref = g.hold
            prog = min(1.0, (self.n_in_leg + 1) / g.frames)
        # v1 corrected the sag on every frame (cutter.Episode.play); here on the cross-track error, not the lead
        self.I = np.clip(self.I + ct.SAG_KI * (ref - q), -ct.SAG_MAX, ct.SAG_MAX)
        jaw = g.jaw if g.jaw_to is None else g.jaw_to[0] + (g.jaw_to[1] - g.jaw_to[0]) * prog
        self.n_in_leg += 1
        self.n_total += 1
        if g.W is not None:
            moved = np.inf if self.q_prev is None else float(np.abs(q - self.q_prev).max())
            self.stall = self.stall + 1 if moved < STALL_RAD else 0
            self.q_prev = q
            close = float(np.abs(q - g.W[-1]).max()) < END_TOL["servo" if g.servo else "free"]
            if self.s + g.lead >= g.L and (close or self.stall >= STALL_FRAMES):
                self._next_leg()
        elif self.n_in_leg >= g.frames:
            self._next_leg()
        return target + self.I, jaw


def run(scn, T, frame_cb=None, push=None, displace=None):
    """Play teacher T in scn's physics (cutter.Episode's stepping, cutters and force monitor), judge the one cut.
    push: (leg name, frame in leg) at which displace(scn) is called (check_teacher.py)."""
    ep = ct.Episode(scn, None, frame_cb)
    m, d = scn.m, scn.d
    p = T.plan_cut
    key = (p.plant, p.node)
    ep.target = key
    ep.mon.reset(p.prefix)
    cut_pt = scn.info["where"][key]["base"]
    near = [c for k, c in ep.cutters.items() if np.linalg.norm(scn.info["where"][k]["base"] - cut_pt) < ct.NEAR_LEAF_M]
    pushed = None
    tgt = ep.cutters[key]
    while not T.done:
        # ⚠️ The cut completes during the slide out (jaw shut, pulling): opening before it left 2 of 2 rows uncut
        # (2026-11-09). If the leaf is still on after the slide out, the jaw holds a leaf that's attached to the plant
        # and the arm can't leave (row 910147 stalled for the full MAX_S): stop there, the run fails anyway.
        if T.leg == "to_start" and tgt.cut_t is None:
            break
        if push is not None and pushed is None and T.leg == push[0] and T.n_in_leg == push[1]:
            pushed = displace(scn)
        leg = T.leg
        a, jaw = T.act(ep.q_now())
        ep.mon.phase = leg
        d.ctrl[scn.act] = a
        ep.jaw = jaw
        d.ctrl[scn.jaw_act] = jaw
        for s in range(ep.steps_per_frame):
            mujoco.mj_step(m, d)
            for c in near:
                c.step()
            if s % 5 == 0:
                ep.mon.step()
        ep.t += 1.0 / FPS
        if frame_cb:
            frame_cb(leg, ep)
    others = [k for k, c in ep.cutters.items() if k != key and c.cut_t is not None]
    mon, why = ep.mon, []
    if tgt.cut_t is None:
        why.append(f"the target leaf wasn't cut (blades {tgt.peak['squeeze']:.1f} N, need {tgt.cut_force:.1f})")
    elif not ct.STUB_OK[0] <= tgt.stub <= ct.STUB_OK[1]:
        why.append(f"stub {tgt.stub * 1000:.1f} mm")
    if others:
        why.append(f"cut the wrong leaf too: {others}")
    for k, lim in ct.FORCE_LIMIT.items():
        if mon.peak[k] > lim:
            why.append(f"pressed the {k} with {mon.peak[k]:.1f} N ({mon.where.get(k)})")
    if mon.blade_on_stalk > 0:
        why.append(f"a blade touched a truss stalk ({mon.blade_on_stalk:.1f} N)")
    off = float(np.abs(ep.q_now() - scn.start).max())
    if off > END_START_TOL:
        why.append(f"arm {np.degrees(off):.1f} deg off the start pose at the end")
    if not T.done or T.i < len(T.legs):
        why.append(f"timed out in {T.leg}")
    return dict(verdict="PASS" if not why else "FAIL", why=why,
                stub_mm=None if tgt.stub is None else round(tgt.stub * 1000, 2),
                fruit_peak_n=round(mon.peak["fruit"], 2), kept_leaves_brushed=len(mon.leaves_touched),
                seconds=round(ep.t, 2), frames=T.n_total, phases=T.phases,
                pushed_mm=None if pushed is None else round(pushed * 1000, 1))
