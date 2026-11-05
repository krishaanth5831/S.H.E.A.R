#!/usr/bin/env python3
"""Correction episodes for the SHEAR policy: sim DAgger, the scripted cutter as the person who steps in
(K7 2026-11-05: "go for the correction data, do another 100 episodes of that and add it to this dataset").

  python generate_corrections.py --ckpt <policy pretrained_model> --episodes 100     appends to krish5831/shear_sim_scripted_v0
  python generate_corrections.py --ckpt ... --episodes 2 --repo-id krish5831/shear_corrections_smoketest --new

Why: the policy trained on the 200 scripted passes reaches the right leaf but drives the jaw into the stem in the
last few mm (v1: 0 of 51 cuts, the stem pressed in 10 of 10 rows, 2026-11-05). Every demonstration ends 2 mm from
the stem and none shows a recovery from a small error, so the policy has never seen how to get out of one.
Dream Machines' π0.5 write-up (2026): intervention data took a policy from 28 % to 88 % (HG-DAgger style).

Per row (seeds SEED_BASE+, apart from the dataset's and the rollouts'), set up like the dataset (pots, domain
randomisation):
1. The policy drives from the start pose, with RTC, exactly as in rollout_policy.py. Nothing is recorded.
2. The scripted cutter takes over (TAKEOVER) as soon as the arm comes within DANGER_M of a stem, pot or the frame, or
   a blade near a truss stalk (rrt.Checker), or the hand has hovered near a leaf to cut for HOVER_S without cutting,
   or after POLICY_MAX_S.
3. If the arm is jammed (no clear start for a route), it first backs off straight (back_off). Then it plans the cut of the leaf to cut nearest the hand (cutter.plan_cut from the current pose:
   an RRT route, the straight slide, the flush cut), plays it, and takes an RRT route back to the start pose. That
   takeover, from its first frame to the start pose, is recorded as one episode, like generate_episodes.py (same
   features, task string, cameras). The policy's own lead-up isn't: it would teach the mistakes.
Kept only if the cut passes the judge and nothing was pressed after the takeover. Logged to ~/.cache/shear_sim/episodes/.

⚠️ Appending changes the dataset in place: back it up first (it was, to shear_sim_scripted_v0_backup_200eps). Run
outside Claude Code's sandbox (the GPU), under systemd-run MemoryMax. Never pushes to the Hub.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")      # ⚠️ cameras must render through EGL (B.L.U.S.H trap)

import numpy as np  # noqa: E402

import cutpose as cp  # noqa: E402
import cutter as ct  # noqa: E402
import generate_cuts as gc  # noqa: E402
import generate_episodes as ge  # noqa: E402
import rollout_policy as rp  # noqa: E402
import stemgen as sg  # noqa: E402

SEED_BASE = 960_000            # apart from the dataset (900000s), the rollouts (980000s) and the trials (700000s)
DANGER_M = 0.006               # arm this close to a stem, pot or the frame: step in (the scripted route keeps 4 mm)
HOVER_M, HOVER_S = 0.03, 2.0   # hand within this of a leaf to cut, this long, without a cut: step in
POLICY_MAX_S = 20.0            # the policy drives at most this long before the cutter steps in anyway
# ⚠️ Where the policy leaves the arm (jammed near the stem) the planner has no clear start: 5 of 5 smoke rows had no
# plan from there (2026-11-05). So the takeover first backs off, as a person would: straight back along the fingers
# (else straight up), gripper turned as it is, until the pose is clear of everything, at most BACKOFF_MAX_M.
BACKOFF_STEP_M, BACKOFF_MAX_M = 0.005, 0.10
REPO_ID = ge.REPO_ID


def rule_cuts(scn):
    out = []
    for k, sp in enumerate(scn.info["specs"]):
        sel = sg.label(sp)
        by_id = {n.id: i for i, n in enumerate(sp.nodes)}
        out += [(k, by_id[sel.nodes[j].id]) for j in sel.cuts]
    return out


def tool_pos(scn):
    """Where the blades' centre is now (the cut tool point, -y pair, 1.5 mm petiole)."""
    o, R = scn.ik.fk(scn.d.qpos[scn.arm_adr])
    return o + R @ cp.tool_point(0.0015, ct.CUT_CLEARANCE)


def back_off(scn, chk):
    """A straight retreat from the current pose to the nearest clear one: (path or None, why it failed per way)."""
    q0 = scn.d.qpos[scn.arm_adr].copy()
    chk.d.qpos[:] = scn.d.qpos
    if chk.free(q0, soft=False):
        return [q0], []
    tool = cp.tool_point(0.0015, ct.CUT_CLEARANCE)
    o, R = scn.ik.fk(q0)
    T = o + R @ tool
    up = np.array([0.0, 0.0, 1.0])
    to_base = scn.info["arm_base"] - T
    to_base[2] = 0.0
    to_base /= max(np.linalg.norm(to_base), 1e-9)
    back = -R[:, 2]
    ways = {"back": back, "up": up, "back+up": (back + up) / np.linalg.norm(back + up), "to the base": to_base}
    whys = []
    for name, d in ways.items():
        path, x, why = [q0], q0, "never clear"
        for s in np.arange(BACKOFF_STEP_M, BACKOFF_MAX_M + 1e-9, BACKOFF_STEP_M):
            x, err, ang = scn.ik.reach_pose(T + d * s, R, tool, x)
            if err > 0.003 or ang > 5.0:
                why = f"out of reach at {s * 100:.1f} cm"
                break
            path.append(x)
            if chk.free(x, soft=False):
                return path, whys
        hit = sorted(chk.hits(x, soft=False))[:2]
        whys.append(f"{name}: {why} {hit if why == 'never clear' else ''}".strip())
    return None, whys


def drive_until_takeover(sim, driver, chk):
    """Let the policy drive; return why it was stopped. chk sees the plant as it is each frame."""
    scn = sim.scn
    cuts = rule_cuts(scn)
    near_since = None
    while sim.ep.t < POLICY_MAX_S:
        sim.act(driver(sim.observation()).reshape(-1).cpu().numpy())
        if any(c.cut_t is not None for c in sim.ep.cutters.values()):
            return "the policy cut a leaf itself"
        chk.d.qpos[:] = scn.d.qpos
        hit = chk.hits(scn.d.qpos[scn.arm_adr], soft=False, firm=False, clearance=DANGER_M)
        if hit:
            return f"danger: {sorted(hit)[:2]}"
        p = tool_pos(scn)
        d_leaf = min((np.linalg.norm(p - scn.info["where"][key]["base"]) for key in cuts), default=np.inf)
        if d_leaf < HOVER_M:
            near_since = sim.ep.t if near_since is None else near_since
            if sim.ep.t - near_since >= HOVER_S:
                return "hovered near a leaf without cutting"
        else:
            near_since = None
    return "time"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--new", action="store_true", help="create --repo-id instead of appending to it")
    ap.add_argument("--rtc", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=None)
    a = ap.parse_args()
    from lerobot.utils.feature_utils import build_dataset_frame
    ckpt = str(pathlib.Path(a.ckpt).resolve())
    policy, pre, post = rp.load_policy(ckpt, rtc_horizon=20)
    driver = rp.RTCDriver(policy, pre, post, a.rtc)
    ds, tap, rop, root = ge.make_dataset(a.repo_id, resume=not a.new)
    n0 = ds.meta.total_episodes
    ge.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ge.LOG_DIR / f"corrections_{a.repo_id.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"policy {ckpt}\ndataset {a.repo_id} ({n0} episodes so far) at {root}\nlog {log_path}", flush=True)
    kept, rows, t_start = 0, 0, time.time()
    max_rows = a.max_rows or 10 * a.episodes
    with log_path.open("w") as log:
        try:
            for k in range(100_000):
                if kept >= a.episodes or rows >= max_rows:
                    break
                seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
                rng = np.random.default_rng(seeds[0])
                turns, found = gc.best_turns(seeds, rng)
                if not any(found):
                    continue
                rows += 1
                t0 = time.time()
                rec = dict(row=seeds[0], seeds=seeds)
                sim = rp.Sim(seeds, turns)
                driver.reset()
                chk, _ = sim.scn.checker("none_")
                why = drive_until_takeover(sim, driver, chk)
                rec.update(takeover=why, policy_s=round(sim.ep.t, 1))
                # step in: back off first if the arm is jammed, recording from this frame on
                r = ge.Recorder(ds, sim.scn, build_dataset_frame, rop, tap)
                sim.ep.cb = r.frame
                sim.ep.jaw = float(sim.scn.d.ctrl[sim.scn.jaw_act])
                sim.ep.I[:] = 0.0
                retreat, stuck = back_off(sim.scn, chk)
                if retreat is None:
                    ds.clear_episode_buffer()
                    r.close(); sim.cams.close()
                    rec.update(kept=False, why=f"no straight back-off clears the arm: {stuck}")
                    print(f"row {seeds[0]}: {why} -> stuck: {stuck} [{time.time() - t0:.0f} s]", flush=True)
                    log.write(json.dumps(rec, default=str) + "\n"); log.flush()
                    continue
                if len(retreat) > 1:
                    sim.ep.route("back_off", retreat, sim.ep.jaw)
                rec["backoff_m"] = round(BACKOFF_STEP_M * (len(retreat) - 1), 3)
                # the leaf to cut nearest the hand that the cutter can plan from here
                p = tool_pos(sim.scn)
                todo = sorted((key for key in rule_cuts(sim.scn) if sim.ep.cutters[key].cut_t is None),
                              key=lambda key: np.linalg.norm(p - sim.scn.info["where"][key]["base"]))
                plan, whys = None, []
                for key in todo:
                    plan = ct.plan_cut(sim.scn, rng, plant=key[0], only=key[1], q_from=sim.ep.q_now())
                    if isinstance(plan, ct.Plan):
                        break
                    whys.append(plan.tries[0][1] if plan.tries else plan.why)
                    plan = None
                if plan is None:
                    ds.clear_episode_buffer()
                    r.close(); sim.cams.close()
                    rec.update(kept=False, why=f"no plan after the back-off: {whys[:2]}")
                    print(f"row {seeds[0]}: {why} -> no plan after backing off {rec['backoff_m']} m: {whys[:2]} "
                          f"[{time.time() - t0:.0f} s]", flush=True)
                    log.write(json.dumps(rec, default=str) + "\n"); log.flush()
                    continue
                res = sim.ep.cut(plan)
                back = sim.ep.go_start()
                sim.ep.hold("end", ct.HOLD_S)
                r.close()
                sim.cams.close()
                ok = res["verdict"] == "PASS" and not back["pressed"]
                rec.update(cut=res, back=back, frames=r.n, wall_s=round(time.time() - t0, 1))
                if ok:
                    ds.save_episode()
                    rec.update(kept=True, episode_index=n0 + kept)
                    kept += 1
                    print(f"✅ correction {kept}/{a.episodes} (episode {n0 + kept - 1}): row {seeds[0]}, took over after "
                          f"{rec['policy_s']} s ({why}), cut {res['leaf']} stub {res['stub_mm']} mm, {r.n} frames "
                          f"[{rec['wall_s']} s]", flush=True)
                else:
                    ds.clear_episode_buffer()
                    rec.update(kept=False, why=res["why"] or back["pressed"])
                    print(f"row {seeds[0]}: {why} -> discarded ({rec['why']}) [{rec['wall_s']} s]", flush=True)
                log.write(json.dumps(rec, default=str) + "\n"); log.flush()
        finally:
            ds.finalize()
            summary = dict(summary=True, kept=kept, rows=rows, minutes=round((time.time() - t_start) / 60, 1),
                           episodes_now=n0 + kept, policy=ckpt)
            log.write(json.dumps(summary) + "\n")
    print(f"\nkept {kept} corrections from {rows} rows in {summary['minutes']} min; {a.repo_id} now has {n0 + kept} "
          f"episodes. Not pushed to the Hub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
