#!/usr/bin/env python3
"""One-cut evaluation of a policy trained on teacher v3 data (BLUSH + SHEAR post-mortem, steps 1+, 2026-11-11).

  python eval_onecut.py <ckpt>/pretrained_model --row 910057 --rollouts 1                  step 1: clean, no noise
  python eval_onecut.py <ckpt>/pretrained_model --row 910057 --rollouts 19 --noise 1.0     step 1: with noise
  python eval_onecut.py <ckpt>/pretrained_model --rows 20 --n-action-steps 100             step 3: 20 held-out rows
  python eval_onecut.py <ckpt>/pretrained_model --rows 20 --holds 5                        ... plus 5 hold rows
  python eval_onecut.py <ckpt>/pretrained_model --rows 20 --hybrid                         ... the hybrid (6.5)
  python eval_onecut.py ... --n-action-steps 100 | 25          how the chunks are played (the execution sweep)
  python eval_onecut.py ... --ensemble 0.01                    ACT temporal ensembling, a new chunk every frame
  python eval_onecut.py ... --rollouts 1 --windowed            watch the policy drive in a MuJoCo window

Each rollout sets the row up as generate_episodes.py --teacher recorded it (check_teacher.scene_for's pots, no domain
randomisation), lets teacher v3 plan it only to know the target leaf, its cue and its cut pose, and then the POLICY
drives from the start pose at 30 fps: every frame the joint state (use_degrees=false), the base + claw frames, the
cue as observation.environment_state (shear_teacher.cue_env), and its action goes to the servos.

--rows N: N rows the teacher can cut, from --row-start upwards in steps of 3 (the generator's spacing; the default
start 1100000 is outside every recorded range: step 1 used 910057, step 2 rows 930000 to about 1020000). A row where
no pot angle gives a clean cut, or the teacher finds no plan, is not teacher-feasible: skipped and counted, never in
the denominator (the step-3 gate is "35 % of teacher-feasible cuts"). --holds K runs the first K such rows as holds
(the cue all zeros, as the dataset's hold episodes): success = nothing cut, nothing pressed, the arm still on the
start pose after HOLD_EVAL_S. Holds are reported on their own, not in the gate.

--hybrid (post-mortem 6.5, K7's fallback 2026-11-05): the policy drives until the blades' cut point is within
--handoff-mm of where the teacher's PRE-CUT pose puts it (4 cm back along the fingers, shear_teacher's slide start);
then the teacher's own legs finish: a HANDOFF_S hold on the pre-cut pose while the jaw opens, the slide in, close,
cut wait, slide out and home, with no noise. ⚠️ The slide is planned from the exact cut point, so in sim the hybrid
only tests the approach; the honest test perturbs the cue by the detector's error (not built yet).

--noise adds the dataset's noise (generate_episodes.Noise) to the policy's arm commands, off within NOISE_NEAR_M of the
cut point and after the cut (as B.L.U.S.H's eval_truss.py --noise).

Judged as shear_teacher.run judges the teacher: success = the target leaf cut with a flush stub (cutter.STUB_OK), no
other leaf cut, nothing pressed over cutter.FORCE_LIMIT, no blade on a truss stalk, the arm back on the start pose.
Also the closest approach of the tool point (the blades' cut point) to where the teacher's cut pose puts it, in mm.

Writes one JSON line per rollout and a summary to ~/.cache/shear_sim/eval/<run>_<step>_<time>.jsonl. Run outside
Claude Code's sandbox (the GPU, EGL), in the lerobot env, under systemd-run MemoryMax. CUDA_VISIBLE_DEVICES="" keeps
it on the CPU while the GPU trains (pick_checkpoint.py --watch --cpu).
"""
import argparse
import json
import os
import pathlib
import sys
import time

# ⚠️ Cameras render through EGL headless (B.L.U.S.H trap), but EGL kills the viewer window: --windowed uses GLFW.
os.environ.setdefault("MUJOCO_GL", "glfw" if "--windowed" in sys.argv else "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import arm_map  # noqa: E402
import cutter as ct  # noqa: E402
import generate_cuts as gc  # noqa: E402
import generate_episodes as ge  # noqa: E402
import shear_teacher as st  # noqa: E402

MAX_S = 30.0                  # the teacher needs ~10.6 s on row 910057 (check_teacher solve, 2026-11-10); about 3x
STILL_S = 0.5                 # after the cut: back on the start pose this long ends the rollout
NOISE_NEAR_M = 0.03           # no noise this close to the cut point (B.L.U.S.H eval_truss.py's value)
HOLD_EVAL_S = 6.0             # a hold row: the dataset's hold episodes are 5 s (0.5 + 4 + 0.5), plus a second
HANDOFF_S = 0.5               # --hybrid: held on the pre-cut pose while the jaw opens before the slide (guess)
HELD_OUT_ROW_START = 1_100_000   # --rows: past step 2's rows (930000 + 3000 * 30 parts, about 1020000)
LOG_DIR = pathlib.Path.home() / ".cache" / "shear_sim" / "eval"
# --rows: each row's pot angles and whether the teacher can cut it. ⚠️ Delete after a change to the teacher, the
# plants or the pot-angle search: a stale "can cut" row is caught (the plan fails, the row is skipped), a stale
# "can't cut" row is not.
ROW_CACHE = LOG_DIR / "row_plans.json"


def load_policy(path, device, overrides=None):
    """B.L.U.S.H sim/eval_truss.py's load_policy (commit fc2af15). A checkpoint via lerobot's make_policy, as
    trained. overrides: config fields for this run (the execution sweep: temporal_ensemble_coeff, n_action_steps).
    Returns (policy, pre, post, cfg, dataset metadata)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import make_policy
    from lerobot.utils.constants import HF_LEROBOT_HOME
    repo = json.loads((path / "train_config.json").read_text())["dataset"]["repo_id"]
    meta = LeRobotDatasetMetadata(repo, root=HF_LEROBOT_HOME / repo)
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = str(path)
    cfg.device = str(device)
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    policy = make_policy(cfg, ds_meta=meta).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": str(device)}})
    return policy, pre, post, cfg, meta


def tool_point(scn, p, q):
    """Where the blades cut (the tool point of cutter.plan_cut's pose p) with the arm at joints q (5, rad)."""
    o, R = scn.ik.fk(q)
    return o + R @ ct.cp.tool_point(p.petiole_r, ct.CUT_CLEARANCE, face=p.pose.face)


def judge(ep, scn, key, tgt):
    """shear_teacher.run's verdict, for whatever drove the arm."""
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
    if off > st.END_START_TOL:
        why.append(f"arm {np.degrees(off):.1f} deg off the start pose at the end")
    return why


def judge_hold(ep, scn):
    """A hold row: the arm had nothing to cut and must have stayed home."""
    mon, why = ep.mon, []
    cut = [k for k, c in ep.cutters.items() if c.cut_t is not None]
    if cut:
        why.append(f"cut {cut} in a hold row")
    for k, lim in ct.FORCE_LIMIT.items():
        if mon.peak[k] > lim:
            why.append(f"pressed the {k} with {mon.peak[k]:.1f} N ({mon.where.get(k)})")
    off = float(np.abs(ep.q_now() - scn.start).max())
    if off > st.END_START_TOL:
        why.append(f"arm {np.degrees(off):.1f} deg off the start pose at the end")
    return why


def rollout(policy, pre, post, meta, wants_env, seeds, turns, noise=None, viewer=None, handoff=None, hold=False,
            kick=0):
    """One rollout on the row seeds. handoff: --hybrid's hand-off distance in metres (None = the policy all the way).
    hold: a hold row (cue zeros, HOLD_EVAL_S, judge_hold). kick: --kick-s in frames, the teacher drives a cut row's
    first frames and the policy takes over (a diagnostic, not a success measure)."""
    import torch
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    from cameras import SimCameras
    scn = ct.Scene(seeds, turns=turns)
    cams, fm = SimCameras(scn.m), arm_map.FollowerMap()
    ep = ct.Episode(scn, None)
    if hold:
        T = p = key = tgt = None
        env = np.zeros(len(st.ENV_NAMES), np.float32)
        ep.mon.reset("")
    else:
        T = st.ShearTeacher(scn).plan()               # the target, its cue and its cut pose (and --hybrid's legs)
        p = T.plan_cut
        key = (p.plant, p.node)
        env = st.cue_env(scn, T.meta)
        cut_at = tool_point(scn, p, np.asarray(p.slide[-1]))
        pre_at = tool_point(scn, p, np.asarray(p.slide[0]))   # the pre-cut pose: --hybrid's hand-off point
        ep.target = key
        ep.mon.reset(p.prefix)
        tgt = ep.cutters[key]
    ep.mon.phase = "policy"
    _, _, rop = make_default_processors()
    obs_feats = {k: v for k, v in meta.features.items() if k != OBS_ENV_STATE}
    m, d = scn.m, scn.d
    live = viewer(scn) if viewer else None
    policy.reset()
    closest, still, frames, t_cut = np.inf, 0, 0, None
    handed, t_hand, mm_hand = None, None, None          # --hybrid: the frame the teacher took over
    while ep.t < (HOLD_EVAL_S if hold else MAX_S):
        q6 = np.r_[d.qpos[scn.arm_adr], d.qpos[scn.jaw_adr]]
        if handoff is not None and handed is None:
            d_pre = float(np.linalg.norm(tool_point(scn, p, q6[:5]) - pre_at))
            if d_pre < handoff:
                # the teacher's legs from the slide in, after a hold on the pre-cut pose that opens the jaw
                # (to_leaf's jaw_to did that for the teacher; the policy may arrive with it anywhere)
                k = next(i for i, g in enumerate(T.legs) if g.name == "slide_in")
                bridge = st.Leg("handoff", hold=np.asarray(p.slide[0]), servo=True,
                                jaw_to=(float(q6[5]), np.radians(ct.JAW_APPROACH_DEG)),
                                frames=max(1, round(HANDOFF_S * ct.FPS)))
                T.legs = [bridge] + T.legs[k:]
                T.reset()
                handed, t_hand, mm_hand = frames, round(ep.t, 2), round(d_pre * 1000, 1)
        if handed is not None:
            # ⚠️ shear_teacher.run's rule: the cut completes on the slide out; going home with the leaf still on
            # would stall holding it, so stop there (the judge fails it anyway)
            if T.done or (T.leg == "to_start" and tgt.cut_t is None):
                break
            a5, jaw = T.act(q6[:5])
            ep.mon.phase = T.leg
            d.ctrl[scn.act] = a5
            d.ctrl[scn.jaw_act] = jaw
        elif not hold and frames < kick:
            a5, jaw = T.act(q6[:5])                   # --kick-s: the teacher's own start, clean (plan() has reset it)
            d.ctrl[scn.act] = a5
            d.ctrl[scn.jaw_act] = jaw
        else:
            obs = {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, fm.rad_to_norm(q6))}
            for c in ge.CAMERAS:
                obs[c] = cams.rgb(d, c)
            frame = build_dataset_frame(obs_feats, rop(obs), prefix=OBS_STR)
            if wants_env:
                frame[OBS_ENV_STATE] = env
            with torch.inference_mode():
                x = prepare_observation_for_inference(frame, policy.config.device, ge.TASK, "so_follower")
                act = make_robot_action(post(policy.select_action(pre(x))), meta.features)
            a = np.array([act[f"{j}.pos"] for j in arm_map.JOINTS], float)
            if not hold:
                dist = float(np.linalg.norm(tool_point(scn, p, q6[:5]) - cut_at))
                if tgt.cut_t is None:
                    closest = min(closest, dist)
                if noise is not None:
                    near = dist < NOISE_NEAR_M or tgt.cut_t is not None
                    a[:5] += noise("close" if near else "to_leaf")          # "close" is in NOISE_OFF: steps, adds 0
            q = fm.norm_to_rad(a)
            d.ctrl[scn.act] = q[:5]
            d.ctrl[scn.jaw_act] = q[5]
        for s in range(ep.steps_per_frame):
            mujoco.mj_step(m, d)
            for c in ep.cutters.values():
                # ⚠️ also after the cut while a nearer blade pair's second cut is pending (plant.Cutter._cut), as
                # shear_teacher.run steps them: stepping only attached cutters missed it and read the far pair's
                # 13-17 mm stub (replay_teacher, 2026-11-13: 2 of 5 clean teacher episodes)
                if c.attached or c.pending:
                    c.step()
            if s % 5 == 0:
                ep.mon.step()
        ep.t += 1.0 / ct.FPS
        frames += 1
        if not hold and t_cut is None and tgt.cut_t is not None:
            t_cut = round(ep.t, 2)
        if live and not live():
            break
        if not hold:
            home = float(np.abs(ep.q_now() - scn.start).max()) < st.END_START_TOL
            still = still + 1 if (tgt.cut_t is not None and home) else 0
            if still >= STILL_S * ct.FPS:
                break
    if live:
        live(close=True)
    cams.close()
    if hold:
        why = judge_hold(ep, scn)
        return dict(kind="hold", success=not why, why=why, cut=any(c.cut_t is not None for c in ep.cutters.values()),
                    seconds=round(ep.t, 2), frames=frames, noise=None if noise is None else noise.sigma)
    why = judge(ep, scn, key, tgt)
    return dict(kind="cut", success=not why, why=why, cut=tgt.cut_t is not None,
                stub_mm=None if tgt.stub is None else round(tgt.stub * 1000, 2),
                closest_mm=round(closest * 1000, 1), t_cut=t_cut, seconds=round(ep.t, 2), frames=frames,
                back_at_start=bool(still >= STILL_S * ct.FPS), leaf=T.meta.get("leaf"), plant=p.plant,
                hybrid=handoff is not None, handoff_t=t_hand, handoff_mm=mm_hand, kick_frames=kick,
                policy_frames=frames if handed is None else handed,
                noise=None if noise is None else noise.sigma, noise_max=None if noise is None else round(noise.max, 2))


def open_viewer(scn):
    import mujoco.viewer
    v = mujoco.viewer.launch_passive(scn.m, scn.d)
    v.cam.lookat[:] = (ct.lo.PLANT_X[0], (ct.lo.ROW_Y + ct.lo.AISLE_Y) / 2, 0.22)
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.75, 245, -22
    last = [time.time()]

    def frame(close=False):
        if close or not v.is_running():
            v.close()
            return False
        v.sync()
        wait = 1.0 / ct.FPS - (time.time() - last[0])
        if wait > 0:
            time.sleep(wait)
        last[0] = time.time()
        return True
    return frame


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def row_plan(row, cache=None):
    """(seeds, turns, why) for a row: why is None when the teacher can cut it, else the reason it can't.
    cache: ROW_CACHE's dict, read and filled (every checkpoint of a run tests the same rows, and the pot-angle search
    plus the plan took about a third of a 3-row run, 2026-11-14)."""
    seeds = tuple(row + j for j in range(3))
    if cache is not None and str(row) in cache:
        c = cache[str(row)]
        return seeds, tuple(c["turns"]), c["why"]       # a tuple as best_turns gives: layout.build tests `if turns`
    turns, found = gc.best_turns(seeds, np.random.default_rng(0))   # check_teacher.scene_for's set-up
    why = None
    if not any(found):
        why = "no pot angle gives a clean cut"
    else:
        try:
            st.ShearTeacher(ct.Scene(seeds, turns=turns)).plan()
        except st.PlanError as e:
            why = f"no plan: {e}"
    if cache is not None:
        cache[str(row)] = dict(turns=[float(x) for x in turns], why=why)
        ROW_CACHE.write_text(json.dumps(cache))
    return seeds, turns, why


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", type=pathlib.Path, help="a checkpoint's pretrained_model folder")
    ap.add_argument("--row", type=int, default=None, help="one row (seeds row, row+1, row+2), --rollouts times")
    ap.add_argument("--rows", type=int, default=None, help="this many teacher-feasible rows from --row-start")
    ap.add_argument("--row-start", type=int, default=HELD_OUT_ROW_START, help="--rows: the first row, then +3")
    ap.add_argument("--holds", type=int, default=0, help="--rows: the first K rows the teacher can't cut too, as holds")
    ap.add_argument("--rollouts", type=int, default=1, help="per row")
    ap.add_argument("--noise", type=float, default=0.0, help="noise std per arm joint, norm units (0 = off)")
    ap.add_argument("--ensemble", type=float, default=None, help="override temporal_ensemble_coeff (ACT)")
    ap.add_argument("--n-action-steps", type=int, default=None, help="override n_action_steps (ACT, no ensembling)")
    ap.add_argument("--hybrid", action="store_true", help="the policy to the pre-cut pose, the teacher's slide after")
    ap.add_argument("--handoff-mm", type=float, default=10.0, help="--hybrid: hand over this close to the pre-cut pose")
    ap.add_argument("--kick-s", type=float, default=0.0,
                    help="diagnostic: the teacher drives the first K seconds of a cut row, then the policy (2026-11-15: "
                         "act_teacher_v3 never left the start pose; this shows what the rest of the policy can do)")
    ap.add_argument("--windowed", action="store_true", help="watch it live in a MuJoCo window (real time)")
    a = ap.parse_args()
    if (a.row is None) == (a.rows is None):
        ap.error("one of --row or --rows")
    import torch
    from lerobot.utils.constants import OBS_ENV_STATE
    # ⚠️ No register_third_party_plugins(): it imports B.L.U.S.H's so101_mujoco plugin, whose `cameras` import then
    # finds THIS repo's sim/cameras.py (ImportError RIG_CAMERAS, 2026-11-11). The eval drives cutter.Scene directly.
    path = a.policy.resolve()
    over = {}
    if a.ensemble is not None:
        over.update(temporal_ensemble_coeff=a.ensemble if a.ensemble > 0 else None,
                    n_action_steps=1 if a.ensemble > 0 else (a.n_action_steps or 100))
    elif a.n_action_steps is not None:
        over.update(temporal_ensemble_coeff=None, n_action_steps=a.n_action_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, pre, post, cfg, meta = load_policy(path, device, over)
    wants_env = OBS_ENV_STATE in getattr(cfg, "input_features", {})
    mode = (f"ensemble {cfg.temporal_ensemble_coeff}" if getattr(cfg, "temporal_ensemble_coeff", None)
            else f"n_action_steps {cfg.n_action_steps}")
    handoff = a.handoff_mm / 1000 if a.hybrid else None
    if a.kick_s > 0:
        mode += f", teacher's first {a.kick_s} s"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run = path.parent.parent.parent.name
    # the mode and the process id are in the name: two evals of one checkpoint started in the same second (policy alone
    # and hybrid, 2026-11-15) opened the same file and wrote over each other
    log_path = LOG_DIR / (f"{run}_{path.parent.name}{'_hybrid' if a.hybrid else ''}_"
                          f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl")
    where = f"row {a.row}" if a.row is not None else f"{a.rows} rows from {a.row_start} (+{a.holds} holds)"
    print(f"policy {path} ({cfg.type}) on {device}, {mode}{', HYBRID' if a.hybrid else ''}; cue as numbers: "
          f"{wants_env}; {where}; noise {a.noise}\nlog {log_path}", flush=True)
    results, skipped = [], []
    viewer = open_viewer if a.windowed else None
    cache = (json.loads(ROW_CACHE.read_text()) if ROW_CACHE.exists() else {}) if a.rows is not None else None
    with log_path.open("w") as log:
        row, n_cut, n_hold = (a.row if a.row is not None else a.row_start) - 3, 0, 0
        while (a.row is not None and n_cut == 0) or (a.rows is not None and (n_cut < a.rows or n_hold < a.holds)):
            row += 3
            seeds, turns, why = row_plan(row, cache)
            if why is not None and a.row is not None:
                sys.exit(f"row {a.row}: {why}")
            hold = why is not None
            if hold and n_hold >= a.holds:
                skipped.append(dict(row=row, why=why))
                print(f"  row {row}: skipped, {why[:90]}", flush=True)
                continue
            if not hold and a.rows is not None and n_cut >= a.rows:
                continue
            for i in range(a.rollouts):
                noise = ge.Noise(a.noise, ge.NOISE_RHO, row, 1000 + i) if a.noise > 0 else None
                try:
                    r = rollout(policy, pre, post, meta, wants_env, seeds, turns, noise, viewer, handoff, hold,
                                kick=round(a.kick_s * ct.FPS))
                except st.PlanError as e:             # a stale ROW_CACHE entry: the teacher can't cut it any more
                    skipped.append(dict(row=row, why=f"cached as cuttable, but no plan: {e}"))
                    print(f"  row {row}: skipped, the cache says cuttable but the teacher finds no plan", flush=True)
                    break
                r.update(row=row, rollout=i, mode=mode)
                results.append(r)
                log.write(json.dumps(r, default=str) + "\n")
                log.flush()
                cuts = [x for x in results if x["kind"] == "cut"]
                ok = sum(x["success"] for x in cuts)
                if hold:
                    print(f"  {'✅' if r['success'] else '❌'} row {row} HOLD: "
                          f"{'; '.join(r['why']) or 'stayed home'}", flush=True)
                else:
                    hand = "" if not a.hybrid else (f", hand-off at {r['handoff_t']} s ({r['handoff_mm']} mm)"
                                                    if r["handoff_t"] is not None else ", no hand-off")
                    print(f"  {'✅' if r['success'] else '❌'} row {row} rollout {i}: "
                          f"{'cut' if r['cut'] else 'not cut'}, stub {r['stub_mm']} mm, closest {r['closest_mm']} mm, "
                          f"cut at {r['t_cut']} s{hand}, {r['seconds']} s; "
                          f"{'; '.join(r['why']) or 'clean'}   [{ok}/{len(cuts)} so far]", flush=True)
            else:
                n_hold += hold
                n_cut += not hold
        cuts = [x for x in results if x["kind"] == "cut"]
        holds = [x for x in results if x["kind"] == "hold"]
        n, ok = len(cuts), sum(x["success"] for x in cuts)
        lo_, hi_ = wilson(ok, n)
        stubs = sorted(x["stub_mm"] for x in cuts if x["stub_mm"] is not None)
        summary = dict(summary=True, n=n, success=ok, wilson95=[round(lo_, 3), round(hi_, 3)],
                       cut=sum(x["cut"] for x in cuts), mode=mode, noise=a.noise, hybrid=a.hybrid,
                       handoff_mm=a.handoff_mm if a.hybrid else None,
                       handed_off=sum(x.get("handoff_t") is not None for x in cuts) if a.hybrid else None,
                       closest_median_mm=float(np.median([x["closest_mm"] for x in cuts])) if cuts else None,
                       stub_median_mm=stubs[len(stubs) // 2] if stubs else None,
                       t_cut=sorted(x["t_cut"] for x in cuts if x["t_cut"] is not None),
                       holds=len(holds), holds_ok=sum(x["success"] for x in holds),
                       skipped=len(skipped), row_start=a.row if a.row is not None else a.row_start, last_row=row)
        log.write(json.dumps(summary) + "\n")
    extra = f"; holds {summary['holds_ok']}/{summary['holds']}" if holds else ""
    extra += f"; hand-off in {summary['handed_off']}" if a.hybrid else ""
    extra += f"; {len(skipped)} rows not teacher-feasible" if skipped else ""
    print(f"\n{ok}/{n} successes ({100 * ok / max(n, 1):.0f} %, 95 % {100 * lo_:.0f}-{100 * hi_:.0f} %), target cut in "
          f"{summary['cut']}; closest approach median {summary['closest_median_mm']} mm{extra}.  log {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
