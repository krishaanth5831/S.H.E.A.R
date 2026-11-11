#!/usr/bin/env python3
"""One-cut evaluation of a policy trained on teacher v3 data (BLUSH + SHEAR post-mortem, steps 1+, 2026-11-11).

  python eval_onecut.py <ckpt>/pretrained_model --row 910057 --rollouts 1                  step 1: clean, no noise
  python eval_onecut.py <ckpt>/pretrained_model --row 910057 --rollouts 19 --noise 1.0     step 1: with noise
  python eval_onecut.py ... --n-action-steps 100 | 25          how the chunks are played (the execution sweep)
  python eval_onecut.py ... --ensemble 0.01                    ACT temporal ensembling, a new chunk every frame
  python eval_onecut.py ... --rollouts 1 --windowed            watch the policy drive in a MuJoCo window

Each rollout sets the row up as generate_episodes.py --teacher recorded it (check_teacher.scene_for's pots, no domain
randomisation), lets teacher v3 plan it only to know the target leaf, its cue and its cut pose, and then the POLICY
drives from the start pose at 30 fps: every frame the joint state (use_degrees=false), the base + claw frames, the
cue as observation.environment_state (shear_teacher.cue_env), and its action goes to the servos.

--noise adds the dataset's noise (generate_episodes.Noise) to the policy's arm commands, off within NOISE_NEAR_M of the
cut point and after the cut (as B.L.U.S.H's eval_truss.py --noise).

Judged as shear_teacher.run judges the teacher: success = the target leaf cut with a flush stub (cutter.STUB_OK), no
other leaf cut, nothing pressed over cutter.FORCE_LIMIT, no blade on a truss stalk, the arm back on the start pose.
Also the closest approach of the tool point (the blades' cut point) to where the teacher's cut pose puts it, in mm.

Writes one JSON line per rollout and a summary to ~/.cache/shear_sim/eval/<run>_<step>_<time>.jsonl. Run outside
Claude Code's sandbox (the GPU, EGL), in the lerobot env, under systemd-run MemoryMax.
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
LOG_DIR = pathlib.Path.home() / ".cache" / "shear_sim" / "eval"


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


def rollout(policy, pre, post, meta, wants_env, seeds, turns, noise=None, viewer=None):
    import torch
    from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    from cameras import SimCameras
    scn = ct.Scene(seeds, turns=turns)
    T = st.ShearTeacher(scn).plan()               # only for the target, its cue and its cut pose
    p = T.plan_cut
    key = (p.plant, p.node)
    env = st.cue_env(scn, T.meta)
    cut_at = tool_point(scn, p, np.asarray(p.slide[-1]))
    cams, fm = SimCameras(scn.m), arm_map.FollowerMap()
    ep = ct.Episode(scn, None)
    ep.target = key
    ep.mon.reset(p.prefix)
    ep.mon.phase = "policy"
    tgt = ep.cutters[key]
    _, _, rop = make_default_processors()
    obs_feats = {k: v for k, v in meta.features.items() if k != OBS_ENV_STATE}
    m, d = scn.m, scn.d
    live = viewer(scn) if viewer else None
    policy.reset()
    closest, still, frames, t_cut = np.inf, 0, 0, None
    while ep.t < MAX_S:
        q6 = np.r_[d.qpos[scn.arm_adr], d.qpos[scn.jaw_adr]]
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
                if c.attached:
                    c.step()
            if s % 5 == 0:
                ep.mon.step()
        ep.t += 1.0 / ct.FPS
        frames += 1
        if t_cut is None and tgt.cut_t is not None:
            t_cut = round(ep.t, 2)
        if live and not live():
            break
        home = float(np.abs(ep.q_now() - scn.start).max()) < st.END_START_TOL
        still = still + 1 if (tgt.cut_t is not None and home) else 0
        if still >= STILL_S * ct.FPS:
            break
    why = judge(ep, scn, key, tgt)
    if live:
        live(close=True)
    cams.close()
    return dict(success=not why, why=why, cut=tgt.cut_t is not None,
                stub_mm=None if tgt.stub is None else round(tgt.stub * 1000, 2),
                closest_mm=round(closest * 1000, 1), t_cut=t_cut, seconds=round(ep.t, 2), frames=frames,
                back_at_start=bool(still >= STILL_S * ct.FPS), leaf=T.meta.get("leaf"), plant=p.plant,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", type=pathlib.Path, help="a checkpoint's pretrained_model folder")
    ap.add_argument("--row", type=int, required=True, help="the row (seeds row, row+1, row+2)")
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--noise", type=float, default=0.0, help="noise std per arm joint, norm units (0 = off)")
    ap.add_argument("--ensemble", type=float, default=None, help="override temporal_ensemble_coeff (ACT)")
    ap.add_argument("--n-action-steps", type=int, default=None, help="override n_action_steps (ACT, no ensembling)")
    ap.add_argument("--windowed", action="store_true", help="watch it live in a MuJoCo window (real time)")
    a = ap.parse_args()
    import torch
    from lerobot.utils.constants import OBS_ENV_STATE
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
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
    seeds = tuple(a.row + j for j in range(3))
    turns, found = gc.best_turns(seeds, np.random.default_rng(0))   # check_teacher.scene_for's set-up
    if not any(found):
        sys.exit(f"row {a.row}: no pot angle gives a clean cut")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run = path.parent.parent.parent.name
    log_path = LOG_DIR / f"{run}_{path.parent.name}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"policy {path} ({cfg.type}) on {device}, {mode}; cue as numbers: {wants_env}; row {a.row}; "
          f"noise {a.noise}\nlog {log_path}", flush=True)
    results = []
    with log_path.open("w") as log:
        for i in range(a.rollouts):
            noise = ge.Noise(a.noise, ge.NOISE_RHO, a.row, 1000 + i) if a.noise > 0 else None
            r = rollout(policy, pre, post, meta, wants_env, seeds, turns, noise, open_viewer if a.windowed else None)
            r.update(row=a.row, rollout=i, mode=mode)
            results.append(r)
            log.write(json.dumps(r, default=str) + "\n")
            log.flush()
            ok = sum(x["success"] for x in results)
            print(f"  {'✅' if r['success'] else '❌'} rollout {i}: "
                  f"{'cut' if r['cut'] else 'not cut'}, stub {r['stub_mm']} mm, closest {r['closest_mm']} mm, "
                  f"cut at {r['t_cut']} s, {r['seconds']} s; {'; '.join(r['why']) or 'clean'}   "
                  f"[{ok}/{len(results)} so far]", flush=True)
        n, ok = len(results), sum(x["success"] for x in results)
        lo_, hi_ = wilson(ok, n)
        summary = dict(summary=True, n=n, success=ok, wilson95=[round(lo_, 3), round(hi_, 3)],
                       cut=sum(x["cut"] for x in results), mode=mode, noise=a.noise,
                       closest_median_mm=float(np.median([x["closest_mm"] for x in results])),
                       t_cut=sorted(x["t_cut"] for x in results if x["t_cut"] is not None))
        log.write(json.dumps(summary) + "\n")
    print(f"\n{ok}/{n} successes ({100 * ok / n:.0f} %, 95 % {100 * lo_:.0f}-{100 * hi_:.0f} %), target cut in "
          f"{summary['cut']}; closest approach median {summary['closest_median_mm']} mm.  log {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
