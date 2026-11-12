#!/usr/bin/env python3
"""Step 5 of the SHEAR sim plan: scripted deleafing passes recorded as a LeRobot dataset (K7 2026-11-03: "get
started on the episode gen").

  python generate_episodes.py --pilot                   5 episodes into krish5831/shear_sim_scripted_v1_pilot
  python generate_episodes.py --episodes 200            into krish5831/shear_sim_scripted_v1 (the v3 training set)
  python generate_episodes.py --episodes 200 --resume   append to it after a crash
  lerobot-dataset-viz --repo-id krish5831/shear_sim_scripted_v1_pilot --episode-index 0     what got recorded
  python generate_episodes.py --teacher --one-row 910057 --episodes 50 --no-dr --noise-share 1.0 \
      --repo-id krish5831/shear_sim_teacher_onerow_v1      post-mortem step 1: teacher v3, one row, noise on
  python generate_episodes.py --teacher --no-dr --hold-share 0.10 --row-start 930000 --episodes 20 \
      --max-attempts 900 --repo-id krish5831/shear_sim_teacher_v3_p00   post-mortem step 2: one part of dataset v3

--teacher (post-mortem steps 1+, 2026-11-11): the closed-loop teacher (shear_teacher.py, one cut per episode) instead
of the v1 pass, with DART-style noise as B.L.U.S.H's generate_truss.py: the servos get the teacher's command plus an
AR(1) noise per arm joint, the dataset records the clean command (the label). observation.environment_state = the cue
(shear_teacher.ENV_NAMES: the target's cut point and petiole direction in the arm base's frame). No start jitter, no
hold episodes or balance skips unless asked (--hold-share, step 2: rows where the rule cuts nothing or no cut is
cleanly reachable, the arm stays home, the cue all zeros; targets on p0 are skipped P0_ONLY_SKIP of the time,
since step 0 solved p0 32 vs p1 15).
--noise-share: share of rows with noise, drawn per row. --one-row: every episode the same row (only the noise differs).
Step 2 runs one process per 20 episodes (its own --row-start and repo id <repo>_pNN, merged after): B.L.U.S.H's
recorder grew to its 6 GB cap after 24-58 episodes (a dry run stays at ~2 GB, so the growth is in the recording
pass), and a dataset is unreadable until finalize (B.L.U.S.H lost 108 episodes that way, 2026-11-12).

One episode = one pass (cutter.run_pass): from the start pose, the rule's cuts on the three plants in front of
the arm, back to the start pose. The pots are set first (generate_cuts.best_turns). Every frame is recorded the
way lerobot-record does it, at 30 fps:
- observation.state: the 6 joints in use_degrees=false units through the follower's calibration (arm_map.py),
  so sim and real SHEAR data line up
- action: what the servos were told during that frame (the reference plus the sag correction), the leader's part
- observation.images.base / .claw (cameras.CAMERAS; v0 had .birdseye / .claw): 640x480 through the lens model,
  clean (no labels drawn: the policy sees what the real cameras see)
- robot_type so_follower, the real follower's, so sim and real datasets can be merged

Kept only if the judge passes the whole pass (every cut PASS, at least one cut, back at the start pose without
pressing anything). Every attempt, kept or not and why, goes to a JSONL log in ~/.cache/shear_sim/episodes/.

⚠️ Writes ~/.cache/huggingface/lerobot/<repo-id> and the log. Never pushes to the Hub (K7 approves every push).
Domain randomisation on by default (randomize.py: light, colours within class, camera poses; K7 2026-11-03);
--no-dr for the plain look. Run in the lerobot env under systemd-run MemoryMax (not ulimit -v).
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")      # ⚠️ cameras must render through EGL (B.L.U.S.H trap)

import numpy as np  # noqa: E402

import arm_map  # noqa: E402
import cutter as ct  # noqa: E402
import generate_cuts as gc  # noqa: E402
import shear_teacher as st  # noqa: E402
from cameras import CAMERAS  # noqa: E402  ("base", "claw") since the v1 rig

# ⚠️ LOCKED: v0 and v1 were recorded with it (their meta/tasks.parquet). Changing it makes earlier episodes a
# different task for the policy.
TASK = "Cut the leaves below the lowest ripening truss off flush at the stem."
# v1 (K7 2026-11-06): the v1 rig, base camera + claw, the arm's base 5 cm back (layout.py). v0 was birdseye + claw.
REPO_ID = "krish5831/shear_sim_scripted_v1"
PILOT_EPISODES = 5
# Data mix (K7 2026-11-06, from the SockBot series: the policy must also see "don't reach"):
HOLD_FRAC = 0.10             # at most this share of the batch: rows where the rule cuts NOTHING, the arm stays home
HOLD_S = 4.0                 # seconds the arm holds the start pose in such an episode
# rows with cuts on p0 only are skipped this often: v0 had 180 / 34 / 0 cuts on p0 / p1 / p2
P0_ONLY_SKIP = 0.5
# the arm starts this far off its start pose (uniform per joint); it still ends exactly on it
START_JITTER_DEG = 3.0
SEED_BASE = 900_000          # row attempt k uses seeds SEED_BASE + 3k .. +2 (apart from the trials' 700000s)
LOG_DIR = pathlib.Path.home() / ".cache" / "shear_sim" / "episodes"
# --teacher's noise (B.L.U.S.H generate_truss.py's values, 2026-11-11): per arm joint an AR(1) process, stationary std
# NOISE_NORM norm units (~1.5-1.8 deg), NOISE_RHO per frame (~0.3 s at 30 fps); its seed per episode is logged.
NOISE_NORM = 1.0                 # guess, unverified (B.L.U.S.H's)
NOISE_RHO = 0.9                  # guess, unverified
# Off on the slide in and from the close on: the cut completes on the slide out with the jaw shut, and the judge
# wants the arm back on the start pose. ⚠️ Half noise on the slide in pressed the stem with 40-83 N in 6 of 6
# episodes on row 910057 (2026-11-11): the slide is the 2 mm move. Half on the standoff (2 cm pushes failed there in
# check_teacher.py push, 1 cm recovered).
NOISE_OFF = {"slide_in", "close", "cut_wait", "slide_out", "to_start", "hold_end"}
NOISE_SCALE = {"standoff_in": 0.5}
TEACHER_ROW_START = 930_000      # --teacher without --one-row: rows from here (apart from 900000s, 910000s, 920000s)
NOISE_SHARE = 0.5                # --teacher: share of rows with noise (B.L.U.S.H step 2's; step 1 used 1.0)


class Noise:
    """B.L.U.S.H sim/generate_truss.py's Noise (commit fc2af15), per leg name: (leg) -> the offset (5,) in norm units
    added to the teacher's arm command (the jaw untouched)."""

    def __init__(self, sigma, rho, seed, episode):
        self.sigma, self.rho = sigma, rho
        self.rng = np.random.default_rng([seed, episode, 0xDA27])
        self.n = np.zeros(5)
        self.max = 0.0

    def __call__(self, leg):
        # steps every frame, also where it's off, so one leg's noise doesn't depend on the legs before
        self.n = self.rho * self.n + self.sigma * np.sqrt(1 - self.rho ** 2) * self.rng.normal(size=5)
        out = (0.0 if leg in NOISE_OFF else NOISE_SCALE.get(leg, 1.0)) * self.n
        self.max = max(self.max, float(np.abs(out).max()))
        return out


class Recorder:
    """Feeds one frame per Episode callback: the observation from the frame before, and the action that was sent
    during this one (lerobot-record's order: observe, then act)."""

    def __init__(self, ds, scn, build, rop, tap, env=None):
        from cameras import SimCameras
        self.ds, self.scn, self.build, self.rop, self.tap = ds, scn, build, rop, tap
        self.env = env                            # --teacher: observation.environment_state, the same every frame
        self.cams = SimCameras(scn.m)
        self.fm = arm_map.FollowerMap()
        self.prev = self.observe()
        self.n = 0

    def observe(self):
        d, scn = self.scn.d, self.scn
        q6 = np.r_[d.qpos[scn.arm_adr], d.qpos[scn.jaw_adr]]
        obs = {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, self.fm.rad_to_norm(q6))}
        for c in CAMERAS:
            obs[c] = self.cams.rgb(d, c)
        return obs

    def frame(self, phase, ep):
        from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STR
        d, scn = self.scn.d, self.scn
        # --teacher: the clean command (ep.label), not what the servos got (that has the noise in it)
        label = getattr(ep, "label", None)
        q6 = np.r_[d.ctrl[scn.act], d.ctrl[scn.jaw_act]] if label is None else label
        sent = {f"{j}.pos": float(v) for j, v in zip(arm_map.JOINTS, self.fm.rad_to_norm(q6))}
        obs_feats = {k: v for k, v in self.ds.features.items() if k != OBS_ENV_STATE}
        fr = {**self.build(obs_feats, self.rop(self.prev), prefix=OBS_STR),
              **self.build(self.ds.features, self.tap((sent, self.prev)), prefix=ACTION), "task": TASK}
        if self.env is not None:
            fr[OBS_ENV_STATE] = self.env
        self.ds.add_frame(fr)
        self.n += 1
        self.prev = self.observe()

    def close(self):
        self.cams.close()


def make_dataset(repo_id, resume, env_names=None):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets import LeRobotDataset, aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import HF_LEROBOT_HOME
    from lerobot.utils.feature_utils import combine_feature_dicts
    root = HF_LEROBOT_HOME / repo_id
    if root.exists() and not resume:
        sys.exit(f"⚠️ {root} exists already. --resume to append, or pick another --repo-id (nothing was written).")
    if resume and not root.exists():
        sys.exit(f"⚠️ --resume, but {root} doesn't exist.")
    tap, _, rop = make_default_processors()
    joints = {f"{j}.pos": float for j in arm_map.JOINTS}
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=tap, initial_features=create_initial_features(action=joints),
                                            use_videos=True),
        aggregate_pipeline_dataset_features(pipeline=rop, initial_features=create_initial_features(
            observation={**joints, **{c: (480, 640, 3) for c in CAMERAS}}), use_videos=True))
    if env_names:
        from lerobot.utils.constants import OBS_ENV_STATE
        features[OBS_ENV_STATE] = {"dtype": "float32", "shape": [len(env_names)], "names": env_names}
    enc = RGBEncoderConfig(vcodec="libsvtav1")                  # as B.L.U.S.H (AV1, lerobot-record's default)
    if resume:
        ds = LeRobotDataset.resume(repo_id, root=root, rgb_encoder=enc, image_writer_threads=4)
    else:
        # "so_follower": the real follower's robot_type, so sim and real SHEAR data merge (CLAUDE.md)
        ds = LeRobotDataset.create(repo_id, ct.FPS, root=root, robot_type="so_follower", features=features,
                                   use_videos=True, rgb_encoder=enc, image_writer_threads=4)
    return ds, tap, rop, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help=f"{PILOT_EPISODES} episodes into <repo-id>_pilot")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=None, help="rows tried at most (default 12 per episode)")
    ap.add_argument("--no-dr", action="store_true", help="no domain randomisation (the pilot's plain look)")
    ap.add_argument("--teacher", action="store_true", help="teacher v3 (one cut per episode) with noise, cue numbers")
    ap.add_argument("--one-row", type=int, default=None, help="--teacher: every episode this row (step 1)")
    ap.add_argument("--noise", type=float, default=NOISE_NORM, help="--teacher: noise std per arm joint, norm units")
    ap.add_argument("--noise-share", type=float, default=NOISE_SHARE,
                    help="--teacher: share of rows with noise, drawn per row (step 1 used 1.0)")
    ap.add_argument("--hold-share", type=float, default=0.0,
                    help="--teacher: share of hold episodes (rule cuts nothing, arm stays home); also turns on the "
                         "p0 balance skip")
    ap.add_argument("--row-start", type=int, default=TEACHER_ROW_START, help="--teacher: first row (step 2 parts)")
    a = ap.parse_args()
    if a.pilot:
        a.repo_id, n_eps = a.repo_id + "_pilot", a.episodes or PILOT_EPISODES
    elif a.episodes:
        n_eps = a.episodes
    else:
        sys.exit("--pilot, or --episodes N for a real batch (after K7 approved the pilot)")
    from lerobot.utils.feature_utils import build_dataset_frame
    ds, tap, rop, root = make_dataset(a.repo_id, a.resume, st.ENV_NAMES if a.teacher else None)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{a.repo_id.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"dataset {a.repo_id} at {root}\ntask {TASK!r}\nattempt log {log_path}", flush=True)
    if a.teacher:
        return teacher_batch(a, n_eps, ds, tap, rop, root, log_path, build_dataset_frame)
    done_rows = set()
    if a.resume:                                  # rows an earlier run already tried
        for p in LOG_DIR.glob(f"{a.repo_id.split('/')[-1]}_2*.jsonl"):
            done_rows |= {json.loads(ln).get("row") for ln in p.read_text().splitlines() if ln.strip()}
    kept, attempts, t_start = ds.meta.total_episodes if a.resume else 0, 0, time.time()
    n_hold = 0
    max_attempts = a.max_attempts or 12 * n_eps
    with log_path.open("w") as log:
        try:
            for k in range(10_000):
                if kept >= n_eps or attempts >= max_attempts:
                    break
                seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
                if seeds[0] in done_rows:
                    continue
                attempts += 1
                rng = np.random.default_rng(seeds[0])
                t0 = time.time()
                rec = dict(row=seeds[0], seeds=seeds)
                turns, found = gc.best_turns(seeds, rng)
                rule_cuts = sum(len(ct.sg.label(ct.sg.make_plant(sd, bend=True)).cuts) for sd in seeds)
                hold = rule_cuts == 0 and n_hold < HOLD_FRAC * n_eps
                if not any(found) and not hold:
                    rec.update(kept=False, why="no pot angle gives a clean cut" if rule_cuts
                               else "the rule cuts nothing (hold quota full)")
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"row {seeds[0]}: skipped, {rec['why']}", flush=True)
                    continue
                if not hold and found[1] + found[2] == 0 and rng.random() < P0_ONLY_SKIP:
                    rec.update(kept=False, why="balance: cuts on p0 only")
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"row {seeds[0]}: skipped for balance (cuts on p0 only)", flush=True)
                    continue
                scn = ct.Scene(seeds, turns=turns, domain_rand=not a.no_dr)
                rec["domain_rand"] = scn.dr
                r = Recorder(ds, scn, build_dataset_frame, rop, tap)
                if hold:
                    res = ct.hold_pass(scn, rng, r.frame, HOLD_S, START_JITTER_DEG)
                    n_hold += 1
                    rec["hold"] = True
                else:
                    res = ct.run_pass(scn, rng, r.frame, start_jitter_deg=START_JITTER_DEG)
                r.close()
                rec.update(pot_turns_deg=[round(float(np.degrees(t)) % 360, 1) for t in turns], frames=r.n,
                           n_cut=res["n_cut"], n_rule=res["n_rule"], cuts=res["cuts"], skipped=res["skipped"],
                           wall_s=round(time.time() - t0, 1))
                if res["verdict"] != "PASS":
                    ds.clear_episode_buffer()
                    why = [c["why"] for c in res["cuts"] if c["why"]] or res["back_to_start"]["pressed"] or ["no cut"]
                    rec.update(kept=False, why=why)
                    print(f"row {seeds[0]}: discarded ({why}) [{rec['wall_s']} s]", flush=True)
                else:
                    ds.save_episode()
                    rec.update(kept=True, episode_index=kept)
                    kept += 1
                    print(f"✅ episode {kept - 1} ({kept}/{n_eps}): row {seeds[0]}, {res['n_cut']} of the rule's "
                          f"{res['n_rule']} cuts, {r.n} frames ({r.n / ct.FPS:.0f} s) [{rec['wall_s']} s wall]",
                          flush=True)
                log.write(json.dumps(rec, default=str) + "\n")
                log.flush()
        finally:
            ds.finalize()
            summary = dict(summary=True, kept=kept, attempts=attempts, minutes=round((time.time() - t_start) / 60, 1))
            log.write(json.dumps(summary) + "\n")
    print(f"\nkept {kept} episodes from {attempts} rows in {summary['minutes']} min -> {root}")
    print("Not pushed to the Hub.")
    return 0


def teacher_batch(a, n_eps, ds, tap, rop, root, log_path, build):
    """--teacher: one cut per episode by teacher v3, noisy execution, clean labels, the cue as numbers."""
    if a.resume:
        sys.exit("--teacher doesn't --resume yet")
    kept, attempts, t_start = 0, 0, time.time()
    n_hold, max_hold = 0, round(a.hold_share * n_eps)
    turns_of = {}
    max_attempts = a.max_attempts or 3 * n_eps
    with log_path.open("w") as log:
        try:
            k = 0
            while kept < n_eps and attempts < max_attempts:
                row = a.one_row if a.one_row is not None else a.row_start + 3 * k
                k += 1
                seeds = tuple(row + j for j in range(3))
                attempts += 1
                t0 = time.time()
                rec = dict(row=row, seeds=seeds, teacher="v3", attempt=attempts)
                if row not in turns_of:     # check_teacher.scene_for's set-up (the rng is unused with route=False)
                    turns_of[row] = gc.best_turns(seeds, np.random.default_rng(0))
                turns, found = turns_of[row]
                # per row, so a part re-run decides the same (0xA015 / 0xBA1: their own streams, as B.L.U.S.H)
                noisy = a.noise > 0 and bool(np.random.default_rng([row, 0xA015]).random() < a.noise_share)
                hold = False
                if not any(found):
                    rule_cuts = sum(len(ct.sg.label(ct.sg.make_plant(sd, bend=True)).cuts) for sd in seeds)
                    # a hold: nothing to cut OR nothing cleanly reachable, either way the cue is zero and the arm
                    # stays home (only 4 of 300 rows have no rule cut at all, too few for 10 %)
                    hold = n_hold < max_hold
                    rec["hold_why"] = "the rule cuts nothing" if rule_cuts == 0 else "no pot angle gives a clean cut"
                    if not hold:
                        rec.update(kept=False, why="no pot angle gives a clean cut" if rule_cuts
                                   else "the rule cuts nothing" + (" (hold quota full)" if max_hold else ""))
                        log.write(json.dumps(rec) + "\n")
                        log.flush()
                        continue
                scn = ct.Scene(seeds, turns=turns, domain_rand=not a.no_dr)
                rec["domain_rand"] = scn.dr
                if hold:
                    env = np.zeros(len(st.ENV_NAMES), np.float32)
                    r = Recorder(ds, scn, build, rop, tap, env=env)
                    res = ct.hold_pass(scn, np.random.default_rng(row), r.frame, HOLD_S, 0.0)
                    r.close()
                    ds.save_episode()
                    n_hold += 1
                    rec.update(hold=True, frames=r.n, env=env.tolist(), kept=True, episode_index=kept,
                               wall_s=round(time.time() - t0, 1))
                    kept += 1
                    print(f"✅ episode {kept - 1} ({kept}/{n_eps}): row {row}, HOLD ({rec['hold_why']}), "
                          f"{r.n} frames [{rec['wall_s']} s wall]", flush=True)
                    log.write(json.dumps(rec, default=str) + "\n")
                    log.flush()
                    continue
                try:
                    T = st.ShearTeacher(scn).plan()
                except st.PlanError as e:
                    rec.update(kept=False, why=f"no plan: {e}")
                    log.write(json.dumps(rec, default=str) + "\n")
                    log.flush()
                    print(f"row {row}: no plan ({e})", flush=True)
                    continue
                if (a.hold_share > 0 and T.meta.get("plant") == 0
                        and np.random.default_rng([row, 0xBA1]).random() < P0_ONLY_SKIP):
                    rec.update(kept=False, why="balance: target on p0")
                    log.write(json.dumps(rec, default=str) + "\n")
                    log.flush()
                    print(f"row {row}: skipped for balance (target on p0)", flush=True)
                    continue
                env = st.cue_env(scn, T.meta)
                noise = Noise(a.noise, NOISE_RHO, row, attempts) if noisy else None
                r = Recorder(ds, scn, build, rop, tap, env=env)
                res = st.run(scn, T, r.frame, noise=noise)
                r.close()
                rec.update(pot_turns_deg=[round(float(np.degrees(x)) % 360, 1) for x in turns], frames=r.n,
                           env=env.round(4).tolist(), plan=T.meta, result=res, wall_s=round(time.time() - t0, 1),
                           noise=None if noise is None else dict(std_norm=a.noise, rho=NOISE_RHO,
                                                                 seed=[row, attempts, 0xDA27],
                                                                 max_norm=round(noise.max, 2)))
                if res["verdict"] != "PASS":
                    ds.clear_episode_buffer()
                    rec.update(kept=False, why=res["why"])
                    print(f"row {row} attempt {attempts}: discarded ({res['why']}) [{rec['wall_s']} s]", flush=True)
                else:
                    ds.save_episode()
                    rec.update(kept=True, episode_index=kept)
                    kept += 1
                    print(f"✅ episode {kept - 1} ({kept}/{n_eps}): row {row}, stub {res['stub_mm']} mm, {r.n} frames "
                          f"({r.n / ct.FPS:.1f} s) [{rec['wall_s']} s wall]", flush=True)
                log.write(json.dumps(rec, default=str) + "\n")
                log.flush()
        finally:
            ds.finalize()
            summary = dict(summary=True, kept=kept, attempts=attempts, minutes=round((time.time() - t_start) / 60, 1))
            log.write(json.dumps(summary) + "\n")
    print(f"\nkept {kept} episodes from {attempts} attempts in {summary['minutes']} min -> {root}")
    print("Not pushed to the Hub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
