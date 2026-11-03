#!/usr/bin/env python3
"""Step 5 of the SHEAR sim plan: scripted deleafing passes recorded as a LeRobot dataset (K7 2026-11-03: "get
started on the episode gen").

  python generate_episodes.py --pilot                   5 episodes into krish5831/shear_sim_scripted_v0_pilot
  python generate_episodes.py --episodes 200            into krish5831/shear_sim_scripted_v0
  python generate_episodes.py --episodes 200 --resume   append to it after a crash
  lerobot-dataset-viz --repo-id krish5831/shear_sim_scripted_v0_pilot --episode-index 0     what got recorded

One episode = one pass (cutter.run_pass): from the start pose, the rule's cuts on the three plants in front of
the arm, back to the start pose. The pots are set first (generate_cuts.best_turns). Every frame is recorded the
way lerobot-record does it, at 30 fps:
- observation.state: the 6 joints in use_degrees=false units through the follower's calibration (arm_map.py),
  so sim and real SHEAR data line up
- action: what the servos were told during that frame (the reference plus the sag correction), the leader's part
- observation.images.birdseye / .claw: 640x480 through the lens model, clean (no labels drawn: the policy sees
  what the real cameras see)
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

# ⚠️ LOCK before the first real batch: changing it later makes earlier episodes a different task for the policy.
TASK = "Cut the leaves below the lowest ripening truss off flush at the stem."
REPO_ID = "krish5831/shear_sim_scripted_v0"
PILOT_EPISODES = 5
SEED_BASE = 900_000          # row attempt k uses seeds SEED_BASE + 3k .. +2 (apart from the trials' 700000s)
LOG_DIR = pathlib.Path.home() / ".cache" / "shear_sim" / "episodes"
CAMERAS = ("birdseye", "claw")


class Recorder:
    """Feeds one frame per Episode callback: the observation from the frame before, and the action that was sent
    during this one (lerobot-record's order: observe, then act)."""

    def __init__(self, ds, scn, build, rop, tap):
        from cameras import SimCameras
        self.ds, self.scn, self.build, self.rop, self.tap = ds, scn, build, rop, tap
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
        from lerobot.utils.constants import ACTION, OBS_STR
        d, scn = self.scn.d, self.scn
        sent = {f"{j}.pos": float(v) for j, v in
                zip(arm_map.JOINTS, self.fm.rad_to_norm(np.r_[d.ctrl[scn.act], d.ctrl[scn.jaw_act]]))}
        self.ds.add_frame({**self.build(self.ds.features, self.rop(self.prev), prefix=OBS_STR),
                           **self.build(self.ds.features, self.tap((sent, self.prev)), prefix=ACTION),
                           "task": TASK})
        self.n += 1
        self.prev = self.observe()

    def close(self):
        self.cams.close()


def make_dataset(repo_id, resume):
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
    a = ap.parse_args()
    if a.pilot:
        a.repo_id, n_eps = a.repo_id + "_pilot", a.episodes or PILOT_EPISODES
    elif a.episodes:
        n_eps = a.episodes
    else:
        sys.exit("--pilot, or --episodes N for a real batch (after K7 approved the pilot)")
    from lerobot.utils.feature_utils import build_dataset_frame
    ds, tap, rop, root = make_dataset(a.repo_id, a.resume)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{a.repo_id.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"dataset {a.repo_id} at {root}\ntask {TASK!r}\nattempt log {log_path}", flush=True)
    done_rows = set()
    if a.resume:                                  # rows an earlier run already tried
        for p in LOG_DIR.glob(f"{a.repo_id.split('/')[-1]}_2*.jsonl"):
            done_rows |= {json.loads(ln).get("row") for ln in p.read_text().splitlines() if ln.strip()}
    kept, attempts, t_start = ds.meta.total_episodes if a.resume else 0, 0, time.time()
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
                if not any(found):
                    rec.update(kept=False, why="no pot angle gives a clean cut")
                    log.write(json.dumps(rec) + "\n"); log.flush()
                    print(f"row {seeds[0]}: skipped, no pot angle gives a clean cut", flush=True)
                    continue
                scn = ct.Scene(seeds, turns=turns, domain_rand=not a.no_dr)
                rec["domain_rand"] = scn.dr
                r = Recorder(ds, scn, build_dataset_frame, rop, tap)
                res = ct.run_pass(scn, rng, r.frame)
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
                log.write(json.dumps(rec, default=str) + "\n"); log.flush()
        finally:
            ds.finalize()
            summary = dict(summary=True, kept=kept, attempts=attempts, minutes=round((time.time() - t_start) / 60, 1))
            log.write(json.dumps(summary) + "\n")
    print(f"\nkept {kept} episodes from {attempts} rows in {summary['minutes']} min -> {root}")
    print("Not pushed to the Hub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
