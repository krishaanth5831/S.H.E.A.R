#!/usr/bin/env python3
"""Roll a trained policy out in the SHEAR sim: does it deleaf new plants the way the scripted cutter does?

  python rollout_policy.py --ckpt ../outputs/train/shear_smolvla_v3_hub/checkpoints/050000/pretrained_model --rows 10
  python rollout_policy.py ... --no-video
  python rollout_policy.py --ckpt ... --rtc 10 --rtc-horizon 20 --smooth 0.5   real-time chunking + command smoothing
  python rollout_policy.py --ckpt ... --n-action-steps 10                     look again every 10 moves, not 50
  python rollout_policy.py --ckpt ... --windowed        watch the policy drive live, row after row (real time)
  python rollout_policy.py --ckpt ... --windowed --rows 1   one row live, then the window closes
  python rollout_policy.py --ckpt ... --start 8 --rows 2 --out <same>   finish an interrupted 10-row test (rows 8-9)
  python rollout_policy.py --ckpt ... --train-rows ~/.cache/shear_sim/episodes/shear_sim_scripted_v1_<time>.jsonl
                                  the same test on rows the policy TRAINED on: fails there too = a pipeline bug or
                                  underfitting; works there but not on new rows = it needs more / more varied data

For each row (new seeds, never in the dataset: SEED_BASE; or, with --train-rows, dataset rows), set up like the
training episodes (generate_cuts.best_turns pots, domain randomisation), the policy drives the arm from the start pose
at 30 fps: every frame it gets the joint state (use_degrees=false units), the frames of the dataset's cameras
(generate_episodes.CAMERAS: base + claw since the v1 rig) and the task string, exactly as the dataset recorded them,
and its action goes to the servos. The episode ends at MAX_S, or once the arm has been back at the start pose
for STILL_S after going out.

Judged per row, the same way as the scripted passes (cutter.py): every leaf cut is checked against the rule (a cut the
rule didn't ask for is a wrong cut), its stub measured, and the peak force of the arm on stems, truss stalks, pots and
the frame kept. The scripted cutter then runs on the same row as the baseline.

Writes: rollouts.json + rollouts.txt, and rollout_KK.mp4 (side | base | claw, the sim's labels drawn on), into
--out (default ~/.cache/shear_sim/rollouts/<checkpoint step>_<n50 | rtc10h20>[_train]). Run outside Claude Code's
sandbox (the GPU), in the lerobot env, under systemd-run MemoryMax.
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
import stemgen as sg  # noqa: E402

SEED_BASE = 980_000           # never used for the dataset (900000s) or the trials (700000s)
MAX_S = 60.0                  # the longest training episode was 49 s
STILL_S = 1.5                 # back at the start pose this long (after leaving it) ends the episode
START_TOL_NORM = 4.0          # "at the start pose": every arm joint within this, norm units
SMOOTH = 0.0                  # set from --smooth
LEFT_START_NORM = 15.0        # "has gone out": some arm joint this far from the start pose


def train_rows(log_path, n):
    """n rows the dataset kept (generate_episodes' attempt log), evenly spread over the episodes, hold episodes (the
    rule cuts nothing) left out. The scene rebuilds the same from the seeds: pots (best_turns, rng from seeds[0]),
    domain randomisation (seeds[0]). Only the scripted pass's start jitter differs: the rollout starts exactly at
    the start pose."""
    recs = [json.loads(ln) for ln in pathlib.Path(log_path).read_text().splitlines() if ln.strip()]
    kept = [r for r in recs if r.get("kept") is True and not r.get("hold")]
    pick = [kept[round(i * (len(kept) - 1) / max(n - 1, 1))] for i in range(min(n, len(kept)))]
    return [(tuple(r["seeds"]), r["episode_index"], r["pot_turns_deg"]) for r in pick]


def load_policy(ckpt, device="cuda", n_action_steps=None, rtc_horizon=None):
    """n_action_steps: how many of each predicted chunk's moves are played before the policy looks again (SmolVLA
    plans 50 and, by default, plays all 50 blind: 1.7 s). Fewer = more often closed-loop, at more inference."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.rollout.context import _load_pretrained_policy
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
    if n_action_steps:
        cfg.n_action_steps = n_action_steps
    if rtc_horizon:
        # Real-time chunking, "guided" (works on any checkpoint): each new chunk is steered toward the old chunk's
        # leftover moves, so chunk handovers don't jump. Dream Machines' π0.5 write-up (2026): 53 % synchronous ->
        # 93 % with it, the biggest serving-side gain.
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        cfg.rtc_config = RTCConfig(enabled=True, execution_horizon=rtc_horizon)
    policy = _load_pretrained_policy(cfg)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt)
    return policy, pre, post


class Sim:
    """The row and its cameras, as the dataset saw them, plus the judge's sensors."""

    def __init__(self, seeds, turns):
        from cameras import SimCameras
        self.scn = ct.Scene(seeds, turns=turns, domain_rand=True)
        self.cams = SimCameras(self.scn.m)
        self.fm = arm_map.FollowerMap()
        self.ep = ct.Episode(self.scn, np.random.default_rng(seeds[0]))    # its Cutters + Monitor
        self.start_norm = self.fm.rad_to_norm(np.r_[self.scn.start, ct.JAW_CLOSED])

    def state_norm(self):
        d, s = self.scn.d, self.scn
        return self.fm.rad_to_norm(np.r_[d.qpos[s.arm_adr], d.qpos[s.jaw_adr]])

    def observation(self):
        import torch
        obs = {"observation.state": torch.tensor(self.state_norm(), dtype=torch.float32)}
        for c in ge.CAMERAS:
            img = self.cams.rgb(self.scn.d, c)
            obs[f"observation.images.{c}"] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        obs["task"] = ge.TASK
        return obs

    def act(self, action_norm):
        q = self.fm.norm_to_rad(np.asarray(action_norm, float))
        d, s = self.scn.d, self.scn
        d.ctrl[s.act] = q[:5]
        d.ctrl[s.jaw_act] = q[5]
        for k in range(self.ep.steps_per_frame):
            mujoco.mj_step(self.scn.m, d)
            for c in self.ep.cutters.values():
                # ⚠️ also after the cut while a nearer blade pair's second cut is pending (plant.Cutter._cut), as
                # shear_teacher.run steps them: stepping only attached cutters missed it and read the far pair's
                # 13-17 mm stub (replay_teacher, 2026-11-13: 2 of 5 clean teacher episodes)
                if c.attached or c.pending:
                    c.step()
            if k % 5 == 0:
                self.ep.mon.step()
        self.ep.t += 1.0 / ct.FPS


def judge(sim):
    scn, ep = sim.scn, sim.ep
    rule = {(k, {n.id: i for i, n in enumerate(sp.nodes)}[sel.nodes[j].id])
            for k, sp in enumerate(scn.info["specs"]) for sel in [sg.label(sp)] for j in sel.cuts}
    cut = {k: c for k, c in ep.cutters.items() if c.cut_t is not None}
    right = [k for k in cut if k in rule]
    wrong = [k for k in cut if k not in rule]
    stubs = [round(cut[k].stub * 1000, 2) for k in right]
    clean = [k for k in right if ct.STUB_OK[0] <= cut[k].stub <= ct.STUB_OK[1]]
    pressed = {k: round(v, 1) for k, v in ep.mon.peak.items() if k in ct.FORCE_LIMIT and v > ct.FORCE_LIMIT[k]}
    return dict(rule_cuts=len(rule), cut_right=len(right), cut_clean=len(clean), cut_wrong=len(wrong),
                wrong=[scn.info["specs"][k].nodes[i].id for k, i in wrong], stubs_mm=stubs, pressed=pressed,
                blade_on_stalk=round(ep.mon.blade_on_stalk, 1),
                peaks_n={k: round(v, 2) for k, v in ep.mon.peak.items()})


class RTCDriver:
    """Plays a SmolVLA policy with real-time chunking: a new 50-move chunk every `reinfer` moves, inpainted over the
    previous chunk's moves that weren't played yet (lerobot rollout/inference/rtc.py, with no latency: the sim waits
    for the policy, so inference_delay is 0)."""

    def __init__(self, policy, pre, post, reinfer):
        self.policy, self.pre, self.post, self.reinfer = policy, pre, post, reinfer
        self.H = policy.config.rtc_config.execution_horizon
        self.chunk = self.actions = None
        self.k = 0

    def reset(self):
        self.policy.reset()
        self.chunk = self.actions = None
        self.k = 0

    def __call__(self, obs):
        import torch
        if self.chunk is None or self.k >= self.reinfer:
            prev = None
            if self.chunk is not None:
                left = self.chunk[self.k:]                     # normalised, not yet played
                prev = torch.zeros((self.H, left.shape[-1]), device=left.device, dtype=left.dtype)
                prev[:min(self.H, len(left))] = left[:self.H]
            # ⚠️ not under inference_mode: guided RTC takes a gradient inside the denoising loop (lerobot turns grad
            # back on there, which inference_mode forbids: "element 0 of tensors does not require grad").
            new = self.policy.predict_action_chunk(self.pre(obs), inference_delay=0, prev_chunk_left_over=prev)
            with torch.no_grad():
                self.chunk = new.detach().squeeze(0)
                self.actions = self.post(new.detach()).squeeze(0)
            self.k = 0
        a = self.actions[self.k]
        self.k += 1
        return a


def rollout(policy, pre, post, seeds, turns, video_path=None, viewer=None, driver=None):
    """viewer: a function(sim) that opens a live window and returns a per-frame callback (--windowed).
    driver: an RTCDriver; None plays the policy's own select_action (synchronous chunks)."""
    import torch
    sim = Sim(seeds, turns)
    video = gc.Video(sim.scn, video_path) if video_path else None
    live = viewer(sim) if viewer else None
    (driver or policy).reset()
    prev = None                       # for --smooth
    gone_out, still, frames = False, 0, 0
    while sim.ep.t < MAX_S:
        if driver:
            a = driver(sim.observation())
        else:
            with torch.inference_mode():
                a = post(policy.select_action(pre(sim.observation())))
        if SMOOTH > 0:
            # SockBot (walnut sensei, part 5): a first-order filter on the commands, alpha 0.5 cut the jitter to 38 %
            # with little lag (0.3: 21 % but laggy). Here alpha is the weight of the NEW command.
            a = a if prev is None else SMOOTH * a + (1.0 - SMOOTH) * prev
            prev = a
        sim.act(a.reshape(-1).cpu().numpy())
        frames += 1
        sim.ep.done |= {key for key, c in sim.ep.cutters.items() if c.cut_t is not None}   # off the overlay
        if video:
            video.frame("policy", sim.ep)
        if live and not live():
            break
        off = np.abs(sim.state_norm()[:5] - sim.start_norm[:5]).max()
        gone_out |= off > LEFT_START_NORM
        still = still + 1 if gone_out and off < START_TOL_NORM else 0
        if still >= STILL_S * ct.FPS:
            break
    res = judge(sim)
    res.update(seconds=round(sim.ep.t, 1), back_at_start=bool(still >= STILL_S * ct.FPS), frames=frames)
    if video:
        video.close()
    if live:
        live(close=True)
    sim.cams.close()
    return res, sim


def windowed(policy, pre, post, driver=None, rows=None):
    """Row after row in the MuJoCo viewer, at real time, `rows` of them (None = until the window is closed;
    Ctrl+C on Wayland). Each row's window closes when its row ends."""
    import mujoco.viewer
    state = {}

    def open_viewer(sim):
        v = mujoco.viewer.launch_passive(sim.scn.m, sim.scn.d)
        v.cam.lookat[:] = (ct.lo.PLANT_X[1], (ct.lo.ROW_Y + ct.lo.AISLE_Y) / 2, 0.22)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.85, 245, -22
        last = [time.time()]

        def frame(close=False):
            if close:
                v.close()
                return False
            if not v.is_running():
                state["stop"] = True
                return False
            v.sync()
            wait = 1.0 / ct.FPS - (time.time() - last[0])
            if wait > 0:
                time.sleep(wait)
            last[0] = time.time()
            return True
        return frame

    k, shown = 0, 0
    while not state.get("stop") and (rows is None or shown < rows):
        seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
        k += 1
        rng = np.random.default_rng(seeds[0])
        turns, found = gc.best_turns(seeds, rng)
        if not any(found):
            continue
        how = (f"RTC, a new blended chunk every {driver.reinfer} moves" if driver else
               f"looking again every {policy.config.n_action_steps} moves")
        print(f"row {seeds[0]}: the policy is driving, {how} "
              f"(new window per row; close it to stop)", flush=True)
        res, _ = rollout(policy, pre, post, seeds, turns, None, open_viewer, driver)
        shown += 1
        print(f"  cut {res['cut_right']} right ({res['cut_clean']} clean) + {res['cut_wrong']} wrong of "
              f"{res['rule_cuts']} rule cuts; pressed {res['pressed'] or 'nothing'}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a checkpoint's pretrained_model folder")
    ap.add_argument("--rows", type=int, default=None,
                    help="rows to run (default 10; with --windowed, row after row until the window is closed)")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="skip the first N test rows: finishes an interrupted test, keeping its videos")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--train-rows", type=pathlib.Path, default=None, metavar="ATTEMPT_LOG",
                    help="test on rows the dataset trained on (generate_episodes' attempt log .jsonl), not new ones")
    ap.add_argument("--windowed", action="store_true", help="watch live in a MuJoCo window (no videos, no baseline)")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="moves played per chunk before looking again (default: the checkpoint's, 50)")
    ap.add_argument("--rtc", type=int, default=None, metavar="REINFER",
                    help="real-time chunking: a new chunk every REINFER moves, blended over the old one's rest")
    ap.add_argument("--rtc-horizon", type=int, default=20, help="old-chunk moves the new one is guided toward")
    ap.add_argument("--smooth", type=float, default=0.0, metavar="ALPHA",
                    help="first-order smoothing of the commands, ALPHA = weight of the new one (SockBot used 0.5); "
                         "0 = off")
    a = ap.parse_args()
    global SMOOTH
    SMOOTH = a.smooth
    ckpt = str(pathlib.Path(a.ckpt).resolve())
    tag = f"rtc{a.rtc}h{a.rtc_horizon}" if a.rtc else f"n{a.n_action_steps or 50}"
    tag += "_train" if a.train_rows else ""
    out = a.out or pathlib.Path.home() / ".cache" / "shear_sim" / "rollouts" / f"{pathlib.Path(ckpt).parent.name}_{tag}"
    policy, pre, post = load_policy(ckpt, n_action_steps=a.n_action_steps, rtc_horizon=a.rtc_horizon if a.rtc else None)
    driver = RTCDriver(policy, pre, post, a.rtc) if a.rtc else None
    print(f"n_action_steps {policy.config.n_action_steps}" + (f", RTC every {a.rtc} moves, horizon {a.rtc_horizon}"
                                                             if a.rtc else ""), flush=True)
    if a.windowed:                     # ⚠️ before touching --out: a live view must not delete a running batch's videos
        return windowed(policy, pre, post, driver, a.rows)
    a.rows = a.rows or 10
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_from{a.start:02d}" if a.start else ""   # a resumed test keeps the first part's videos and summary
    old = [f for f in out.glob("rollout_*.mp4") if int(f.stem.split("_")[1]) >= a.start]   # only our own outputs
    for f in old + list(out.glob(f"rollouts{suffix}.*")):
        f.unlink()
    print(f"policy {ckpt}\nout {out}", flush=True)
    rows, skipped = [], 0
    k = 0
    fixed = train_rows(a.train_rows, a.start + a.rows) if a.train_rows else None
    while len(rows) < a.rows and k < (len(fixed) if fixed else 30 * (a.rows + a.start)):
        if fixed:
            seeds, episode, logged_deg = fixed[k]
        else:
            seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
        k += 1
        rng = np.random.default_rng(seeds[0])
        turns, found = gc.best_turns(seeds, rng)
        if fixed:
            # ⚠️ the check that this IS the training row: the pots must turn as they did when it was recorded
            deg = [round(float(np.degrees(t)) % 360, 1) for t in turns]
            print(f"{'PASS' if deg == logged_deg else 'FAIL'}: row {seeds[0]} is episode {episode}, pots {deg} "
                  f"(recorded {logged_deg})", flush=True)
        elif not any(found):
            continue                                   # the scripted cutter can't do anything here either
        if skipped < a.start:
            skipped += 1
            continue
        t0 = time.time()
        n = a.start + len(rows)
        res, sim = rollout(policy, pre, post, seeds, turns, None if a.no_video else out / f"rollout_{n:02d}.mp4",
                           driver=driver)
        base = ct.run_pass(ct.Scene(seeds, turns=turns, domain_rand=True), np.random.default_rng(seeds[0]))
        res.update(row=seeds[0], scripted_cut=base["n_cut"], scripted_verdict=base["verdict"],
                   wall_s=round(time.time() - t0, 1))
        rows.append(res)
        print(f"rollout {n} (row {seeds[0]}): policy cut {res['cut_right']} right ({res['cut_clean']} clean) + "
              f"{res['cut_wrong']} wrong of {res['rule_cuts']} rule cuts; scripted {res['scripted_cut']}; "
              f"pressed {res['pressed'] or 'nothing'}; back at start {res['back_at_start']} [{res['wall_s']} s]",
              flush=True)
    tot = {k_: sum(r[k_] for r in rows) for k_ in ("rule_cuts", "cut_right", "cut_clean", "cut_wrong", "scripted_cut")}
    damaged = sum(bool(r["pressed"]) or r["blade_on_stalk"] > 0 for r in rows)
    lines = [f"{len(rows)} rollouts of {ckpt}, {tag}",
             f"policy: {tot['cut_right']} right cuts ({tot['cut_clean']} with a flush stub), {tot['cut_wrong']} wrong "
             f"cuts, {damaged} rollouts pressed something it mustn't; scripted cutter on the same rows: "
             f"{tot['scripted_cut']} cuts; the rule asked for {tot['rule_cuts']}",
             f"back at the start pose: {sum(r['back_at_start'] for r in rows)} of {len(rows)}", ""]
    for i, r in enumerate(rows, start=a.start):
        lines.append(f"rollout_{i:02d}.mp4  row {r['row']}: right {r['cut_right']} (stubs {r['stubs_mm']}), wrong "
                     f"{r['cut_wrong']} {r['wrong']}, scripted {r['scripted_cut']}, pressed {r['pressed'] or '-'}, "
                     f"{r['seconds']} s")
    (out / f"rollouts{suffix}.json").write_text(json.dumps(dict(ckpt=ckpt, rows=rows, totals=tot), indent=1,
                                                           default=str))
    (out / f"rollouts{suffix}.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
