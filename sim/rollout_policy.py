#!/usr/bin/env python3
"""Roll a trained policy out in the SHEAR sim: does it deleaf new plants the way the scripted cutter does?

  python rollout_policy.py --ckpt ../outputs/train/shear_smolvla_lora_v0/checkpoints/last/pretrained_model --rows 10
  python rollout_policy.py ... --no-video

For each row (new seeds, never in the dataset: SEED_BASE), set up like the training episodes (generate_cuts.best_turns
pots, domain randomisation), the policy drives the arm from the start pose at 30 fps: every frame it gets the joint
state (use_degrees=false units), the birdseye and claw frames and the task string, exactly as the dataset recorded
them, and its action goes to the servos. The episode ends at MAX_S, or once the arm has been back at the start pose
for STILL_S after going out.

Judged per row, the same way as the scripted passes (cutter.py): every leaf cut is checked against the rule (a cut the
rule didn't ask for is a wrong cut), its stub measured, and the peak force of the arm on stems, truss stalks, pots and
the frame kept. The scripted cutter then runs on the same row as the baseline.

Writes: rollouts.json + rollouts.txt, and rollout_KK.mp4 (side | birdseye | claw, the sim's labels drawn on), into
--out (default ~/.cache/shear_sim/rollouts/<checkpoint name>). Run outside Claude Code's sandbox (the GPU), in the
lerobot env, under systemd-run MemoryMax.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")      # ⚠️ cameras must render through EGL (B.L.U.S.H trap)

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
LEFT_START_NORM = 15.0        # "has gone out": some arm joint this far from the start pose


def load_policy(ckpt, device="cuda"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.rollout.context import _load_pretrained_policy
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = device
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
                if c.attached:
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
                blade_on_stalk=round(ep.mon.blade_on_stalk, 1), peaks_n={k: round(v, 2) for k, v in ep.mon.peak.items()})


def rollout(policy, pre, post, seeds, turns, video_path=None):
    import torch
    sim = Sim(seeds, turns)
    video = gc.Video(sim.scn, video_path) if video_path else None
    policy.reset()
    gone_out, still, frames = False, 0, 0
    while sim.ep.t < MAX_S:
        with torch.inference_mode():
            a = post(policy.select_action(pre(sim.observation())))
        sim.act(a.squeeze(0).cpu().numpy())
        frames += 1
        sim.ep.done |= {key for key, c in sim.ep.cutters.items() if c.cut_t is not None}   # off the overlay
        if video:
            video.frame("policy", sim.ep)
        off = np.abs(sim.state_norm()[:5] - sim.start_norm[:5]).max()
        gone_out |= off > LEFT_START_NORM
        still = still + 1 if gone_out and off < START_TOL_NORM else 0
        if still >= STILL_S * ct.FPS:
            break
    res = judge(sim)
    res.update(seconds=round(sim.ep.t, 1), back_at_start=bool(still >= STILL_S * ct.FPS), frames=frames)
    if video:
        video.close()
    sim.cams.close()
    return res, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a checkpoint's pretrained_model folder")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    ckpt = str(pathlib.Path(a.ckpt).resolve())
    out = a.out or pathlib.Path.home() / ".cache" / "shear_sim" / "rollouts" / pathlib.Path(ckpt).parent.name
    out.mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("rollout_*.mp4")) + list(out.glob("rollouts.*")):     # only our own outputs
        f.unlink()
    policy, pre, post = load_policy(ckpt)
    print(f"policy {ckpt}\nout {out}", flush=True)
    rows = []
    k = 0
    while len(rows) < a.rows and k < 30 * a.rows:
        seeds = tuple(SEED_BASE + 3 * k + j for j in range(3))
        k += 1
        rng = np.random.default_rng(seeds[0])
        turns, found = gc.best_turns(seeds, rng)
        if not any(found):
            continue                                   # the scripted cutter can't do anything here either
        t0 = time.time()
        n = len(rows)
        res, sim = rollout(policy, pre, post, seeds, turns, None if a.no_video else out / f"rollout_{n:02d}.mp4")
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
    lines = [f"{len(rows)} rollouts of {ckpt}",
             f"policy: {tot['cut_right']} right cuts ({tot['cut_clean']} with a flush stub), {tot['cut_wrong']} wrong "
             f"cuts, {damaged} rollouts pressed something it mustn't; scripted cutter on the same rows: "
             f"{tot['scripted_cut']} cuts; the rule asked for {tot['rule_cuts']}",
             f"back at the start pose: {sum(r['back_at_start'] for r in rows)} of {len(rows)}", ""]
    for i, r in enumerate(rows):
        lines.append(f"rollout_{i:02d}.mp4  row {r['row']}: right {r['cut_right']} (stubs {r['stubs_mm']}), wrong "
                     f"{r['cut_wrong']} {r['wrong']}, scripted {r['scripted_cut']}, pressed {r['pressed'] or '-'}, "
                     f"{r['seconds']} s")
    (out / "rollouts.json").write_text(json.dumps(dict(ckpt=ckpt, rows=rows, totals=tot), indent=1, default=str))
    (out / "rollouts.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
