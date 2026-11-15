#!/usr/bin/env python3
"""Pick a training run's checkpoint by rollouts, not by "last" or the loss. B.L.U.S.H sim/pick_checkpoint.py
(commit a16e522, --watch added in both the same day, 2026-11-14) over eval_onecut.py.

    python pick_checkpoint.py ../outputs/train/act_teacher_onerow_v1 -- --row 910057 --rollouts 19 --noise 1.0 \
        --n-action-steps 100
    python pick_checkpoint.py ../outputs/train/act_teacher_v3 --watch --until 100000 --cpu -- --rows 20 \
        --n-action-steps 100                                                  # step 3: test each checkpoint as it lands

Why: step 1 peaked at 45k (18 of 19) and drifted long by 50k (8 of 19, stubs median 3.8 mm) with the loss flat;
B.L.U.S.H's collapsed from 95 % to 5 % between 40k and 50k. The loss doesn't show it, rollouts do.

Runs eval_onecut.py (everything after "--" is passed through) on every numbered checkpoint of the run, oldest first,
skipping ones already ranked with the same arguments, then ranks them by successes, then cuts, then the median
closest approach. Writes ../outputs/eval/<run>_ranking.json and prints the table and the best.

--watch (K7 2026-11-14: a test every 5k steps while training): keeps going while lerobot-train writes checkpoints,
evaluating each one once it is finished (lerobot points `checkpoints/last` at a checkpoint only after it is fully
written, lerobot_train.py), until the checkpoint --until is ranked, or the training process --pid is gone and
nothing new has landed. --cpu runs the rollouts with CUDA_VISIBLE_DEVICES="" so training keeps the GPU to itself.
Run as eval_onecut.py is run (lerobot env, outside the Claude sandbox, under systemd-run MemoryMax).
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
EVAL = HERE / "eval_onecut.py"
POLL_S = 60                   # --watch: how often to look for a new checkpoint (a checkpoint is ~40 min apart)


def summary(log_path):
    rows = [json.loads(line) for line in pathlib.Path(log_path).read_text().splitlines() if line.strip()]
    return next((r for r in rows if r.get("summary")), None)


def rank_key(r):
    return (-r["success"], -r["cut"], r["closest_mm"] if r["closest_mm"] is not None else 1e9)


def row_of(s, ck, log):
    return dict(checkpoint=ck.name, success=s["success"], n=s["n"], cut=s["cut"], closest_mm=s["closest_median_mm"],
                stub_mm=s.get("stub_median_mm"), holds_ok=s.get("holds_ok"), holds=s.get("holds"),
                handed_off=s.get("handed_off"), log=log)


def describe(r):
    holds = f", holds {r['holds_ok']}/{r['holds']}" if r.get("holds") else ""
    hand = f", hand-off {r['handed_off']}" if r.get("handed_off") is not None else ""
    return (f"{r['success']}/{r['n']}, cut {r['cut']}, closest {r['closest_mm']} mm, stub median "
            f"{r['stub_mm']} mm{holds}{hand}")


def finished(run, watch):
    """The run's numbered checkpoints, oldest first; with watch only the ones `last` says are complete."""
    d = run / "checkpoints"
    cks = sorted(p for p in d.iterdir() if p.name.isdigit()) if d.is_dir() else []
    if not watch:
        return cks
    last = run / "checkpoints" / "last"
    done = int(pathlib.Path(os.readlink(last)).name) if last.is_symlink() else -1
    return [p for p in cks if int(p.name) <= done]


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    argv = sys.argv[1:]
    passthru = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=pathlib.Path, help="a training run's output dir (holds checkpoints/)")
    ap.add_argument("--only", nargs="*", default=None, help="these checkpoint names only, e.g. 040000 050000")
    ap.add_argument("--watch", action="store_true", help="keep evaluating new checkpoints as training writes them")
    ap.add_argument("--until", type=int, default=None, help="--watch: stop once this step's checkpoint is ranked")
    ap.add_argument("--pid", type=int, default=None, help="--watch: stop when this (training) process is gone")
    ap.add_argument("--cpu", action="store_true", help="rollouts on the CPU (CUDA_VISIBLE_DEVICES empty)")
    a = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    out = HERE.parent / "outputs" / "eval" / f"{a.run.name}_ranking.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked = json.loads(out.read_text()) if out.exists() else {}
    key = " ".join(passthru)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="") if a.cpu else None
    failed = set()                # --watch: a checkpoint whose eval crashed isn't tried again every POLL_S
    while True:
        cks = finished(a.run, a.watch)
        if a.only:
            cks = [p for p in cks if p.name in a.only]
        for ck in cks:
            tag = f"{ck.name} | {key}"
            if tag in ranked or tag in failed:
                continue
            print(f"▶ {ck.name}  ({time.strftime('%H:%M')})", flush=True)
            p = subprocess.run([sys.executable, str(EVAL), str(ck / "pretrained_model"), *passthru],
                               capture_output=True, text=True, cwd=HERE, env=env)
            m = re.search(r"log (\S+\.jsonl)", p.stdout)
            if p.returncode or not m:
                print(f"  failed (exit {p.returncode}):\n{p.stdout[-800:]}{p.stderr[-800:]}", flush=True)
                failed.add(tag)
                continue
            s = summary(m.group(1))
            ranked[tag] = row_of(s, ck, m.group(1))
            # merged with the file as it is now: a second watcher (other eval arguments) writes the same ranking
            ranked = {**(json.loads(out.read_text()) if out.exists() else {}), **ranked}
            out.write_text(json.dumps(ranked, indent=1))
            print(f"  {describe(ranked[tag])}", flush=True)
        if not a.watch:
            break
        if a.until is not None and any(int(r["checkpoint"]) >= a.until for t, r in ranked.items()
                                       if t.endswith(f"| {key}")):
            break
        if a.until is not None and any(int(t.split(" | ")[0]) >= a.until for t in failed):
            print(f"the eval of checkpoint {a.until} failed: stopping", flush=True)
            break
        if a.pid is not None and not alive(a.pid) and finished(a.run, True) == cks:
            print(f"training process {a.pid} is gone and nothing new landed: stopping", flush=True)
            break
        time.sleep(POLL_S)
    rows = sorted((r for t, r in ranked.items() if t.endswith(f"| {key}")), key=rank_key)
    print(f"\nranking ({key}):")
    for r in rows:
        print(f"  {r['checkpoint']}: {describe(r)}")
    if rows:
        print(f"best: {a.run / 'checkpoints' / rows[0]['checkpoint'] / 'pretrained_model'}  ({out})")


if __name__ == "__main__":
    main()
