#!/usr/bin/env bash
# ACT on teacher-v3 data (BLUSH + SHEAR post-mortem §6.3, vault, 2026-11-11), copied from B.L.U.S.H
# sim/scripts/train_sim.sh act-teacher (commit fc2af15): ACT defaults (chunk 100, batch 8, lr 1e-5; the cue = cut
# point + petiole direction as observation.environment_state), lerobot's image augmentation on (colour, sharpness,
# RandomAffine +-5 deg / 5 % shift). How chunks are played is set at evaluation (sim/eval_onecut.py --ensemble /
# --n-action-steps), not here. Never pushes to the Hub (K7 approves every push).
#
#   policy/train_act_teacher.sh krish5831/shear_sim_teacher_onerow_v1                       step 1: STEPS=20000 SAVE=5000
#   DRYRUN=1 policy/train_act_teacher.sh <ds>                                                print the command only
#
# Run from the repo root, in the lerobot env, under systemd-run MemoryMax (CLAUDE.md). Writes outputs/train/$RUN.
set -euo pipefail
DS=${1:?dataset, e.g. krish5831/shear_sim_teacher_onerow_v1}
RUN=${RUN:-act_teacher_v1}
CMD=(lerobot-train --dataset.repo_id="$DS" --policy.type=act --policy.device=cuda
     --policy.push_to_hub=false --dataset.image_transforms.enable=true
     --batch_size="${BATCH:-8}" --steps="${STEPS:-100000}" --save_freq="${SAVE:-10000}" --log_freq=100 --num_workers=8
     --output_dir="outputs/train/$RUN" --job_name="$RUN" --wandb.enable=false)
if [[ -n "${DRYRUN:-}" ]]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
