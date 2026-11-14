#!/usr/bin/env bash
# ACT on teacher-v3 data (BLUSH + SHEAR post-mortem §6.3, vault, 2026-11-11), copied from B.L.U.S.H
# sim/scripts/train_sim.sh act-teacher (commit fc2af15): ACT defaults (chunk 100, batch 8, lr 1e-5; the cue = cut
# point + petiole direction as observation.environment_state), lerobot's image augmentation on (colour, sharpness,
# RandomAffine +-5 deg / 5 % shift). How chunks are played is set at evaluation (sim/eval_onecut.py --ensemble /
# --n-action-steps), not here. Never pushes to the Hub (K7 approves every push).
#
#   policy/train_act_teacher.sh krish5831/shear_sim_teacher_onerow_v1                       step 1: STEPS=20000 SAVE=5000
#   RUN=act_teacher_v3 policy/train_act_teacher.sh krish5831/shear_sim_teacher_v3            step 3: 100k, save every 5k
#   LR_SCHED=constant ...                                                                    step 1's constant 1e-5
#   EMA=1 ...                                                                                also an EMA copy per checkpoint
#   DRYRUN=1 policy/train_act_teacher.sh <ds>                                                print the command only
#
# Learning rate (step 3, 2026-11-14): step 1 drifted past its peak at a constant 1e-5 (45k 18 of 19, 50k 8 of 19 with
# the stubs long; B.L.U.S.H's collapsed 95 % -> 5 % between 40k and 50k, post-mortem §6.9 / §6.10), so the default is
# now a cosine decay to zero at --steps after WARMUP steps (lerobot's cosine_annealing_with_warmup). ACT's preset is
# AdamW lr 1e-5, weight decay 1e-4 and no scheduler (configuration_act.py get_optimizer_preset), so the same optimiser
# is spelled out here. Without the preset lerobot passes policy.parameters() as one group (optim/factory.py) instead
# of ACT's two (backbone apart); the same thing here, since optimizer_lr_backbone is 1e-5 too. Checked 2026-11-14:
# lr 0 at step 0, 1e-5 at 1000, 5.1e-6 at 50k, 1.5e-6 at 75k, 0 at 100k. Warmup 1000 steps: pi0's value (guess,
# unverified). EMA=1 adds lerobot's weight average (pretrained_model_ema per checkpoint).
#
# Run from the repo root, in the lerobot env, under systemd-run MemoryMax (CLAUDE.md). Writes outputs/train/$RUN.
set -euo pipefail
DS=${1:?dataset, e.g. krish5831/shear_sim_teacher_onerow_v1}
RUN=${RUN:-act_teacher_v1}
CMD=(lerobot-train --dataset.repo_id="$DS" --policy.type=act --policy.device=cuda
     --policy.push_to_hub=false --dataset.image_transforms.enable=true
     --batch_size="${BATCH:-8}" --steps="${STEPS:-100000}" --save_freq="${SAVE:-5000}" --log_freq=100 --num_workers=8
     --output_dir="outputs/train/$RUN" --job_name="$RUN" --wandb.enable=false)
if [[ "${LR_SCHED:-cosine}" == cosine ]]; then
  CMD+=(--use_policy_training_preset=false --optimizer.type=adamw --optimizer.lr=1e-5 --optimizer.weight_decay=1e-4
        --scheduler.type=cosine_annealing_with_warmup --scheduler.num_warmup_steps="${WARMUP:-1000}")
fi
[[ -n "${EMA:-}" ]] && CMD+=(--ema.enable=true)
if [[ -n "${DRYRUN:-}" ]]; then echo "${CMD[@]}"; else "${CMD[@]}"; fi
