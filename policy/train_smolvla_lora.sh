#!/usr/bin/env bash
# train_smolvla_lora.sh: SmolVLA fine-tuned with LoRA on SHEAR's scripted sim passes, on the laptop GPU
# (K7 2026-11-04). Adapted from B.L.U.S.H harvest/scripts/train_smolvla.sh.
#   policy/train_smolvla_lora.sh smoke    a few steps: memory and speed on the 6 GB RTX 1000 Ada
#   policy/train_smolvla_lora.sh full     the LoRA run (v0)
#   policy/train_smolvla_lora.sh expert   no LoRA: the whole action expert (v1); expert-smoke first
# Run outside Claude Code's sandbox (the GPU isn't visible inside it), in the lerobot env.
# v3 on Google Colab (T4, 16 GB) instead of the laptop: policy/colab_train_v3.ipynb (same recipe, checkpoints on the Hub,
# resumes after a dead session). Open: https://colab.research.google.com/github/krishaanth5831/S.H.E.A.R/blob/dev/policy/colab_train_v3.ipynb
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=krish5831/shear_sim_scripted_v0        # 200 passes, local + private on the Hub
OUT=outputs/train
# ⚠️ SmolVLA's base expects its own camera keys (camera1..3, lerobot/smolvla_base config.json). The birdseye is
# the overview, the claw the wrist view. The SAME map must go to any rollout of the trained policy.
RENAME_MAP='{"observation.images.birdseye": "observation.images.camera1", "observation.images.claw": "observation.images.camera2"}'
# LoRA: the backbone stays frozen, so it fits the 6 GB card (lerobot-train --peft.*; smoke: batch 4 = 1.7 GB,
# 0.19 s/step). lerobot's default SmolVLA targets: the action expert's q/v projections + state/action projections.
PEFT=(--peft.method_type=LORA --peft.r=32 --peft.lora_alpha=64)    # r 16 = lerobot default (0.7M params); 32 for more room
COMMON=(--policy.path=lerobot/smolvla_base --dataset.repo_id="$DATASET" --policy.device=cuda
        --policy.push_to_hub=false --rename_map="$RENAME_MAP" --wandb.enable=false "${PEFT[@]}")
COMMON_NOPEFT=("${COMMON[@]:0:${#COMMON[@]}-${#PEFT[@]}}")
case "${1:-}" in
  smoke)
    lerobot-train "${COMMON[@]}" --batch_size=4 --steps=30 --save_freq=30 --log_freq=5 --num_workers=4 \
      --output_dir=$OUT/shear_smolvla_lora_smoke --job_name=shear_smolvla_lora_smoke ;;
  full)
    lerobot-train "${COMMON[@]}" --batch_size="${BATCH:-16}" --steps="${STEPS:-20000}" --save_freq=2000 \
      --log_freq=100 --num_workers=8 --output_dir=$OUT/shear_smolvla_lora_v0 --job_name=shear_smolvla_lora_v0 ;;
  # No LoRA: SmolVLA's own recipe, the vision-language part frozen and the whole action expert trained
  # (train_expert_only, freeze_vision_encoder: the smolvla_base defaults). Dream Machines' π0.5 write-up (2026):
  # LoRA 45-53 % vs full fine-tune 76 %, and a bigger effective batch helped (256 > 64). The v0 LoRA policy reached
  # the right petiole but missed by a few mm (2026-11-04). Effective batch BATCH x ACCUM (8 x 4 = 32 on the 6 GB card).
  # ⚠️ lerobot counts every batch of 8 as a step, accumulating or not: 40k steps = 320k samples, as the LoRA run.
  # Smoke: 3.2 GB, 0.37 s per step -> ~4 h.
  expert-smoke)
    lerobot-train "${COMMON_NOPEFT[@]}" --batch_size=8 --accelerator.gradient_accumulation.steps=4 --steps=10 \
      --save_freq=10 --log_freq=2 --num_workers=4 --output_dir=$OUT/shear_smolvla_expert_smoke \
      --job_name=shear_smolvla_expert_smoke ;;
  expert)
    lerobot-train "${COMMON_NOPEFT[@]}" --batch_size="${BATCH:-8}" \
      --accelerator.gradient_accumulation.steps="${ACCUM:-4}" --steps="${STEPS:-40000}" --save_freq=4000 \
      --log_freq=50 --num_workers=8 --output_dir=$OUT/shear_smolvla_expert_v1 --job_name=shear_smolvla_expert_v1 ;;
  # v2 (2026-11-05): v1 continued on the dataset plus the 100 DAgger corrections (generate_corrections.py): the
  # scripted cutter's take-overs from where v1 went wrong (0 of 51 cuts, pressed on the stem). Starts from v1's last
  # checkpoint (the whole action expert again), half the steps: the policy already knows the reach.
  expert-v2)
    lerobot-train "${COMMON_NOPEFT[@]:1}" --policy.path="${FROM:-$OUT/shear_smolvla_expert_v1/checkpoints/040000/pretrained_model}" \
      --batch_size="${BATCH:-8}" --accelerator.gradient_accumulation.steps="${ACCUM:-4}" --steps="${STEPS:-20000}" \
      --save_freq=4000 --log_freq=50 --num_workers=8 --output_dir=$OUT/shear_smolvla_expert_v2 \
      --job_name=shear_smolvla_expert_v2 ;;
  # v3 (K7 2026-11-06): the v1 rig's dataset (base camera + claw, generate_episodes.py REPO_ID v1), the action expert
  # AND the vision encoder trained (--policy.freeze_vision_encoder=false; needs policy/patch_lerobot_smolvla.py run once
  # on the installed lerobot: otherwise train_expert_only freezes the vision encoder too, and the bf16 weights break
  # --accelerator.mixed_precision=fp16), the language
  # layers frozen, 100k steps (K7's call; Dream Machines saw a plateau past 30k, so a checkpoint every 10k and the
  # pick is by rollout cuts, not loss). DATASET_V3 / RENAME_V3 override the dataset and its camera map (the smoke
  # ran on v0 before v1 existed: DATASET_V3=krish5831/shear_sim_scripted_v0 RENAME_V3="$RENAME_MAP").
  v3-smoke | v3)
    RENAME_V1='{"observation.images.base": "observation.images.camera1", "observation.images.claw": "observation.images.camera2"}'
    V3=(--policy.path=lerobot/smolvla_base --dataset.repo_id="${DATASET_V3:-krish5831/shear_sim_scripted_v1}"
        --policy.device=cuda --policy.push_to_hub=false --rename_map="${RENAME_V3:-$RENAME_V1}" --wandb.enable=false
        --policy.freeze_vision_encoder=false --batch_size="${BATCH:-4}" --accelerator.gradient_accumulation.steps="${ACCUM:-8}")
    if [ "$1" = v3-smoke ]; then
      lerobot-train "${V3[@]}" --steps=10 --save_freq=10 --log_freq=2 --num_workers=4 \
        --output_dir=$OUT/shear_smolvla_v3_smoke --job_name=shear_smolvla_v3_smoke
    else
      lerobot-train "${V3[@]}" --steps="${STEPS:-100000}" --save_freq=10000 --log_freq=50 --num_workers=8 \
        --output_dir=$OUT/shear_smolvla_v3 --job_name=shear_smolvla_v3
    fi ;;
  *) echo "usage: $0 smoke | full | expert-smoke | expert | expert-v2 | v3-smoke | v3   (full: LoRA, BATCH=16 STEPS=20000; expert: BATCH=8 ACCUM=4 STEPS=40000; expert-v2: from v1, STEPS=20000)"; exit 1 ;;
esac
