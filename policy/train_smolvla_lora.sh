#!/usr/bin/env bash
# train_smolvla_lora.sh: SmolVLA fine-tuned with LoRA on SHEAR's scripted sim passes, on the laptop GPU
# (K7 2026-11-04). Adapted from B.L.U.S.H harvest/scripts/train_smolvla.sh.
#   policy/train_smolvla_lora.sh smoke    a few steps: memory and speed on the 6 GB RTX 1000 Ada
#   policy/train_smolvla_lora.sh full     the real run
# Run outside Claude Code's sandbox (the GPU isn't visible inside it), in the lerobot env.
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
case "${1:-}" in
  smoke)
    lerobot-train "${COMMON[@]}" --batch_size=4 --steps=30 --save_freq=30 --log_freq=5 --num_workers=4 \
      --output_dir=$OUT/shear_smolvla_lora_smoke --job_name=shear_smolvla_lora_smoke ;;
  full)
    lerobot-train "${COMMON[@]}" --batch_size="${BATCH:-16}" --steps="${STEPS:-20000}" --save_freq=2000 \
      --log_freq=100 --num_workers=8 --output_dir=$OUT/shear_smolvla_lora_v0 --job_name=shear_smolvla_lora_v0 ;;
  *) echo "usage: $0 smoke | full   (full: BATCH=16 STEPS=20000 by default)"; exit 1 ;;
esac
