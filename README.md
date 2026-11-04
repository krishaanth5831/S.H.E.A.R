# S.H.E.A.R (Selective Harvest of Excess And Redundant leaves)

Vinea's deleafing and pruning module, built on an SO-101 arm. The arm cuts old tomato
leaves off flush at the stem, using the stock rigid jaws with a blade at the tips. A fixed
rule picks the leaves. The grower sets that rule's parameters (how far up to strip, how many
leaves to keep, how many cuts per pass). The learned part is the cut itself.

v1 is deleafing. Side-shoot pruning and truss pruning are on the roadmap.

## Status

Sim steps 1 and 2 are built.
- Step 1: one stem that collides, one leaf whose petiole the blades cut, `sim/check_cut.py`.
- Step 2: random plants (`sim/stemgen.py`) and the leaf rule (`rule/select_leaves.py`) that labels which
  leaves come off, `sim/check_plant.py`. Default plant: a dwarf cherry tomato at its real size.
- The scene: the mini greenhouse, three potted plants, the SO-101 square to the row, the birdseye on a mast
  looking straight at the row, the claw camera on the wrist, `sim/check_layout.py`.

Each check prints PASS/FAIL and has a `--windowed` view (see `sim/README.md`). The plan and the research
behind the rule are in the Vinea vault, under `08 Modules/SHEAR/`.

## Git

Work happens on `dev`. `main` is protected and only takes PRs from `dev`; the
`only-dev-into-main` check enforces that.

## Policy: SmolVLA + LoRA on the sim passes

    policy/train_smolvla_lora.sh smoke     a few steps on the laptop GPU (memory and speed)
    policy/train_smolvla_lora.sh full      batch 16, 20k steps (BATCH= / STEPS= to change)

Fine-tunes `lerobot/smolvla_base` on `krish5831/shear_sim_scripted_v0` with LoRA (`lerobot-train --peft.*`,
rank 32), into `outputs/train/shear_smolvla_lora_v0`.
- ⚠️ SmolVLA's base expects camera1..3: birdseye -> camera1, claw -> camera2 (`RENAME_MAP`). A rollout has to
  use the same map.
- Run it outside Claude Code's sandbox: the GPU isn't visible inside it.
