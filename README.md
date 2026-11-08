# S.H.E.A.R (Selective Harvest of Excess And Redundant leaves)

Vinea's deleafing and pruning module, built on an SO-101 arm. The arm cuts old tomato
leaves off flush at the stem, using the stock rigid jaws with a blade at the tips. A fixed
rule picks the leaves. The grower sets that rule's parameters (how far up to strip, how many
leaves to keep, how many cuts per pass). The learned part is the cut itself.

v1 is deleafing. Side-shoot pruning and truss pruning are on the roadmap.

The plan and the research behind the rule are in the Vinea vault, under `08 Modules/SHEAR/`.

## Where it stands (2026-11-08)

Everything so far runs in a MuJoCo simulation. Nothing has run on the real arm yet.

- **The sim is built** (`sim/`, details in `sim/README.md`): the cut itself, random dwarf cherry tomato
  plants, the leaf rule, the mini greenhouse with three potted plants, and a scripted cutter that plans
  every arm move with RRT and does a full deleafing pass along the row.
- **The cameras changed on 2026-11-06 (the "v1 rig").** The policy now sees a **base camera** (low, on a
  bracket next to the arm's base) and the **claw camera** on the wrist. The old birdseye camera on a mast
  behind the row is still in the scene, but only the label tools use it.
- **Training data:** `krish5831/shear_sim_scripted_v1`, 200 scripted passes recorded on the v1 rig
  (private on the Hugging Face Hub, also local).
- **Policy:** SmolVLA. v3 trained to 50k steps (`krish5831/shear_smolvla_v3`). v3b carries on from v3's
  50k weights with a fixed learning-rate schedule (`krish5831/shear_smolvla_v3b`). Neither cuts reliably
  yet (see "Policy" below).
- **No vision model (YOLO) is trained yet.** The sim makes the labels for it.

## What is where

    rule/      the leaf rule (plain Python, no MuJoCo) and its hand-checked test cases
    sim/       the MuJoCo sim, the scripted cutter, dataset recording, policy rollouts
    policy/    SmolVLA training: a laptop script and a Colab/Kaggle notebook
    outputs/   local training runs (not in git)

## Try it

    cd sim
    ./fetch_so101.sh                     once: downloads the SO-101 model
    conda activate lerobot
    python check_cut.py --windowed       a window opens: the jaw closes, the blades cut a leaf, it drops
    python check_cutter.py --windowed    a window opens: the scripted cutter does a whole pass along the row

Each step of the sim has a check that prints PASS/FAIL and a `--windowed` view to watch. All of them are
in `sim/README.md`.

## Policy: SmolVLA trained on the scripted passes

The policy is SmolVLA (`lerobot/smolvla_base`), fine-tuned to copy the scripted cutter: it gets the joint
angles, the two camera images and a task sentence, and it outputs the next arm moves.

### Training

There are two ways to train:

- **On the laptop GPU** (6 GB): `policy/train_smolvla_lora.sh <mode>`. Run it outside Claude Code's sandbox
  (the GPU isn't visible inside it), in the lerobot env.

  | mode | what it trains | data | output (`outputs/train/`) |
  |---|---|---|---|
  | `smoke`, `full` | v0: LoRA adapters only, 20k steps | `shear_sim_scripted_v0` | `shear_smolvla_lora_v0` |
  | `expert-smoke`, `expert` | v1: the whole action expert, no LoRA, 40k steps | `shear_sim_scripted_v0` | `shear_smolvla_expert_v1` |
  | `expert-v2` | v2: v1 carried on, 20k steps | v0 + 100 correction episodes | `shear_smolvla_expert_v2` |
  | `v3-smoke`, `v3` | v3: action expert **and** vision encoder, 100k steps | `shear_sim_scripted_v1` | `shear_smolvla_v3` |

  The v3 modes need `python policy/patch_lerobot_smolvla.py` run once on the installed lerobot. Without
  it, lerobot freezes the vision encoder anyway, and fp16 training crashes on the bfloat16 weights.

- **On a free T4 GPU, Google Colab or Kaggle**: `policy/colab_train_v3.ipynb`. v3 was trained this way,
  and v3b is being trained this way now. The notebook installs the pinned lerobot, applies the same patch, trains, and saves a
  checkpoint to the Hub every 5k steps. If the session dies, run it again and it picks up from the last
  checkpoint. The notebook's first cell explains the setup for each platform.

Once a checkpoint exists, watch it drive the sim arm (next section).
  - v3: from `lerobot/smolvla_base`, stopped at 50k steps, repo `krish5831/shear_smolvla_v3`.
  - v3b (the notebook's current setting, in training): starts from v3's step-50k weights, with a short warm-up and a
    learning rate that slows down over exactly the 30k steps of the run (one 12-hour Kaggle session).
    Why: v3's preset schedule had already dropped the learning rate to its floor at 30k, so the loss
    hardly moved after that (0.042 at 30k, 0.037 at 50k). Repo `krish5831/shear_smolvla_v3b`.

⚠️ **Camera names.** SmolVLA expects cameras called `camera1`, `camera2`, ... The base camera becomes
`camera1` and the claw `camera2`. This map is saved inside each checkpoint, and `sim/rollout_policy.py`
uses the saved one. The v0, v1 and v2 checkpoints were trained on the birdseye, so by the code they
expect a birdseye image that today's sim no longer gives them: use v3 or later.

### Testing a checkpoint

The training loss says little about whether the arm cuts, so every checkpoint is tested in the sim on
plants it never saw. Download one, then watch it:

    hf download krish5831/shear_smolvla_v3 --include "checkpoints/050000/pretrained_model/*" --local-dir outputs/train/shear_smolvla_v3_hub
    python sim/rollout_policy.py --ckpt outputs/train/shear_smolvla_v3_hub/checkpoints/050000/pretrained_model --rtc 10 --smooth 0.5 --windowed --rows 1

For v3b, swap in `shear_smolvla_v3b` and a step that is already on the Hub (a checkpoint lands every 5k steps).

Without `--windowed` it runs 10 rows, records videos, and scores every cut against the rule, next to the
scripted cutter on the same rows. All its options, and `sim/check_replay.py` (which checks the dataset
itself), are in `sim/README.md` under "Testing a trained policy".

What the tests show so far (local rollouts, 10 rows each):
- v1 made no cut, and pressed something it mustn't in every row.
- The v3 checkpoints from 5k to 40k steps (run with `--rtc 10`) made at most one right cut in 10 rows, none
  of them flush, and pressed something they mustn't in nearly every row. 50k has only 2 rows scored so far.
- The scripted cutter cut 9 of the rule's 52 leaves on the same rows.

## Git

Work happens on `dev`. `main` is protected and only takes PRs from `dev`; the
`only-dev-into-main` check enforces that.
