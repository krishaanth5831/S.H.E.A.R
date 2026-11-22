# S.H.E.A.R (Selective Harvest of Excess And Redundant leaves)

Vinea's deleafing and pruning module, built on an SO-101 arm. The arm cuts old tomato
leaves off flush at the stem, using the stock rigid jaws with a blade at the tips. It cuts;
it never snaps a leaf off. A fixed rule picks the leaves. The grower sets that rule's
parameters (how far up to strip, how many leaves to keep, how many cuts per pass). The
learned part is the cut itself.

v1 is deleafing. Side-shoot pruning and truss pruning are on the roadmap.

The plan and the research behind the rule are in the Vinea vault, under `08 Modules/SHEAR/`.

## Where it stands (2026-11-22)

Everything so far runs in a MuJoCo simulation. **Nothing has run on the real arm yet**, and
the cut force, plant sizes and blade forces in the code are sim estimates or guesses marked
`unverified` until they are measured.

**Parked since 2026-11-16.** Work is on Vinea's harvest module (B.L.U.S.H) for now; what is
learned there comes back here.

- **The sim is built** (`sim/`, details in `sim/README.md`): the cut itself, random dwarf
  cherry tomato plants, the leaf rule, the mini greenhouse with three potted plants, the
  cameras (a low **base camera** beside the arm and the **claw camera** on the wrist), and a
  scripted cutter that plans every arm move with RRT.
- **Teacher v3** (`sim/shear_teacher.py`): a deterministic, closed-loop cutter that makes one
  cut per episode and can steer back after a push. It cuts 47 of the 85 rows it can plan.
  This replaced the scripted cutter as the source of training data.
- **Dataset:** `krish5831/shear_sim_teacher_v3`, 600 teacher episodes (540 cuts, 60 "hold"
  episodes where the arm stays home), private on the Hugging Face Hub.
- **Policy:** ACT, trained 100k steps on that dataset (`policy/train_act_teacher.sh`). It gets
  the joint angles, both camera images, and the cut point and petiole direction as numbers.
  On its own it does not finish the cut, even on rows it trained on. The **hybrid** works
  better: the policy drives up to the leaf, then the teacher does the last slide and the
  cut. The hybrid cut 20 of 50 held-out rows the teacher can solve, and 30 of 40 training
  rows. ⚠️ In sim the teacher's slide knows the exact cut point, so this only tests the
  approach.
- **No vision model (YOLO) is trained yet.** The sim makes the labels for it.

The first try, SmolVLA trained on the scripted cutter, never cut reliably and was dropped
on 2026-11-08. Its checkpoints and the v0 dataset were deleted; the scripts are still in
`policy/` (see "Earlier route" below).

## What is where

    rule/      the leaf rule (plain Python, no MuJoCo) and its hand-checked test cases
    sim/       the MuJoCo sim: plants, scene, cut model, IK, RRT, scripted cutter, teacher v3,
               dataset recording, policy evaluation (sim/README.md lists every file)
    policy/    training scripts: ACT on teacher data (current), SmolVLA (earlier route)
    outputs/   local training runs, logs and evals (not in git)

## Setup

You need Linux with a display (for the viewer windows), conda, and an NVIDIA GPU only for
training and GPU rollouts.

1. A conda env called `lerobot` with LeRobot and MuJoCo 3.8.1 (the sim was built and tested
   on 3.8.1). The same env is shared with B.L.U.S.H.
2. Download the SO-101 model once (from TheRobotStudio/SO-ARM100, into `sim/assets/so101/`,
   which is gitignored):

       cd sim
       ./fetch_so101.sh

3. Activate the env: `conda activate lerobot`.

## Open the sim viewer

Run these from `sim/`. Each opens a MuJoCo window; close it with Ctrl+C in the terminal (on
Wayland the window has no title bar).

    python check_cut.py --windowed                       one leaf: the jaw closes, the blades cut it, it drops
    python check_layout.py --windowed                    the whole greenhouse scene ([ and ] switch to the cameras)
    python check_cutter.py --windowed                    the scripted cutter does a whole pass along the row
    python check_teacher.py push --row 910003 --windowed teacher v3 is pushed mid-approach and still cuts

Heavy runs (rendering, recording, rollouts) go under a memory fuse so a runaway job can't
freeze the machine, for example:

    systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 python check_cutter.py

Don't use `ulimit -v` instead: the lerobot import reserves so much address space that the
offscreen renderer then fails.

## Tests

There is no pytest suite. Every step has a check script that prints PASS/FAIL per case:

    python rule/check_rule.py                  from the repo root: the leaf rule's hand-checked cases, no MuJoCo
    cd sim
    python check_cut.py                        the cut (a few seconds)
    python check_plant.py                      random plants and the rule's labels (~20 s)
    python check_layout.py                     the scene and the cameras
    python check_approach.py                   the cut across the ways the arm meets a leaf (~20 s)
    python check_labels.py                     cut-point labels for the vision model (~10 s)
    python check_cutter.py                     a scripted pass on one known row (~30 s)
    python check_teacher.py repeat --row 910003   teacher v3 repeats itself exactly
    python check_teacher.py push --row 910003     teacher v3 recovers from pushes

⚠️ `check_cutter.py` fails on its default row (700033) since the camera rig changed on
2026-11-06; the other checks pass.

Lint: `ruff check .` from the repo root (settings in `ruff.toml`).

## Training and evaluating the policy

All of this is described step by step in `sim/README.md` under "Teacher v3 pipeline". In short:

    # record teacher episodes (one process per 20-episode part)
    cd sim && python generate_episodes.py --teacher --no-dr --hold-share 0.10 --noise-share 0.5 \
        --row-start 930000 --episodes 20 --max-attempts 900 --repo-id krish5831/shear_sim_teacher_v3_p00
    # train ACT (from the repo root, GPU, outside Claude Code's sandbox)
    RUN=act_teacher_v3 policy/train_act_teacher.sh krish5831/shear_sim_teacher_v3
    # watch a checkpoint drive (from sim/)
    python eval_onecut.py ../outputs/train/act_teacher_v3/checkpoints/<step>/pretrained_model --rows 3 \
        --n-action-steps 100 --windowed
    # the held-out test, policy alone and hybrid
    python eval_onecut.py <checkpoint>/pretrained_model --rows-from heldout_rows_v3.jsonl --n-action-steps 100
    python eval_onecut.py <checkpoint>/pretrained_model --rows-from heldout_rows_v3.jsonl --n-action-steps 100 --hybrid

Nothing is pushed to the Hub by these scripts.

## Earlier route: SmolVLA on the scripted cutter

Until 2026-11-08 the policy was SmolVLA (`lerobot/smolvla_base`) fine-tuned on
passes of the scripted cutter (`krish5831/shear_sim_scripted_v1`, 200 passes, still on the
Hub). No version cut reliably; the post-mortem (vault note "BLUSH + SHEAR Policy
Post-Mortem") found the main cause: the scripted cutter picked its leaf by a search the
camera can't see, and SmolVLA had no input for the target. The scripts stay for reference:

- `policy/train_smolvla_lora.sh <mode>`: laptop training (modes v0 to v3, see the script).
- `policy/colab_train_v3.ipynb`: the same on a free Colab/Kaggle T4.
- `policy/patch_lerobot_smolvla.py`: the lerobot patch the v3 modes need (trains the vision
  encoder, fixes fp16 on bfloat16 weights).
- `sim/rollout_policy.py`, `sim/generate_corrections.py`, `sim/check_replay.py`: rollouts,
  sim DAgger and the dataset replay check for that route (documented in `sim/README.md`).

## Git

Work happens on `dev`. `main` is protected and only takes PRs from `dev`; the
`only-dev-into-main` check enforces that.

## License

Source-available, not open source. Copyright (c) 2026 Krishaanth Ramaraj. You may read and
use the code under the PolyForm Strict License 1.0.0 (see `LICENSE`); commercial use,
modification and redistribution need a separate written license. Third-party files keep
their own licenses (see `sim/NOTICE.md`).
