# SHEAR sim

A MuJoCo simulation of the SHEAR deleafing module: an SO-101 arm in a mini greenhouse, cutting old leaves
off dwarf cherry tomato plants. It was built step by step. Each step below has a check that prints
PASS/FAIL and a `--windowed` view to watch. Later steps record a training dataset and test trained
policies.

The plan and the research behind it are in the Vinea vault, under `08 Modules/SHEAR/`.

## Setup

    ./fetch_so101.sh          once: the SO-101 model into sim/assets/so101 (gitignored)
    conda activate lerobot    it has mujoco and lerobot

The commands below run from this folder (`sim/`). Heavy runs (rendering, episode recording, rollouts) go under
a memory fuse, so a runaway job can't freeze the laptop. The limit was 9 GB and was raised to 20 GB on
2026-11-03:

    systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 python ...

⚠️ Not under `ulimit -v`: the lerobot import reserves so much address space that the EGL renderer then
fails ("Offscreen framebuffer is not complete").

On Wayland the MuJoCo window opens without a title bar. Close it with Ctrl+C in the terminal.

Scripts that use the GPU (`rollout_policy.py`, `generate_corrections.py`) must run outside Claude Code's
sandbox: the GPU isn't visible inside it.

## Step 1: one stem, one leaf, the cut

    python check_cut.py --windowed     watch it: the jaw closes, the blades cut the petiole next to the stem,
                                       the leaf drops, and it starts over
    python check_cut.py                PASS/FAIL checks, a few seconds

- `scene.py`: the SO-101 holds one pose (all joints at zero, the wrist rolled so the jaw hinge is level).
  The stock jaws collide through boxes, and each fingertip has blades. The plant is built around the blades.
- `plant.py`: the stem; the leaf (rigid petiole, a midrib that droops, leaflets), welded to the stem;
  and the `Cutter`. The cut happens only when both blade edges press the petiole hard enough for a full
  stroke. Then the leaf drops and the stub it leaves is measured from the stem surface.

The checks cover these cases:
- the leaf hangs
- the blades hold a stalk they can't cut
- a cut at 2 mm and at 8 mm leaves that stub
- the leaf falls clear and the jaw closes through
- the stem isn't touched
- no cut at 25 % torque, with only the pads squeezing, or with the jaw closing on air
- a 20 N pull doesn't tear the leaf off

They also print what matters for the real gripper: the blade force the stock SO-101 jaw reaches on a
7 mm petiole. That is a sim estimate from the 12 V servo's stall torque: about 17 N at 50 % torque, and
7.5 N once the stalled servo drops to 25 %. Both are upper bounds, because the real PSU sags under load.
⚠️ Since 2026-10-29 the fingertips carry a blade pair on both faces (`scene.BLADE_FACES`). Both pairs press the
petiole at once and share the squeeze: 8.8 N per pair at 50 % torque, 3.9 N at 25 %.

All plant sizes and the cut force are guesses marked `unverified` in the code, until measured on real
plants.

## Step 2: random plants, and the leaf rule's labels

    python check_plant.py --windowed --seed 3    watch one dwarf cherry tomato settle (--profile highwire for the
                                                 commercial plant). Red = leaves the rule cuts, orange =
                                                 flagged for a person, blue = held back (keep guard, pass cap).
                                                 --truth for the plant's own colours. The terminal prints the stem.
    python check_plant.py                        PASS/FAIL checks on both profiles, about 20 s
    python ../rule/check_rule.py                 just the rule's hand-checked cases (no mujoco)

- `../rule/select_leaves.py`: the leaf rule from the research note (section 5), plain Python with no mujoco,
  so the sim labeller, the sim judge and the real robot share one copy. It takes the stem's nodes from the
  bottom up and the grower's `Settings`, and returns the leaves to cut, oldest first, with a decision and a
  reason for every leaf: cut / keep / flag / guard / cap / unknown / scar / blocked. It never cuts an
  unknown, and it fills gaps in the crop's leaves-per-truss pattern with unknowns (`Settings.leaves_per_truss`:
  3 for high wire, 1 for a determinate dwarf). `confirm_cut` is the last check at the claw before each cut;
  until there is a vision model, the sim fills it from the truth.
- `stemgen.py`: random plants, in two profiles.
  - `dwarf_cherry` (default): a dwarf, determinate cherry tomato like 'Tiny Tim' (20–45 cm, ~2 cm fruit), at
    its real size, in a pot. Real-size high-wire leaves were far too big and crowded for the 0.52 m mini
    greenhouse. Built whole. First truss after 5–7 leaves, then one every 1–2 leaves, ending in a truss.
  - `highwire`: three leaves then a truss, endless. Only the work band is built; the stem above it is
    abstract (rule nodes with no geoms, which the keep guard counts). Kept for real-greenhouse scale.
  - Both: trusses ripen from the bottom up, leaves below the last pass are flush scars, 0–4 have aged past
    the line, an earlier pass sometimes missed one, old leaves are often yellow, a few diseased, sometimes a
    young crop. Every leaf differs: length, leaflet pairs, leaflet size, petiole thickness, angle, droop,
    colour, and where it sits around the stem. Leaves are compound tomato leaves (`plant.py` style "tomato").
  - ⚠️ No source gave 'Tiny Tim' leaf sizes, and nobody deleafs a dwarf: its leaf numbers and rule settings
    are placeholders until a bought plant is measured.
- `truss.py`: the trusses, copied from B.L.U.S.H `tomato.py` (see NOTICE.md for what changed).
- Every plant geom has a segmentation class (`stemgen.seg_class`: stem, petiole, midrib, leaflet, scar,
  rachis, pedicel, calyx, fruit) and a node, and every present leaf has its cut point (where the petiole
  meets the stem surface). Those are the labels for the vision model (step 4).

The checks cover these cases:
- the hand-checked cases: small stems with the cuts worked out by hand
- across 400 plants per profile: the profile's leaves-per-truss pattern, and every cut is a present leaf in
  the band, below the strip line, oldest first, within the pass cap and the keep guard
- the labels vary: plants with no cuts, with some cuts, with the full cap, with the guard or cap holding
  leaves back, with flags, young crops, missed leaves
- six dwarf and two high-wire plants built in MuJoCo: inside the geom budget with the arm, everything settles,
  nothing on the floor, every geom classed, and the label cut points sit where the settled petioles really are

⚠️ `min_leaves_kept = 16` is a placeholder from an open-source sim, not from growers. On many plants it
decides how many leaves come off, so ask growers before trusting the labels.

## The scene: mini greenhouse, plants, arm, cameras

    python check_layout.py --windowed            the whole scene (press [ and ] to switch to the cameras)
    python check_layout.py --picture out.png     one image: overview with each camera's view cone, the birdseye
                                                 frame, the claw frame
    python check_layout.py                       PASS/FAIL checks

- `greenhouse.py`: the laser-cut 6 mm ply frame (geometry from B.L.U.S.H, see NOTICE.md).
- `layout.py` (K7 approved 2026-10-28, changed 2026-11-06 for the "v1 rig"):
  - three dwarf cherry tomatoes in pots along row R1 (p0, p1, p2)
  - the SO-101 in the aisle, its base square to the row, in front of plant p0. Since 2026-11-06 the base sits
    5 cm back from the aisle centreline (`ARM_SETBACK`): the arm was cramped against p0, and on the 3 rows
    tried, more cuts could be planned from further back, not fewer.
  - the **base camera** (since 2026-11-06): low on a bracket off the arm's base plate, 14 cm to the left,
    15 cm forward, 4 cm up, aimed at p0's stem. It sees the blade and the petiole from the side. This is
    the camera the policy and the dataset use, with the claw camera.
  - the **claw camera** on the SO-101's stock wrist mount
  - the **birdseye** on a mast behind the empty row R0: level, looking straight at the row, wide enough for
    the whole row. Until 2026-11-06 the dataset used it. K7 replaced it because it looks along the arm's own
    line of reach, so at the cut the arm hides the cut point. It stays in the scene for the label tools
    (step 4) and for `check_layout.py`.
- `lens.py`, `cameras.py`: the InnoMaker lens model and the renderers, from B.L.U.S.H (camera hardware only).
  `cameras.CAMERAS` lists the policy's cameras: base and claw.
- `ik.py`: position IK for the fingertips, used for the ready pose. The jaw's orientation is `cutpose.py`'s
  (step 3).

The checks cover these cases:
- the birdseye sits on its mast and never sees it; it sees every leaf's cut point and every plant top to bottom
- the arm's base stands outside the birdseye frame, square to the row, in front of its plant
- the claw camera looks along the jaw
- the plants settle inside the greenhouse
- IK reaches the ready pose

## Step 3: the cut across the ways the arm meets a leaf

    python check_approach.py --windowed --case az=-60,el=45      watch one case (keys in the docstring)
    python check_approach.py                     PASS/FAIL checks, about 20 s (--table: one line per case)
    python check_approach.py --reach             reach survey on the greenhouse plants, kinematic, ~6 min

- `cutpose.py`: the cut pose and the 5-joint IK that reaches it. The default cut is **flush**: the blade plane
  parallel to the stem, 2 mm off it, the way growers cut ("tot tegen de stam"). The jaw then closes along the
  stem, as in step 1: hinge level, fingers level. Twist (the blade turned about the stem, off the petiole's own
  direction) only slants the cut face, so it is given up first.
- `check_approach.py`: step 1's rig with the arm moved there by `cutpose.CutIK`. It covers leaf directions,
  blade lean, finger pitch, stub length, petiole size, and aim and sag errors. The stem is dwarf-size, the petiole
  bends (`LeafParams.bend`), and the cut force scales with the petiole (`plant.cut_force_for`, unverified).
- `reach.py`: for every leaf on 20 random dwarf plants, whether the arm finds a clean flush pose clear of the
  stem, truss stalks, pot and frame. It tries the base where it stands, the base slid along the aisle, and the
  far side of the row.

What it found (2026-10-29):
- **Flush, not square.** A blade square to a petiole that rises 5-30 deg leans into the stem and can't cut a
  2-6 mm stub (15/15 cases: the moving pad or the blade meets the stem). Flush cuts leave 2.0-2.3 mm at every
  petiole angle.
- **2 mm is the closest clean flush cut** with the stock fingertips. At 1 mm the jaw presses the stem with
  ~38 N. 2, 4 and 8 mm stubs follow the offset on 2.4-3.6 mm petioles.
- **Tolerances:**
  - finger pitch -22..+10 deg (+52 puts the finger along the stem)
  - ±2 mm along the blades
  - up to 1 mm of sag above the fixed edge (at 2 mm the squeeze goes into bending the petiole)
  - blade lean < 5 deg. This is not caused by the 25 deg jaw opening: at 0-25 deg openings, 10-20 deg of
    lean still puts the moving pad into the stem, and 5 deg cuts only at a 4 mm stub with the jaw open 5 deg or
    less (`--case mode=square,el=5,offset=4,jaw=5`). So the jaw should come in barely open.
- ⚠️ **The SO-101's 5 joints decide which leaves it can cut.** Its fingers stay in the pan plane, so a flush cut
  fits a petiole that points along the row, within ±30 deg. Leaves pointing at or away from the arm would need
  the fingers straight up or down, which the arm can't reach in the work band.
- **Blades on both fingertip faces** (K7 2026-10-29). With one pair, a leaf pointing to the arm's right needed the
  jaw flipped over (wrist roll ~183 deg). The wrist has physical stops at ~±165 deg, so that leaf couldn't be cut
  flush. Now the other pair cuts it, with no flip. The cost is that both pairs share the squeeze (step 1 above).
  An 8 mm stub on a 3.6 mm petiole then doesn't part; 2-4 mm stubs do.
- **Reach on the dwarf row** (kinematic, both blade pairs, fruit may be pushed 15 mm, stalks may not). Measured
  before the base moved 5 cm back on 2026-11-06:

  | where the base is | leaves | rule's cuts | one blade pair, for comparison |
  |---|---|---|---|
  | one base | 20 % | 23 % | 9 % / 13 % |
  | base sliding ±15 cm along the aisle | 65 % | 71 % | 31 % / 29 % |
  | sliding, plus the far side of the row | 71 % | 81 % | 58 % / 58 % |

  (Rerun 2026-11-01 after the truss fix in step 5; before it, both pairs read 16/16, 53/65, 66/71 %.)

  What's left: twist (petioles pointing at or away from the arm), then truss stalks in the way.
- The rig's leaflets pass through the hand: it tests the cut, not the approach. Approach paths, servo sag and
  pushing past real leaves are step 5's (the scripted cutter).

## Step 4 (started): cut-point labels for the vision model

    python check_labels.py --windowed            the row with every cut point marked by the rule's decision
    python check_labels.py                       PASS/FAIL checks on four rows, ~10 s
    python check_labels.py --picture out.png     each plant zoomed in, leaves numbered, one legend line per leaf
    python render_cutpoints.py --pilot 6         a PILOT set: 6 labelled frames into ~/.cache/shear_sim/cutpoints_pilot

These tools label the **birdseye** view, not the base camera.

- `labels.py`: YOLO-pose labels. Each object has a box and two keypoints.

  | class | keypoint 0 | keypoint 1 |
  |---|---|---|
  | leaf (green / yellow / diseased) | the cut point | 15 mm out along the petiole (its direction, for the IK) |
  | truss (green / turning / ripe / harvested) | where the stalk leaves the stem | 15 mm along the stalk |
  | scar | the scar | not labelled |
  | stem | bottom | top |

  That is what `rule/select_leaves.py` needs (the node kinds, leaf health, truss stage, bottom-up order) and
  what `cutpose.py` needs.
- Only leaves are cut, never trusses: the cut point is always where a leaf's petiole meets the stem, and only
  leaves are ever targets (`cutpose.py`, `plant.Cutter`). A truss's keypoint only tells the rule where the
  truss sits on the stem. ⚠️ The first preview drew truss keypoints as red dots, and a hidden leaf cut point sat
  on top of the fruit hiding it, so cut points looked like they were on the trusses (K7, 2026-10-31). Previews
  now draw leaves only, one plant per panel (`preview/*_leaves.png`). Check: 56 of 57 seen cut points lie within
  3 px of their own petiole; the 85 hidden ones are behind fruit (29), a stem or another leaf.
- Whether a keypoint is visible comes from a depth render, not the segmentation alone. ⚠️ A cut point behind
  the stem projects onto the stem's front; the segmentation calls it seen, but it's hidden.
- **Finding:** the birdseye sees only about a quarter of the rule's cut points (7 of 27 in the pilot). The low
  leaves are behind trusses or on the stem's far side, and parking the arm didn't change that. The birdseye can
  find the leaves; the claw camera has to find most cut points.
- Not done yet: claw-camera and base-camera labels, a randomised label set, training a YOLO model.
  `render_cutpoints.py` is capped at 24 frames until K7 approves the pilot.

## Step 5: the scripted cutter, every move planned with RRT

    python check_cutter.py --windowed            watch a whole pass on one known row, in real time (it loops)
    python check_cutter.py                       PASS/FAIL on that row, ~30 s (--start N: another row)
    python generate_cuts.py --trial 6            a TRIAL batch: 6 passes as videos + summary,
                                                 into ~/.cache/shear_sim/cuts_trial (capped at 12)

- `rrt.py`: RRT-Connect in joint space, then shortcutting. K7 (2026-11-01): every arm move is planned with RRT.
  It plans against three lists:
  - **hard**: stems, pots, the frame, and since 2026-11-06 the floor. Never touched, 4 mm clearance.
  - **soft**: fruit, truss stalks, and the stalks and midribs of kept leaves. Avoided when a route exists; when
    none does, they may be brushed, and the episode says so. Leaflets just get pushed.
  - **blade guard**: the blades never come within 2 mm of a truss stalk. A blade can cut a truss off; a finger
    pad resting on it doesn't.
- `cutter.py`: **one episode is a full deleafing pass** (K7 2026-11-03). It starts and ends at the robot's start
  pose (B.L.U.S.H's `START_NORM`, folded beside the base) and cuts the rule's leaves on p0, then p1, then p2,
  oldest first. Any leaf without a clean cut is skipped and logged. Each cut:
  1. an RRT route from wherever the arm is to the pre-cut pose, 4 cm back along the fingers
  2. a straight slide onto the petiole
  3. close: the blades cut and the leaf drops
  4. a straight slide back out

  After the last cut, an RRT route back to the start pose. IK (`cutpose.py`) gives the goal poses; RRT gives
  the routes. The servos follow a 30 Hz reference with a correction for servo sag, and the arm lands within
  ~0.2 mm. Since 2026-11-06 a route waits for the arm to catch up (at most 6 deg behind), and the route home
  first tries to keep 15 mm clear. That came from a trial where a route home went into the floor and the
  physics dragged the arm into p0's pot (300 N) and the floor rail (417 N).

  The judge checks, per cut:
  - the right leaf came off, and no other one did
  - the stub is 1.5-3.5 mm
  - no stem, pot or frame was pressed
  - no truss stalk took more than 1 N
  - no blade touched a truss stalk
- `generate_cuts.py`: a row of three random dwarf plants per episode. Before the pass, each pot is turned once,
  to the angle that lets the arm cut the most of that plant's leaves (`generate_cuts.best_turns`), as a person
  turns pots on the bench. The base stays put (K7: no sliding base, SPINE last). The videos show the side view,
  the base camera and the claw camera, with the sim's own labels drawn on (what a YOLO model would learn; no
  model is trained yet).
- Found on the way (2026-11-01):
  - ⚠️ `stemgen` put every truss ~40 deg off the leaf below it instead of opposite it, so it drooped across the
    cut point. Fixed. That changed every plant.
  - ⚠️ The follower's shoulder goes to ±102 deg and the MJCF's to ±100. The IK now stays inside both.
  - ⚠️ A petiole rising across the fingertips meets the outer blade pair first when the jaw is flipped. The
    inner pair then cuts the rest as the jaw keeps closing (modelled in `plant.Cutter`, unverified).
  - From a fixed base, without turning the pots, none of 150 plants had a cut the arm could plan. Turning the
    pot so the oldest leaf to cut points along the row, only about 1 plant in 20 had one. The dwarf's trusses
    crowd the leaves.
- With each pot at its best angle (2026-11-03), the arm cuts about 1 leaf per plant on the dense dwarf, and
  often none on p1 and p2, at the edge of the fixed arm's reach.

## Recording the dataset

    lerobot-dataset-viz --repo-id krish5831/shear_sim_scripted_v1_pilot --episode-index 0
                                                 opens a viewer window: what got recorded, frame by frame
    python generate_episodes.py --pilot          5 episodes into krish5831/shear_sim_scripted_v1_pilot
    python generate_episodes.py --episodes 200   the real batch into krish5831/shear_sim_scripted_v1 (--resume after a crash)

`generate_episodes.py` records scripted passes (step 5) as a LeRobot dataset. The current one is
`krish5831/shear_sim_scripted_v1`: 200 passes on the v1 rig (796 rows tried, about 6.5 hours, 2026-11-06).
The older `krish5831/shear_sim_scripted_v0` used the birdseye instead of the base camera.

- Every frame is recorded the way `lerobot-record` does it, at 30 fps:
  - `observation.state`: the 6 joints in lerobot's `use_degrees=false` units, through the real follower's
    calibration (`arm_map.py`)
  - `action`: what the servos were told
  - `observation.images.base` / `.claw`: clean 640x480 frames (no labels drawn)
  - the task sentence ("Cut the leaves below the lowest ripening truss off flush at the stem.")
  - `robot_type` `so_follower`, so sim and real SHEAR data can merge
- Only passes the judge accepts are kept. Every attempt, kept or not and why, is logged in
  `~/.cache/shear_sim/episodes/`. That log is what `check_replay.py` and `rollout_policy.py --train-rows` read.
- The mix (since v1): up to 10 % "hold" episodes, where the rule cuts nothing and the arm stays at its start
  pose; rows with cuts on p0 only are skipped half the time (v0 had 180 / 34 / 0 cuts on p0 / p1 / p2); and the
  arm starts up to 3 deg off its start pose.
- Domain randomisation, on by default (`randomize.py`, adapted from B.L.U.S.H; K7 2026-11-03). It varies:
  - light brightness and direction, and the ambient light
  - greenhouse, floor and pot shades
  - leaf, stem and fruit colours, each inside its class, so a green leaf stays green and a ripe fruit red
  - small camera pose jitter

  The physics isn't randomised. `--no-dr` gives the plain look.
- The script writes locally only. Nothing goes to the Hub without K7's OK (v1 is on the Hub as a private repo).

## Correction episodes (sim DAgger)

    python check_cutter.py --windowed            watch the scripted cutter, which does the takeover
    python generate_corrections.py --ckpt <policy pretrained_model> --episodes 100 --repo-id <dataset>

`generate_corrections.py` has no live view of its own. To watch the two halves of a correction:
`rollout_policy.py --windowed` shows the policy driving, and `check_cutter.py --windowed` shows the scripted
cutter's cut and its return to the start pose.

Why: the v1 policy reached the right leaf but drove the jaw into the stem in the last few mm (0 of 51 cuts,
2026-11-05). Every scripted demonstration ends cleanly 2 mm from the stem, so the policy never saw how to
recover from a small error.

How: the policy drives the arm (with RTC, as in `rollout_policy.py`). As soon as it gets into trouble (within
a few mm of a stem, pot or the frame, a blade near a truss stalk, hovering near a leaf without cutting, or
after 20 s), the scripted cutter takes over: it backs off if stuck, cuts the nearest leaf the rule wants, and
goes back to the start pose. Only the takeover is recorded, appended to the dataset. Nothing is recorded
while the policy drives: that would teach its mistakes.

On 2026-11-05, 100 correction episodes were appended to `shear_sim_scripted_v0` (now 300 episodes; the 200-pass
original is backed up as `shear_sim_scripted_v0_backup_200eps`). That data trained v2 (`expert-v2` in
`policy/train_smolvla_lora.sh`), which still cut nothing.

⚠️ Appending changes the dataset in place: back it up first. Without `--repo-id`, the script now appends to
`generate_episodes.py`'s dataset, which is v1, the current training set. Use `--new` to create a fresh one.

## Testing a trained policy

Two tools. `rollout_policy.py` lets a trained policy drive the sim arm. `check_replay.py` checks the
dataset itself, without any policy.

### rollout_policy.py

    python rollout_policy.py --ckpt <checkpoint>/pretrained_model --windowed --rows 1    watch one row live
    python rollout_policy.py --ckpt <checkpoint>/pretrained_model --rtc 10 --smooth 0.5  the 10-row test

Each row is a new row of plants the policy never trained on (seeds from 980000), set up like the training
episodes: pots turned the same way, domain randomisation on. The policy drives from the start pose at 30 fps.
Every frame it gets the joint angles, the base and claw images and the task sentence, exactly as the dataset
recorded them. A row ends after 60 s, or once the arm has been back at the start pose for 1.5 s.

Every cut is judged like the scripted passes: was it a leaf the rule asked for, how long is the stub, and did
the arm press a stem, truss stalk, pot or the frame. The scripted cutter then runs on the same row, for
comparison.

Options:
- `--rows N`: how many rows (default 10; with `--windowed`, row after row until you close the window).
- `--windowed`: watch live in a MuJoCo window, in real time. No videos, no scripted comparison.
- `--start N`: skip the first N rows. Finishes an interrupted test, keeping its videos: `--start 8 --rows 2`
  with the same `--out` runs rows 8-9.
- `--train-rows <attempt log .jsonl>`: test on rows the policy **trained** on, instead of new ones. It first
  checks (PASS/FAIL) that each row rebuilds exactly as recorded. If the policy fails there too, it's a pipeline
  bug or the model is underfitting. If it works there but not on new rows, it needs more or more varied data.
- `--n-action-steps N`: how many of each predicted chunk of moves are played before the policy looks again
  (default: the checkpoint's, 50, which is 1.7 s blind). Fewer means it reacts more often.
- `--rtc N`: real-time chunking. A new chunk every N moves, steered to blend into the rest of the old one, so
  the handover doesn't jump. `--rtc-horizon` (default 20) sets how many old moves it blends toward.
- `--smooth ALPHA`: smooths the commands; ALPHA is the weight of the new one (0.5 as SockBot used; 0 = off).
- `--no-video`, `--out DIR`.

It writes `rollouts.json`, `rollouts.txt` and `rollout_KK.mp4` (side view, base camera, claw camera, the sim's
labels drawn on) into `~/.cache/shear_sim/rollouts/<step folder>_<settings>` unless `--out` says otherwise:
for `.../checkpoints/020000/pretrained_model` with `--rtc 10` that is `020000_rtc10h20` (without RTC,
`020000_n50`), with `_train` added for `--train-rows`. A run with `--start` writes `rollouts_fromNN.*` beside
the first part's summary.

### check_replay.py

    python check_replay.py --log ~/.cache/shear_sim/episodes/shear_sim_scripted_v1_<time>.jsonl --episodes 0 --windowed
                                                 watch episode 0's recorded actions play back
    python check_replay.py --log ~/.cache/shear_sim/episodes/shear_sim_scripted_v1_<time>.jsonl
                                                 PASS/FAIL on 10 episodes (--episodes 0 23 44: these ones)

It rebuilds a recorded episode's row, puts the arm where the recording starts, and plays the recorded actions
back the same way a policy's actions are played. If the recorded cut doesn't happen, recording and rollout
move the arm differently, and no policy trained on that data could ever cut. Checks, each PASS/FAIL: the
episode is the right row, the joints track the recording, and the judge counts the same cuts as the scripted
pass. It reads the local dataset (`~/.cache/huggingface/lerobot/krish5831/shear_sim_scripted_v1` by default,
`--repo` for another), writes nothing, and needs no GPU. Episode 0 passed on 2026-11-08 (1 cut, joints within
0.43 of the recording, in lerobot's normalised units).

## Other files

- `arm_map.py`: converts between sim joint angles (radians) and the real follower's calibrated units, so sim
  and real data line up. Copied from B.L.U.S.H.
- `fetch_so101.sh`: downloads the SO-101 model (see Setup).

Where the copied code comes from is in `NOTICE.md`.
