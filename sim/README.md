# SHEAR sim

MuJoCo sim of the SHEAR deleafing module. The plan and the research behind it are in the Vinea vault,
under `08 Modules/SHEAR/`.

## Setup

    ./fetch_so101.sh          once: the SO-101 model into sim/assets/so101 (gitignored)
    conda activate lerobot    it has mujoco

Every run goes under the 9 GB cap:
`systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 python ...`

## Step 1: one stem, one leaf, the cut

    python check_cut.py --windowed     watch it: the jaw closes, the blades cut the petiole next to the stem,
                                       the leaf drops, and it starts over
    python check_cut.py                PASS/FAIL checks, a few seconds

On Wayland the window opens without a title bar. Close it with Ctrl+C in the terminal.

- `scene.py`: the SO-101 holds one pose (all joints at zero, the wrist rolled so the jaw hinge is level).
  The stock jaws collide through boxes, and each fingertip has a blade on its stem side. The plant is
  built around the blades.
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
    python ../rule/check_rule.py                 just the rule's golden cases (no mujoco)

- `../rule/select_leaves.py`: the leaf rule from the research note (section 5), plain Python with no mujoco,
  so the sim labeller, the sim judge and the real robot share one copy. It takes the stem's nodes from the
  bottom up and the grower's `Settings`, and returns the leaves to cut, oldest first, with a decision and a
  reason for every leaf: cut / keep / flag / guard / cap / unknown / scar / blocked. It never cuts an
  unknown, and it fills gaps in the crop's leaves-per-truss pattern with unknowns (`Settings.leaves_per_truss`:
  3 for high wire, 1 for a determinate dwarf). `confirm_cut` is the
  veto at the claw before each cut; until perception exists (step 4) the sim fills it from truth.
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
  meets the stem surface). Those are the labels for the vision model in step 4.

The checks cover these cases:
- the golden cases: hand-built stems with the cuts worked out by hand
- across 400 plants per profile: the profile's leaves-per-truss pattern, and every cut is a present leaf in the band, below the strip
  line, oldest first, within the pass cap and the keep guard
- the labels vary: plants with no cuts, with some cuts, with the full cap, with the guard or cap holding
  leaves back, with flags, young crops, missed leaves
- six dwarf and two high-wire plants built in MuJoCo: inside the geom budget with the arm, everything settles, nothing on the
  floor, every geom classed, and the label cut points sit where the settled petioles really are

⚠️ `min_leaves_kept = 16` is a placeholder from an open-source sim, not from growers. On many plants it
decides how many leaves come off, so ask growers before trusting the labels.

## The scene: mini greenhouse, plants, arm, cameras

    python check_layout.py --windowed            the whole scene (press [ and ] for the birdseye and claw views)
    python check_layout.py --picture out.png     overview, top view, birdseye frame, claw frame in one image
    python check_layout.py                       PASS/FAIL checks

- `greenhouse.py`: the laser-cut 6 mm ply frame (geometry from B.L.U.S.H, see NOTICE.md).
- `layout.py` (K7 approved 2026-10-28):
  - three dwarf cherry tomatoes in pots along row R1
  - the SO-101 on the aisle centreline, its base square to the row, in front of plant p0
  - the birdseye on a mast behind the empty row R0: level, looking straight at the row through R0, wide
    enough for the whole row. Not over the arm's shoulder like B.L.U.S.H.
  - the claw camera on the SO-101's stock wrist mount
- `lens.py`, `cameras.py`: the InnoMaker lens model and renderers, from B.L.U.S.H (camera hardware only).
- `ik.py`: position IK for the fingertips, used for the ready pose. The jaw's orientation is `cutpose.py`'s (step 3).

The checks cover these cases:
- the birdseye sits on its mast and never sees it; it sees every leaf's cut point and every plant top to bottom
- the arm's base stands outside the birdseye frame, square to the row, in front of its plant
- the claw camera looks along the jaw
- the plants settle inside the greenhouse
- IK reaches the ready pose

## Step 3: the cut across the ways the arm meets a leaf

    python check_approach.py                     PASS/FAIL checks, about 20 s (--table: one line per case)
    python check_approach.py --windowed --case az=-60,el=45      watch one case (keys in the docstring)
    python check_approach.py --reach             reach survey on the greenhouse plants, kinematic, ~6 min

- `cutpose.py`: the cut pose and the 5-joint IK that reaches it. The default cut is **flush**: the blade plane
  parallel to the stem, 2 mm off it, the way growers cut ("tot tegen de stam"). The jaw then closes along the
  stem, as in step 1: hinge level, fingers level. Twist (the blade turned about the stem, off the petiole's own
  direction) only slants the cut face, so it is traded first.
- `check_approach.py`: step 1's rig with the arm moved there by `cutpose.CutIK`. It covers leaf directions,
  blade lean, finger pitch, stub length, petiole size, and aim and sag errors. The stem is dwarf-size, the petiole
  now bends (`LeafParams.bend`), and the cut force scales with the petiole (`plant.cut_force_for`, unverified).
- `reach.py`: for every leaf on 20 random dwarf plants, whether the arm finds a clean flush pose clear of the
  stem, truss stalks, pot and frame. It tries today's base, the base slid along the aisle, and the far side of
  the row.

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
  - blade lean < 5 deg. This is not an artifact of the 25 deg jaw opening: at 0-25 deg openings, 10-20 deg of
    lean still puts the moving pad into the stem, and 5 deg cuts only at a 4 mm stub with the jaw open 5 deg or
    less (`--case mode=square,el=5,offset=4,jaw=5`). Step 5 should come in with the jaw barely open.
- ⚠️ **The SO-101's 5 joints decide which leaves it can cut.** Its fingers stay in the pan plane, so a flush cut
  fits a petiole that points along the row, within ±30 deg. Leaves pointing at or away from the arm would need
  the fingers straight up or down, which the arm can't reach in the work band.
- **Blades on both fingertip faces** (K7 2026-10-29). With one pair, a leaf pointing to the arm's right needed the
  jaw flipped over (wrist roll ~183 deg). The wrist has physical stops at ~±165 deg, so that leaf couldn't be cut
  flush. Now the other pair cuts it, with no flip. The cost is that both pairs share the squeeze (step 1 above).
  An 8 mm stub on a 3.6 mm petiole then doesn't part; 2-4 mm stubs do.
- **Reach on the dwarf row** (kinematic, both blade pairs, fruit may be pushed 15 mm, stalks may not):

  | where the base is | leaves | rule's cuts | one blade pair, for comparison |
  |---|---|---|---|
  | one base | 20 % | 23 % | 9 % / 13 % |
  | base sliding ±15 cm along the aisle | 65 % | 71 % | 31 % / 29 % |
  | sliding, plus the far side of the row | 71 % | 81 % | 58 % / 58 % |

  (Rerun 2026-11-01 after the truss fix in step 5; before it, both pairs read 16/16, 53/65, 66/71 %.)

  What's left: twist (petioles pointing at or away from the arm), then truss stalks in the way.
- The rig's leaflets pass through the hand: it tests the cut, not the approach. Approach paths, servo sag and
  the real leaf pushing are step 5's (the scripted cutter).

## Step 4 (started): cut-point labels for the vision model

    python check_labels.py                       PASS/FAIL checks on four rows, ~10 s
    python check_labels.py --picture out.png     each plant zoomed in, leaves numbered, one legend line per leaf
    python check_labels.py --windowed            the row with every cut point marked by the rule's decision
    python render_cutpoints.py --pilot 6         a PILOT set: 6 labelled frames into ~/.cache/shear_sim/cutpoints_pilot

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
- Keypoint visibility comes from a depth render, not the segmentation alone. ⚠️ A cut point behind the stem
  projects onto the stem's front; the segmentation calls it seen, but it's hidden.
- **Finding:** the birdseye sees only about a quarter of the rule's cut points (7 of 27 in the pilot). The low
  leaves are behind trusses or on the stem's far side, and parking the arm didn't change that. The birdseye can
  find the leaves; the claw camera has to find most cut points.
- Not done yet: claw-camera frames (from step 5's approach poses), domain randomisation, the dataset, training.
  `render_cutpoints.py` is capped at 24 frames until K7 approves the pilot.


## Step 5 (started): the scripted cutter, routes planned with RRT

    python check_cutter.py                       PASS/FAIL on one known plant, ~30 s
    python check_cutter.py --windowed            watch that cut in real time (it loops)
    python generate_cuts.py --trial 6            a TRIAL batch: 6 cut episodes as videos + summary,
                                                 into ~/.cache/shear_sim/cuts_trial (capped at 12)

- `rrt.py`: RRT-Connect in joint space, then shortcutting. K7 (2026-11-01): every arm move is planned with RRT.
  It plans against three lists:
  - **hard**: stems, pots, the frame. Never touched, 4 mm clearance.
  - **soft**: fruit, truss stalks, and the stalks and midribs of kept leaves. Avoided when a route exists; when
    none does, they may be brushed, and the episode says so. Leaflets just get pushed.
  - **blade guard**: the blades never come within 2 mm of a truss stalk. A blade can cut a truss off; a finger
    pad resting on it doesn't.
- `cutter.py`, one episode:
  1. hold at HOME (raised above the aisle), jaw opening 12 mm
  2. RRT route to the pre-cut pose, 4 cm back along the fingers
  3. straight slide onto the petiole
  4. close: the blades cut
  5. slide back out
  6. a new RRT route home

  IK (`cutpose.py`) gives the goal poses; RRT gives the routes. The servos follow a 30 Hz reference with an
  integral sag correction, and the arm lands within ~0.2 mm.

  The judge checks:
  - the right leaf came off, and no other one did
  - the stub is 1.5-3.5 mm
  - no stem, pot or frame was pressed
  - no truss stalk took more than 1 N
  - no blade touched a truss stalk
- `generate_cuts.py`: per episode, three random dwarf plants. The pot in front of the arm is turned so the
  oldest leaf the rule cuts points along the row, as a person turns a pot on the bench. The base stays put (K7: no
  sliding base, SPINE last).
- Found on the way (2026-11-01):
  - ⚠️ `stemgen` put every truss ~40 deg off the leaf below it instead of opposite it, so it drooped across the
    cut point. Fixed. That changed every plant.
  - ⚠️ The follower's shoulder goes to ±102 deg and the MJCF's to ±100. The IK now stays inside both.
  - ⚠️ A petiole rising across the fingertips meets the outer blade pair first when the jaw is flipped. The
    inner pair then cuts the rest as the jaw keeps closing (modelled in `plant.Cutter`, unverified).
  - From a fixed base, even with pots turned, only about 1 plant in 20 has a cut the arm can plan cleanly. The
    dwarf's trusses crowd the leaves.
- **One episode is a full deleafing pass** (K7 2026-11-03). It starts and ends at the robot's start pose
  (B.L.U.S.H's `START_NORM`), cuts the rule's leaves on p0, then p1, then p2, oldest first, each route planned
  from where the arm is, and skips any leaf without a clean cut.
  - Each pot is turned once beforehand, to the angle that lets the arm cut the most of that plant's leaves
    (`generate_cuts.best_turns`). Even at the best angle that is about 1 leaf per plant on the dense dwarf, and
    often none on p1 and p2 at the edge of the fixed arm's reach.
  - `generate_cuts.py` videos carry the sim's labels on the birdseye and claw views. That is what the YOLO model
    will learn; no model is trained yet.

## Episode generation: the LeRobot dataset

    python generate_episodes.py --pilot          5 episodes into krish5831/shear_sim_scripted_v0_pilot
    python generate_episodes.py --episodes 200   the real batch into krish5831/shear_sim_scripted_v0 (--resume)

Run under a real-memory fuse: `systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 python ...`.
⚠️ Not under `ulimit -v`: the lerobot import reserves so much address space that the EGL renderer then fails
("Offscreen framebuffer is not complete").

- Every frame is recorded the way `lerobot-record` does it, at 30 fps:
  - `observation.state`: the 6 joints in use_degrees=false units, through the follower's calibration
  - `action`: what the servos were told
  - `observation.images.birdseye` / `.claw`: clean 640x480 frames
  - `robot_type` `so_follower`, so sim and real SHEAR data can merge
- Only passes the judge accepts are kept. Every attempt is logged in `~/.cache/shear_sim/episodes/`.
- Writes locally only. Nothing goes to the Hub without K7's OK.
- Domain randomisation, on by default (`randomize.py`, adapted from B.L.U.S.H; K7 2026-11-03). It varies:
  - light brightness and direction, and the ambient light
  - greenhouse, floor and pot shades
  - leaf, stem and fruit colours, each inside its class, so a green leaf stays green and a ripe fruit red
  - small camera pose jitter

  The physics isn't randomised. `--no-dr` gives the plain look.
- Not done yet:
  - claw-camera labels for step 4 from these episodes
  - the task string is a draft: lock it before the real batch
