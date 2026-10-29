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
  | one base | 16 % | 16 % | 9 % / 13 % |
  | base sliding ±15 cm along the aisle | 53 % | 65 % | 31 % / 29 % |
  | sliding, plus the far side of the row | 66 % | 71 % | 58 % / 58 % |

  What's left: twist (petioles pointing at or away from the arm), then truss stalks in the way.
- The rig's leaflets pass through the hand: it tests the cut, not the approach. Approach paths, servo sag and
  the real leaf pushing are step 5's (the scripted cutter).

## Step 4 (started): cut-point labels for the vision model

    python check_labels.py                       PASS/FAIL checks on four rows, ~10 s
    python check_labels.py --picture out.png     birdseye frames with the labels drawn on
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
- Keypoint visibility comes from a depth render, not the segmentation alone. ⚠️ A cut point behind the stem
  projects onto the stem's front; the segmentation calls it seen, but it's hidden.
- **Finding:** the birdseye sees only about a quarter of the rule's cut points (7 of 27 in the pilot). The low
  leaves are behind trusses or on the stem's far side, and parking the arm didn't change that. The birdseye can
  find the leaves; the claw camera has to find most cut points.
- Not done yet: claw-camera frames (from step 5's approach poses), domain randomisation, the dataset, training.
  `render_cutpoints.py` is capped at 24 frames until K7 approves the pilot.

