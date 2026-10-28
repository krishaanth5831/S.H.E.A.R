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

All plant sizes and the cut force are guesses marked `unverified` in the code, until measured on real
plants.

## Step 2: random plants, and the leaf rule's labels

    python check_plant.py --windowed --seed 3    watch one plant settle. Red = leaves the rule cuts, orange =
                                                 flagged for a person, blue = held back (keep guard, pass cap).
                                                 --truth for the plant's own colours. The terminal prints the stem.
    python check_plant.py                        PASS/FAIL checks, about 10 s
    python ../rule/check_rule.py                 just the rule's golden cases (no mujoco)

- `../rule/select_leaves.py`: the leaf rule from the research note (section 5), plain Python with no mujoco,
  so the sim labeller, the sim judge and the real robot share one copy. It takes the stem's nodes from the
  bottom up and the grower's `Settings`, and returns the leaves to cut, oldest first, with a decision and a
  reason for every leaf: cut / keep / flag / guard / cap / unknown / scar / blocked. It never cuts an
  unknown, and it fills gaps in the three-leaves-per-truss pattern with unknowns. `confirm_cut` is the
  veto at the claw before each cut; until perception exists (step 4) the sim fills it from truth.
- `stemgen.py`: random plants. Three leaves then a truss, repeating. Trusses ripen from the bottom up:
  harvested, ripe, turning, green. Leaves below the last pass are flush scars, 0–4 have aged past the
  line since then, and now and then an earlier pass missed one. Old leaves are often yellow, a few are
  diseased, and sometimes it's a young crop with nothing turned yet. Every leaf differs: length, 3–4
  leaflet pairs, leaflet size, petiole thickness, angle, droop, colour, and where it sits around the stem.
  Only the work band is built in MuJoCo. The stem above it is abstract: rule nodes with no geoms, which
  the keep guard counts.
- `truss.py`: the trusses, copied from B.L.U.S.H `tomato.py` (see NOTICE.md for what changed).
- Every plant geom has a segmentation class (`stemgen.seg_class`: stem, petiole, midrib, leaflet, scar,
  rachis, pedicel, calyx, fruit) and a node, and every present leaf has its cut point (where the petiole
  meets the stem surface). Those are the labels for the vision model in step 4.

The checks cover these cases:
- the golden cases: hand-built stems with the cuts worked out by hand
- across 400 plants: three leaves per truss, and every cut is a present leaf in the band, below the strip
  line, oldest first, within the pass cap and the keep guard
- the labels vary: plants with no cuts, with some cuts, with the full cap, with the guard or cap holding
  leaves back, with flags, young crops, missed leaves
- six plants built in MuJoCo: inside the geom budget with the arm, everything settles, nothing on the
  floor, every geom classed, and the label cut points sit where the settled petioles really are

⚠️ `min_leaves_kept = 16` is a placeholder from an open-source sim, not from growers. On many plants it
decides how many leaves come off, so ask growers before trusting the labels.
