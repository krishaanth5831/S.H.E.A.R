# CLAUDE.md — S.H.E.A.R

Vinea deleafing and pruning module: the SO-101 cuts tomato leaves flush at the stem, with a
blade at the tips of the stock rigid jaws. The leaves are picked by a fixed rule with grower
settings. Plan and research: Vinea vault `08 Modules/SHEAR/`.

## Git workflow (hard rules)

- Commit and push straight to `dev`. **No feature branches**, not even for a PR. Never push
  to `main`. It's protected (PR required, enforce_admins on, `only-dev-into-main` required).
- Every PR is `dev` → `main`, and only when asked: `gh pr create --base main --head dev`.
  Always pass `--base` explicitly.
- **Never merge a PR**, to any branch. The user merges.

## Running heavy jobs

Sim, rendering and training runs go under a 9 GB memory cap (the user's choice, 2026-10-27):
`systemd-run --user --scope -q -p MemoryMax=9G -p MemorySwapMax=0 <cmd>`.

## Code style

Every constant carries a comment with where its value came from: measured, copied from a
source, or a guess marked `unverified`. Use ⚠️ on traps. Code copied from B.L.U.S.H names
the B.L.U.S.H commit it came from. Every step gets a check that prints PASS/FAIL and a
`--windowed` view to watch.

## Sim

The lerobot plugin type is `so101_mujoco_shear`, because B.L.U.S.H owns `so101_mujoco` in
the shared `lerobot` env. It reports `robot_type=so_follower`, so sim and real data can merge.

Hugging Face user is `krish5831`. GitHub user is `krishaanth5831`.
