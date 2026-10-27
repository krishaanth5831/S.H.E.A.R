# S.H.E.A.R (Selective Harvest of Excess And Redundant leaves)

Vinea's deleafing and pruning module, built on an SO-101 arm. The arm cuts old tomato
leaves off flush at the stem, using the stock rigid jaws with a blade at the tips. A fixed
rule picks the leaves. The grower sets that rule's parameters (how far up to strip, how many
leaves to keep, how many cuts per pass). The learned part is the cut itself.

## Status

Nothing is built yet. The first step is the MuJoCo sim: one stem that collides, and one leaf
whose petiole the blade can cut. The plan and the research behind the rule are in the Vinea
vault, under `08 Modules/SHEAR/`.

## Git

Work happens on `dev`. `main` is protected and only takes PRs from `dev`; the
`only-dev-into-main` check enforces that.
