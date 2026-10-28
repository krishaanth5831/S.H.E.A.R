"""Random tomato plants for SHEAR sim step 2: one stem with leaves and trusses, and the rule's labels.

    spec = make_plant(seed)               the whole stem as rule nodes, plus geometry for the work band
    body, eq, where = plant_xml(spec)     MJCF for the band; where[node index] = the leaf's `where` dict
    sel = label(spec)                     rule.select_leaves on sim truth: which leaves the rule cuts
    seg_class(geom_name)                  segmentation class of a plant geom (stem, petiole, leaflet, ...)

The plant, bottom (oldest) up, like a high-wire crop:
- Three leaves, then a truss, repeating (research note, section 1). Trusses ripen from the bottom up:
  harvested, ripe, turning, green.
- Leaves below the last pass's strip line are flush scars. Some weeks of growth have passed since then, so
  0-4 leaves have aged past the line again. Now and then an earlier pass missed one.
- Old leaves turn yellow more often; a few are diseased. Now and then a yellow leaf sits above the line,
  for the rule to flag.
- Sometimes a young crop: nothing has turned yet.

Only the work band (BAND_Z) is built in MuJoCo. The stem above it is ABSTRACT: rule nodes with no geoms.
⚠️ They matter: the keep guard (min_leaves_kept) counts every present leaf on the stem, and the band alone
holds only a handful. Without the abstract leaves the guard blocks every cut and every label is empty.

Every size and probability here is a guess marked unverified until real plants are measured
(SHEAR Sim Plan, "Measure first"). The leaf scale (real ~0.45 m leaves vs the 0.52 m SO-101 test
greenhouse) is K7's call: --leaf-len and INTERNODE_M scale together.
"""
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

import plant as pl
import truss as tr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rule.select_leaves import (DISEASED, GREEN, LEAF, Node, PRESENT, SCAR, TRUSS, YELLOW,  # noqa: E402
                                Settings, select_leaves)

# --- the stem ---
STEM_XY = (0.0, 0.0)
STEM_Z = (0.02, 0.62)             # stem capsule in the scene, metres above the floor
BAND_Z = (0.20, 0.58)             # nodes built in MuJoCo; above it they are abstract. ⚠️ Low leaves hit the
                                  # floor below ~0.2 m with 0.3 m leaves (check_plant.py reports floor contact)
STEM_R = (0.005, 0.007)           # 10-14 mm stem in the band, guess (unverified)
# Leaf to leaf along the stem. ⚠️ Guess scaled to the 0.30 m sim leaf (unverified): measure real plants.
INTERNODE_M = (0.040, 0.050)
# Tomato leaves spiral around the stem; the golden angle with jitter is a guess (unverified).
PHYLLOTAXIS_DEG, PHYLLO_JITTER_DEG = 137.5, 15.0

# --- the crop's state ---
P_YOUNG = 0.12                    # share of plants with no turned truss yet (unverified, for label variety)
N_HARVESTED = (0, 1)              # harvested trusses at the bottom of the band (unverified)
N_RIPE = (1, 2)
N_TURNING = (0, 2)
NEW_BELOW_LINE = (0, 4)           # leaves aged past the strip line since the last pass (weekly passes, guess)
P_MISSED = 0.08                   # an earlier pass left a leaf below the line (Kompano finds ~85 %, research)
P_YELLOW_BELOW, P_DISEASED_BELOW = 0.25, 0.05   # leaves below the strip line (unverified)
P_YELLOW_ABOVE, P_DISEASED_ABOVE = 0.06, 0.02   # in the band above it: the rule flags these (unverified)
P_YELLOW_YOUNG = 0.4              # the lowest two leaves of a young crop (unverified)
# Present leaves on the stem above the band (abstract). ⚠️ unverified: with the placeholder
# min_leaves_kept = 16 this range makes the keep guard bite on some plants and not on others.
ABOVE_BAND_LEAVES = (8, 16)

# --- one leaf (plant.LeafParams), drawn per node ---
LEAF_LEN_SCALE = (0.85, 1.15)     # of --leaf-len (unverified)
LEAF_PAIRS = (3, 4)               # lateral leaflet pairs: "3-4" (misina-lab notes, research section 6)
LEAFLET_SCALE = (0.8, 1.25)       # of plant.LEAFLET_L (unverified)
LEAFLET_ASPECT = (0.45, 0.6)      # width / length (unverified, from photos)
PETIOLE_SCALE = (0.85, 1.15)      # of plant.PETIOLE_R (unverified)
ELEVATION_DEG = {GREEN: (-5.0, 20.0), YELLOW: (-12.0, 5.0), DISEASED: (-12.0, 5.0)}   # petiole above level
DROOP_DEG = {GREEN: (6.0, 10.0), YELLOW: (10.0, 15.0), DISEASED: (10.0, 15.0)}       # per midrib hinge
# Colours: petiole/midrib, leaflets (guesses from photos).
LEAF_COLOURS = {GREEN: (pl.LEAF_RGBA, pl.LEAFLET_RGBA),
                YELLOW: ("0.70 0.66 0.25 1", "0.80 0.74 0.28 1"),
                DISEASED: ("0.42 0.38 0.24 1", "0.40 0.33 0.20 1")}
SCAR_LEN = 0.0015                 # a flush scar stands this proud of the stem (unverified)

# --- one truss (truss.truss_xml) ---
TRUSS_FRUIT = (4, 7)              # ⚠️ desk scale; real cherry trusses are thinned to 8-12 (research)
FRUIT_PITCH_M = 0.024             # rachis length per fruit, ~ one fruit diameter (unverified)
TRUSS_DOWN_DEG = (20.0, 40.0)     # rachis below level (unverified)

# Soft geom budget for the whole model, arm included. ⚠️ Not a hard limit any more: B.L.U.S.H crashed past
# 256 geoms only while segmentation shared the MSAA renderer (fixed in its e91ef0c). It's here for speed.
GEOM_BUDGET = 512


@dataclass
class PlantSpec:
    seed: int
    nodes: list                       # rule Nodes, bottom up, the whole stem (band + abstract)
    z: list                           # node height on the stem (None for abstract nodes)
    azimuth: list                     # radians
    leaf: dict = field(default_factory=dict)    # node index -> (LeafParams, elevation rad)
    truss: dict = field(default_factory=dict)   # node index -> (n fruit, ripeness list, down rad)
    stem_r: float = 0.006
    young: bool = False

    def in_band(self, i):
        return self.z[i] is not None


def _ripeness(stage, n, rng):
    """Fruit colours along the rachis, from the stem out. The fruit nearest the stem ripens first."""
    if stage == "green":
        return ["green"] * n
    if stage == "turning":
        k = int(rng.integers(1, max(2, (n + 1) // 2) + 1))
        first = ["ripe" if rng.random() < 0.3 else "turning"] + ["turning"] * (k - 1)
        return first + ["green"] * (n - k)
    # ripe: the last one or two may still be turning
    k = int(rng.integers(0, 3))
    return ["ripe"] * (n - k) + ["turning"] * k


def make_plant(seed, leaf_len=pl.LEAF_LEN):
    rng = np.random.default_rng(seed)
    young = bool(rng.random() < P_YOUNG)
    if young:
        stages_low = []
    else:
        stages_low = (["harvested"] * int(rng.integers(N_HARVESTED[0], N_HARVESTED[1] + 1))
                      + ["ripe"] * int(rng.integers(N_RIPE[0], N_RIPE[1] + 1))
                      + ["turning"] * int(rng.integers(N_TURNING[0], N_TURNING[1] + 1)))
    phase = int(rng.integers(0, 3))           # leaves below the band's first truss

    # Walk up the stem: leaves at internode steps, a truss half an internode above every third leaf.
    nodes, z, az = [], [], []
    h, phi, n_leaf, n_truss = BAND_Z[0], rng.uniform(0, 2 * np.pi), 0, 0
    above_band = int(rng.integers(ABOVE_BAND_LEAVES[0], ABOVE_BAND_LEAVES[1] + 1))
    abstract_leaves = 0
    while abstract_leaves < above_band:
        if n_leaf >= phase and (n_leaf - phase) % 3 == 0 and (n_leaf - phase) // 3 == n_truss:
            stage = stages_low[n_truss] if n_truss < len(stages_low) else "green"
            zt = h - 0.5 * rng.uniform(*INTERNODE_M)
            band = zt <= BAND_Z[1]
            nodes.append(Node(TRUSS, stage=stage, id=f"n{len(nodes):02d}_truss"))
            z.append(zt if band else None)
            az.append(phi + np.pi + rng.uniform(-0.5, 0.5))      # roughly opposite the leaf below (unverified)
            n_truss += 1
            continue
        band = h <= BAND_Z[1]
        nodes.append(Node(LEAF, id=f"n{len(nodes):02d}_"))
        z.append(h if band else None)
        az.append(phi)
        abstract_leaves += not band
        n_leaf += 1
        h += rng.uniform(*INTERNODE_M)
        phi += np.radians(PHYLLOTAXIS_DEG + rng.uniform(-PHYLLO_JITTER_DEG, PHYLLO_JITTER_DEG))

    leaves = [i for i, n in enumerate(nodes) if n.kind == LEAF]
    trusses = [i for i, n in enumerate(nodes) if n.kind == TRUSS]
    turned = [i for i in trusses if nodes[i].stage in ("turning", "ripe")]
    if turned:
        # Ordinal (among leaves) of the strip line: every leaf below the anchor, plus the one above it.
        line = sum(i < turned[0] for i in leaves)
        scar_below = max(0, line + 1 - int(rng.integers(NEW_BELOW_LINE[0], NEW_BELOW_LINE[1] + 1)))
    else:
        line, scar_below = -1, 0
    for k, i in enumerate(leaves):
        n = nodes[i]
        if k < scar_below and rng.random() >= P_MISSED:
            n.state = SCAR
            continue
        n.state = PRESENT
        if young:
            p_y, p_d = (P_YELLOW_YOUNG, 0.0) if k < 2 else (0.0, 0.0)
        elif k <= line:
            p_y, p_d = P_YELLOW_BELOW, P_DISEASED_BELOW
        elif z[i] is not None:
            p_y, p_d = P_YELLOW_ABOVE, P_DISEASED_ABOVE
        else:
            p_y, p_d = 0.0, 0.0                    # abstract leaves near the head: green
        u = rng.random()
        n.health = DISEASED if u < p_d else YELLOW if u < p_d + p_y else GREEN

    spec = PlantSpec(seed=seed, nodes=nodes, z=z, azimuth=az, stem_r=float(rng.uniform(*STEM_R)), young=young)
    for i in leaves:
        n = nodes[i]
        if z[i] is None or n.state != PRESENT:
            continue
        L = leaf_len * rng.uniform(*LEAF_LEN_SCALE)
        ll = pl.LEAFLET_L * rng.uniform(*LEAFLET_SCALE)
        mid, leaflet = LEAF_COLOURS[n.health]
        P = pl.LeafParams(length=L, mass=pl.LEAF_MASS * (L / pl.LEAF_LEN) ** 2 * rng.uniform(0.85, 1.15),
                          n_seg=int(rng.integers(LEAF_PAIRS[0], LEAF_PAIRS[1] + 1)),
                          petiole_r=pl.PETIOLE_R * rng.uniform(*PETIOLE_SCALE),
                          petiole_len=pl.PETIOLE_LEN * L / pl.LEAF_LEN,
                          leaflet_l=ll, leaflet_w=ll * rng.uniform(*LEAFLET_ASPECT),
                          droop_deg=rng.uniform(*DROOP_DEG[n.health]), rgba=mid, leaflet_rgba=leaflet)
        spec.leaf[i] = (P, np.radians(rng.uniform(*ELEVATION_DEG[n.health])))
    for i in trusses:
        if z[i] is None:
            continue
        stage = nodes[i].stage
        nf = 0 if stage == "harvested" else int(rng.integers(TRUSS_FRUIT[0], TRUSS_FRUIT[1] + 1))
        spec.truss[i] = (nf, _ripeness(stage, nf, rng), np.radians(rng.uniform(*TRUSS_DOWN_DEG)))
    return spec


def _dir(phi, elev):
    return np.array([np.cos(elev) * np.cos(phi), np.cos(elev) * np.sin(phi), np.sin(elev)])


def plant_xml(spec, xy=STEM_XY):
    """MJCF for the band. Returns (worldbody xml, equality xml, where) with where[i] = plant.leaf_xml's
    dict for each present band leaf (base = the point where its petiole meets the stem surface)."""
    rng = np.random.default_rng(spec.seed + 7_000_000)    # truss_xml's own draws, kept off make_plant's stream
    stem_pos = np.array([xy[0], xy[1], 0.5 * (STEM_Z[0] + STEM_Z[1])])
    bodies, eqs, inner, where = [], [], [], {}
    for i, n in enumerate(spec.nodes):
        if not spec.in_band(i):
            continue
        S = np.array([xy[0], xy[1], spec.z[i]])
        if n.kind == LEAF and n.state == SCAR:
            a = _dir(spec.azimuth[i], 0.0)
            c = S + a * (spec.stem_r + SCAR_LEN / 2) - stem_pos
            R = pl.leaf_frame(a)
            Rs = np.column_stack([np.cross(a, R[:, 1]), R[:, 1], a])
            inner.append(f'<geom name="{n.id}scar" type="cylinder" size="{pl.PETIOLE_R:.5f} {SCAR_LEN / 2:.5f}" '
                         f'pos="{pl._f(c)}" quat="{pl._f(pl._quat(Rs))}" contype="0" conaffinity="0" group="2" '
                         f'rgba="{pl.STUB_RGBA}"/>')
        elif n.kind == LEAF:
            P, elev = spec.leaf[i]
            leaf, stub, eq, w = pl.leaf_xml(n.id, S, _dir(spec.azimuth[i], elev), P, stem_pos, spec.stem_r)
            bodies.append(leaf); inner.append(stub); eqs.append(eq)
            w["node"] = i
            where[i] = w
            n.point = tuple(w["base"])
        else:
            nf, rip, down = spec.truss[i]
            d = _dir(spec.azimuth[i], -down)
            start = S + d * spec.stem_r
            end = start + d * max(1, nf) * FRUIT_PITCH_M
            rachis, fruit, eq, _ = tr.truss_xml(n.id, start, end, nf, rng, ripeness=rip)
            bodies += [rachis, fruit]; eqs.append(eq)
    stem, _ = pl.stem_xml(xy, STEM_Z, spec.stem_r, "".join(inner))
    return stem + "".join(bodies), "".join(eqs), where


def label(spec, settings=None, conditions=None):
    """The rule on sim truth. ⚠️ Run plant_xml first if the cut points (Node.point) are needed."""
    return select_leaves(spec.nodes, settings or Settings(), conditions)


# Segmentation classes, from the geom names plant_xml writes. Order matters: first match wins.
SEG_CLASSES = (("stem", lambda g: g == "stem"), ("scar", lambda g: g.endswith("scar")),
               ("stub", lambda g: g.endswith("stub")), ("petiole", lambda g: g.endswith("petiole")),
               ("midrib", lambda g: "midrib" in g), ("leaflet", lambda g: "leaflet" in g),
               ("rachis", lambda g: g.endswith("_rachis")), ("calyx", lambda g: "_calyx" in g),
               ("pedicel", lambda g: g.endswith("_ped")), ("fruit", lambda g: g.endswith("_g")))


def seg_class(name):
    for cls, match in SEG_CLASSES:
        if match(name or ""):
            return cls
    return None


def node_of(name):
    """Node index a plant geom belongs to (its 'nNN_' prefix), or None for the stem."""
    return int(name[1:3]) if name and name[0] == "n" and name[1:3].isdigit() else None
