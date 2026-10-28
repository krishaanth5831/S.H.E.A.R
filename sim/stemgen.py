"""Random tomato plants for the SHEAR sim: one stem with leaves and trusses, and the rule's labels.

    spec = make_plant(seed)                      a dwarf cherry tomato (the default profile)
    spec = make_plant(seed, profile="highwire")  a high-wire plant's work band
    body, eq, where = plant_xml(spec)            MJCF; where[node index] = the leaf's `where` dict
    sel = label(spec)                            rule.select_leaves on sim truth, with the profile's settings
    seg_class(geom_name)                         segmentation class of a plant geom (stem, petiole, leaflet, ...)

Two profiles (PROFILES):
- **dwarf_cherry** (default since 2026-10-28). K7: real-size tomato leaves are far too big and crowded for
  the 0.52 m mini greenhouse. A dwarf, determinate cherry tomato like 'Tiny Tim' is a real plant at its real
  size that fits: 20-45 cm tall, ~2 cm fruit (UNH 1945 cultivar; West Coast Seeds, Epic Gardening listings).
  It stands in a pot in the row and is built whole: every node has geometry. Determinate: the first truss
  after 5-7 leaves, then one every 1-2 leaves, ending in a truss (the self-pruning habit; general
  horticulture, unverified for this cultivar). ⚠️ No source gave its leaf sizes: every leaf number below is
  a guess until a bought plant is measured.
- **highwire**: the commercial crop, three leaves then a truss, endless. Only the work band (band_z) is built;
  the stem above it is ABSTRACT: rule nodes with no geoms, which the keep guard counts. Too big for the mini
  greenhouse (0.30 m leaves are already scaled down from ~0.45 m), kept for the real greenhouse.

Both, bottom (oldest) up:
- Trusses ripen from the bottom up: harvested, ripe, turning, green.
- Leaves below the last pass's strip line are flush scars. Since then 0-4 leaves have aged past the line
  again. Now and then an earlier pass missed one.
- Old leaves turn yellow more often; a few are diseased. Now and then a yellow leaf sits above the line,
  for the rule to flag.
- Sometimes a young crop: nothing has turned yet.

⚠️ The rule's grower settings for the dwarf (Profile.settings) are placeholders too: no grower deleafs a
patio tomato. The dwarf is a stand-in that shows the rule and the cut at a scale the SO-101 greenhouse fits.
"""
import pathlib
import re
import sys
from dataclasses import dataclass, field

import numpy as np

import plant as pl
import truss as tr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rule.select_leaves import (DISEASED, GREEN, LEAF, Node, PRESENT, SCAR, TRUSS, YELLOW,  # noqa: E402
                                Settings, select_leaves)

STEM_XY = (0.0, 0.0)
GROUND_Z = 0.036          # top of a greenhouse row's anchor rail (greenhouse.ANCHOR_TOP): pots and stems start here
# Tomato leaves spiral around the stem; the golden angle with jitter is a guess (unverified).
PHYLLOTAXIS_DEG, PHYLLO_JITTER_DEG = 137.5, 15.0
SCAR_LEN = 0.0015         # a flush scar stands this proud of the stem (unverified)
# A tomato petiole leaves the stem angled up and the rachis arches over and down (photos, unverified).
# Old and yellow leaves sit flatter and hang more.
ELEVATION_DEG = {GREEN: (25.0, 45.0), YELLOW: (5.0, 20.0), DISEASED: (5.0, 20.0)}    # petiole above level
# Per midrib hinge. ⚠️ 16-24 deg on yellow high-wire leaves bent them straight down, onto the floor (seed 3).
DROOP_DEG = {GREEN: (8.0, 13.0), YELLOW: (12.0, 17.0), DISEASED: (12.0, 17.0)}
# Colours: petiole/midrib, leaflets (guesses from photos).
LEAF_COLOURS = {GREEN: ("0.36 0.55 0.22 1", "0.18 0.42 0.13 1"),
                YELLOW: ("0.70 0.66 0.25 1", "0.80 0.74 0.28 1"),
                DISEASED: ("0.42 0.38 0.24 1", "0.40 0.33 0.20 1")}
POT_RGBA = "0.72 0.38 0.22 1"     # terracotta
TRUSS_DOWN_DEG = (20.0, 40.0)     # rachis below level (unverified)
P_YOUNG = 0.12                    # share of plants with no turned truss yet (unverified, for label variety)
P_MISSED = 0.08                   # an earlier pass left a leaf below the line (Kompano finds ~85 %, research)
P_YELLOW_BELOW, P_DISEASED_BELOW = 0.25, 0.05   # leaves below the strip line (unverified)
P_YELLOW_ABOVE, P_DISEASED_ABOVE = 0.06, 0.02   # built leaves above it: the rule flags these (unverified)
P_YELLOW_YOUNG = 0.4              # the lowest two leaves of a young crop (unverified)
LEAF_MASS_REF = (pl.LEAF_LEN, pl.LEAF_MASS)     # mass scales with leaf area from step 1's 0.30 m / 25 g leaf

# Soft geom budget for the whole scene (3 plants, greenhouse, arm). ⚠️ Not a hard limit: B.L.U.S.H crashed past
# 256 geoms only while segmentation shared the MSAA renderer (fixed in its e91ef0c). It's here for speed; most
# plant geoms are calyx sepals and leaflets.
GEOM_BUDGET = 1024


@dataclass(frozen=True)
class Profile:
    name: str
    stem_r: tuple                 # stem radius
    internode: tuple              # leaf to leaf along the stem
    band_z: tuple                 # nodes built in MuJoCo (highwire: the band; dwarf: from the lowest leaf up)
    first_truss_leaves: tuple     # leaves below the first truss in the built part
    leaves_between: tuple         # leaves between two trusses
    above_band_leaves: tuple      # abstract present leaves above the band (None: the plant is built whole)
    leaf_len: tuple               # stem surface to leaf tip
    leaf_pairs: tuple             # lateral leaflet pairs
    leaflet_l: tuple
    leaflet_aspect: tuple         # width / length
    petiole_r: tuple
    petiole_frac: float           # petiole length / leaf length
    mid_r: float
    n_ripe: tuple
    n_turning: tuple
    p_harvested: float            # chance the lowest built truss is already harvested
    new_below_line: tuple         # leaves aged past the strip line since the last pass
    truss_fruit: tuple
    fruit_radius: tuple
    fruit_pitch: float            # rachis length per fruit
    settings: Settings            # the rule's settings for this crop
    pot: tuple = None             # (radius, height) of the pot it stands in, or None (hangs from the crop rail)
    plant_height: tuple = None    # dwarf: stem top above the pot rim
    stem_top: float = None        # highwire: the stem's top (the crop rail)


PROFILES = {
    "dwarf_cherry": Profile(
        name="dwarf_cherry",
        stem_r=(0.003, 0.0045),          # 6-9 mm (unverified)
        internode=(0.020, 0.028),        # compact: the dwarf habit is short internodes (Micro-Tom: Plant Methods 7:18)
        band_z=(0.0, 1.0),               # whole plant
        first_truss_leaves=(5, 7),       # determinate: first truss after 5-7 leaves (unverified)
        leaves_between=(1, 2),           # self-pruning: a truss every 1-2 leaves (unverified)
        above_band_leaves=None,
        leaf_len=(0.08, 0.13),           # ⚠️ guess, no source found: measure a 'Tiny Tim'
        leaf_pairs=(2, 3),               # "leaves with five leaflets" (Micro-Tom, Plant Methods 7:18); 2-3 pairs
        # ⚠️ guess (unverified). 22-34 mm made the leaves look skeletal next to the bushy 'Tiny Tim' in photos.
        leaflet_l=(0.032, 0.045),
        leaflet_aspect=(0.55, 0.70),
        petiole_r=(0.0012, 0.0018),      # 2.4-3.6 mm petiole (unverified): the cut force follows it
        petiole_frac=0.18,
        mid_r=0.0012,
        n_ripe=(1, 2), n_turning=(0, 2), p_harvested=0.25, new_below_line=(0, 3),
        truss_fruit=(3, 5),              # small trusses (unverified)
        fruit_radius=(0.008, 0.011),     # ~2 cm fruit, 3/4-1 inch (West Coast Seeds, Epic Gardening listings)
        fruit_pitch=0.019,
        # ⚠️ placeholders: no grower deleafs a dwarf. Keep 4 so a young plant keeps its leaves (unverified).
        settings=Settings(min_leaves_kept=4, leaves_per_truss=1),
        pot=(0.045, 0.07),               # a 9 cm nursery pot, 7 cm tall (unverified)
        plant_height=(0.26, 0.36),       # above the pot rim; 'Tiny Tim' is 20-45 cm overall (listings above)
    ),
    "highwire": Profile(
        name="highwire",
        stem_r=(0.005, 0.007),           # 10-14 mm stem in the band (unverified)
        # ⚠️ 0.040-0.050 left 2-3 leaves per plant in the band (K7: "doesn't look like tomato plants")
        internode=(0.035, 0.042),
        # ⚠️ low leaves hit the floor below ~0.2 m with 0.3 m leaves, and up-angled young leaves hit the crop
        # rail (z 0.52) from nodes above ~0.47 (check_plant.py / check_layout.py)
        band_z=(0.20, 0.47),
        first_truss_leaves=(0, 2),
        leaves_between=(3, 3),           # three leaves, then a truss (Development 125:1979, research)
        # ⚠️ unverified: with min_leaves_kept = 16 this range makes the keep guard bite on some plants only
        above_band_leaves=(8, 16),
        leaf_len=(0.255, 0.345),         # 0.30 m +-15 %, scaled from real ~0.45 m leaves (misina-lab)
        leaf_pairs=(3, 4),               # "3-4" lateral pairs (misina-lab notes, research section 6)
        leaflet_l=(0.060, 0.081),        # unverified; tomato leaflets fill most of the leaf (photos)
        leaflet_aspect=(0.50, 0.62),
        petiole_r=(0.0030, 0.0040),      # 6-8 mm (unverified)
        petiole_frac=pl.PETIOLE_LEN / pl.LEAF_LEN,
        mid_r=pl.MID_R,
        # A harvested truss at the bottom of the band. ⚠️ always 0-1 put the anchor so high the band was
        # mostly scars (seeds 11-13, 2026-10-28). In high wire the harvested part is mostly the lowered bare stem.
        n_ripe=(1, 2), n_turning=(0, 2), p_harvested=0.25, new_below_line=(0, 4),
        truss_fruit=(4, 7),              # ⚠️ desk scale; real cherry trusses are thinned to 8-12 (research)
        fruit_radius=tr.RADIUS_M,
        fruit_pitch=0.024,
        settings=Settings(),             # min_leaves_kept 16 placeholder, 3 leaves per truss
        stem_top=0.52,                   # the crop rail it hangs from
    ),
}
DEFAULT_PROFILE = "dwarf_cherry"


@dataclass
class PlantSpec:
    seed: int
    profile: Profile
    nodes: list                       # rule Nodes, bottom up, the whole stem (built + abstract)
    z: list                           # node height on the stem (None for abstract nodes)
    azimuth: list                     # radians
    stem_z: tuple = (GROUND_Z, 0.52)
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


def _ri(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def make_plant(seed, profile=DEFAULT_PROFILE, leaf_scale=1.0):
    """leaf_scale multiplies every leaf length (the open leaf-scale question; --leaf-scale)."""
    P = PROFILES[profile] if isinstance(profile, str) else profile
    rng = np.random.default_rng(seed)
    young = bool(rng.random() < P_YOUNG)
    stages_low = [] if young else (["harvested"] * int(rng.random() < P.p_harvested) + ["ripe"] * _ri(rng, P.n_ripe)
                                   + ["turning"] * _ri(rng, P.n_turning))
    if P.pot:
        z0 = GROUND_Z + P.pot[1]
        stem_z = (z0 - 0.01, z0 + rng.uniform(*P.plant_height))
        h = z0 + rng.uniform(0.02, 0.04)                 # lowest leaf above the rim (unverified)
        top = stem_z[1] - 0.01
    else:
        stem_z, h, top = (GROUND_Z, P.stem_top), P.band_z[0], P.band_z[1]

    # Walk up the stem: leaves at internode steps, a truss half an internode above a leaf when it's due.
    nodes, z, az = [], [], []
    phi = rng.uniform(0, 2 * np.pi)
    due = _ri(rng, P.first_truss_leaves)                 # leaves until the next truss
    n_truss, abstract_leaves = 0, 0
    above = _ri(rng, P.above_band_leaves) if P.above_band_leaves else 0
    while True:
        if P.above_band_leaves is None and h > top:
            break
        if P.above_band_leaves is not None and abstract_leaves >= above:
            break
        if due == 0:
            stage = stages_low[n_truss] if n_truss < len(stages_low) else "green"
            zt = h - 0.5 * rng.uniform(*P.internode)
            built = zt <= top
            nodes.append(Node(TRUSS, stage=stage, id=f"n{len(nodes):02d}_truss"))
            z.append(zt if built else None)
            az.append(phi + np.pi + rng.uniform(-0.5, 0.5))      # roughly opposite the leaf below (unverified)
            n_truss += 1
            due = _ri(rng, P.leaves_between)
            continue
        built = h <= top
        nodes.append(Node(LEAF, id=f"n{len(nodes):02d}_"))
        z.append(h if built else None)
        az.append(phi)
        abstract_leaves += not built
        due -= 1
        h += rng.uniform(*P.internode)
        phi += np.radians(PHYLLOTAXIS_DEG + rng.uniform(-PHYLLO_JITTER_DEG, PHYLLO_JITTER_DEG))
    if P.pot and nodes and nodes[-1].kind == LEAF and z[-1] is not None and z[-1] + 0.01 < stem_z[1]:
        # Determinate: the stem ends in a truss (the self-pruning habit).
        nodes.append(Node(TRUSS, stage="green", id=f"n{len(nodes):02d}_truss"))
        z.append(z[-1] + 0.5 * (stem_z[1] - z[-1]))
        az.append(phi + np.pi)

    leaves = [i for i, n in enumerate(nodes) if n.kind == LEAF]
    trusses = [i for i, n in enumerate(nodes) if n.kind == TRUSS]
    turned = [i for i in trusses if nodes[i].stage in ("turning", "ripe")]
    if turned:
        # Ordinal (among leaves) of the strip line: every leaf below the anchor, plus the one above it.
        line = sum(i < turned[0] for i in leaves)
        scar_below = max(0, line + 1 - _ri(rng, P.new_below_line))
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

    spec = PlantSpec(seed=seed, profile=P, nodes=nodes, z=z, azimuth=az, stem_z=stem_z,
                     stem_r=float(rng.uniform(*P.stem_r)), young=young)
    for i in leaves:
        n = nodes[i]
        if z[i] is None or n.state != PRESENT:
            continue
        L = rng.uniform(*P.leaf_len) * leaf_scale
        ll = rng.uniform(*P.leaflet_l) * leaf_scale
        mid, leaflet = LEAF_COLOURS[n.health]
        lp = pl.LeafParams(length=L, mass=LEAF_MASS_REF[1] * (L / LEAF_MASS_REF[0]) ** 2 * rng.uniform(0.85, 1.15),
                           n_seg=_ri(rng, P.leaf_pairs), petiole_r=rng.uniform(*P.petiole_r),
                           petiole_len=P.petiole_frac * L, leaflet_l=ll, leaflet_w=ll * rng.uniform(*P.leaflet_aspect),
                           droop_deg=rng.uniform(*DROOP_DEG[n.health]), rgba=mid, leaflet_rgba=leaflet,
                           style="tomato", mid_r=P.mid_r)
        spec.leaf[i] = (lp, np.radians(rng.uniform(*ELEVATION_DEG[n.health])))
    for i in trusses:
        if z[i] is None:
            continue
        stage = nodes[i].stage
        nf = 0 if stage == "harvested" else _ri(rng, P.truss_fruit)
        spec.truss[i] = (nf, _ripeness(stage, nf, rng), np.radians(rng.uniform(*TRUSS_DOWN_DEG)))
    return spec


def _dir(phi, elev):
    return np.array([np.cos(elev) * np.cos(phi), np.cos(elev) * np.sin(phi), np.sin(elev)])


def plant_xml(spec, xy=STEM_XY, pp=""):
    """MJCF for the built part. Returns (worldbody xml, equality xml, where) with where[i] = plant.leaf_xml's
    dict for each present built leaf (base = the point where its petiole meets the stem surface).
    pp: prefix for every name, so several plants fit in one scene ("p1_" -> p1_stem, p1_n03_leaf, ...)."""
    P = spec.profile
    rng = np.random.default_rng(spec.seed + 7_000_000)    # truss_xml's own draws, kept off make_plant's stream
    stem_pos = np.array([xy[0], xy[1], 0.5 * (spec.stem_z[0] + spec.stem_z[1])])
    bodies, eqs, inner, where = [], [], [], {}
    for i, n in enumerate(spec.nodes):
        if not spec.in_band(i):
            continue
        S = np.array([xy[0], xy[1], spec.z[i]])
        nid = pp + n.id
        if n.kind == LEAF and n.state == SCAR:
            a = _dir(spec.azimuth[i], 0.0)
            c = S + a * (spec.stem_r + SCAR_LEN / 2) - stem_pos
            R = pl.leaf_frame(a)
            Rs = np.column_stack([np.cross(a, R[:, 1]), R[:, 1], a])
            inner.append(f'<geom name="{nid}scar" type="cylinder" size="{np.mean(P.petiole_r):.5f} {SCAR_LEN / 2:.5f}" '
                         f'pos="{pl._f(c)}" quat="{pl._f(pl._quat(Rs))}" contype="0" conaffinity="0" group="2" '
                         f'rgba="{pl.STUB_RGBA}"/>')
        elif n.kind == LEAF:
            lp, elev = spec.leaf[i]
            leaf, stub, eq, w = pl.leaf_xml(nid, S, _dir(spec.azimuth[i], elev), lp, stem_pos, spec.stem_r, pp + "stem")
            bodies.append(leaf); inner.append(stub); eqs.append(eq)
            w["node"] = i
            where[i] = w
            n.point = tuple(w["base"])
        else:
            nf, rip, down = spec.truss[i]
            d = _dir(spec.azimuth[i], -down)
            start = S + d * spec.stem_r
            end = start + d * max(1, nf) * P.fruit_pitch
            rachis, fruit, eq, _ = tr.truss_xml(nid, start, end, nf, rng, ripeness=rip, radius=P.fruit_radius)
            bodies += [rachis, fruit]; eqs.append(eq)
    stem, _ = pl.stem_xml(xy, spec.stem_z, spec.stem_r, "".join(inner), pp + "stem")
    pot = ""
    if P.pot:
        r, hgt = P.pot
        pot = (f'<geom name="{pp}pot" type="cylinder" size="{r} {hgt / 2}" pos="{xy[0]} {xy[1]} {GROUND_Z + hgt / 2:.4f}" '
               f'rgba="{POT_RGBA}" contype="{pl.FLOOR_BITS[0]}" conaffinity="{pl.FLOOR_BITS[1]}"/>')
    return pot + stem + "".join(bodies), "".join(eqs), where


def label(spec, settings=None, conditions=None):
    """The rule on sim truth, with the profile's settings unless given. ⚠️ Run plant_xml first if the cut
    points (Node.point) are needed."""
    return select_leaves(spec.nodes, settings or spec.profile.settings, conditions)


# Segmentation classes, from the geom names plant_xml writes. Order matters: first match wins.
SEG_CLASSES = (("stem", lambda g: g.endswith("stem")), ("pot", lambda g: g.endswith("pot")),
               ("scar", lambda g: g.endswith("scar")),
               ("stub", lambda g: g.endswith("stub")), ("petiole", lambda g: g.endswith("petiole")),
               ("midrib", lambda g: "midrib" in g), ("leaflet", lambda g: "leaflet" in g),
               ("rachis", lambda g: g.endswith("_rachis")), ("calyx", lambda g: "_calyx" in g),
               ("pedicel", lambda g: g.endswith("_ped")), ("fruit", lambda g: g.endswith("_g")))
NODELESS = ("stem", "pot")        # plant geoms that belong to no node


def seg_class(name):
    for cls, match in SEG_CLASSES:
        if match(name or ""):
            return cls
    return None


_NODE = re.compile(r"^(?:p\d+_)?n(\d\d)_")
_PLANT = re.compile(r"^p(\d+)_")


def node_of(name):
    """Node index a plant geom belongs to (its 'nNN_' part), or None for the stem and pot."""
    k = _NODE.match(name or "")
    return int(k.group(1)) if k else None


def plant_of(name):
    """Which plant ('pK_' prefix) a geom belongs to; 0 for a single unprefixed plant, None if not a plant geom."""
    if seg_class(name) is None:
        return None
    k = _PLANT.match(name)
    return int(k.group(1)) if k else 0
