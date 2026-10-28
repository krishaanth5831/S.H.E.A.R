"""Cherry-tomato trusses for the SHEAR plant, copied from B.L.U.S.H sim/tomato.py (commit e91ef0c).

    rachis, fruit_bodies, eq, info = truss_xml(name, rachis_from, rachis_to, n, rng, ripeness)

Changed from B.L.U.S.H:
- ⚠️ Collision bits. B.L.U.S.H's fruit kept MuJoCo's default contype/conaffinity 1/1, and in SHEAR bit 1
  is the robot (plant.B_ROBOT), so fruit would have collided with the stem and every leaflet. Fruit and
  rachis now use plant.PLANT_BITS: they touch the robot and the floor, nothing else in the plant.
- The rachis collides (it was visual only), so step 3 can measure blade contact on truss stalks.
- Rachis, pedicel and calyx geoms have names, for segmentation labels.
- No Detacher and no break force: SHEAR doesn't pick. The tether weld stays on.
- The ripeness per fruit comes from the truss stage (stemgen.py), not a random draw.
"""
import numpy as np

import plant as pl

_bits = lambda b: f'contype="{b[0]}" conaffinity="{b[1]}"'
FRUIT_BITS = _bits(pl.PLANT_BITS)
RACHIS_BITS = _bits(pl.PLANT_BITS)

# Cherry tomato diameter 20-30 mm (guess from supermarket cherry tomatoes, unverified).
RADIUS_M = (0.010, 0.015)
# Water-like density gives 4-14 g for that diameter range; real cherry tomatoes are ~10-20 g
# (guess, unverified), so density is set a little above water.
DENSITY = 1050.0

# Pedicel from the truss stem (rachis) to the calyx, ~10-15 mm (guess, unverified).
PEDICEL_LEN_M = (0.010, 0.015)
# Ball joint at the rachis lets the fruit swing like a real one. Stiffness/damping are guesses
# picked so a 10 g fruit sags a few degrees and stops swinging within ~1 s (unverified).
PEDICEL_STIFFNESS = 0.004     # N.m/rad
PEDICEL_DAMPING = 2e-4        # N.m.s/rad
PEDICEL_ARMATURE = 1e-6       # ⚠️ without it the near-massless pedicel body makes the ball joint jitter
# Pedicels lean out from the rachis, alternating sides. Was a fixed 30 deg; K7 (2026-10-27) asked for
# fruit in different orientations, so each pedicel now draws its lean and a sideways swing (guess from
# photos, unverified).
PEDICEL_LEAN_DEG = (15.0, 45.0)
PEDICEL_SWING_DEG = 25.0      # +- turn of the lean direction around the vertical
# The fruit doesn't hang dead below its calyx: the centre sits up to this far off straight down from
# the pedicel tip, so the calyx (below) shows at different angles (guess from photos, unverified).
HANG_TILT_DEG = (0.0, 35.0)
# Calyx: 5 sepals lying on the fruit around the pedicel. Visual only (no collision, no mass), so the
# grasp physics is the same as the bare sphere; it gives the fruit a visible orientation and looks
# more like the photos the gate learned from. Sizes are a guess from supermarket vine tomatoes.
CALYX_SEPALS = 5
CALYX_SPREAD_DEG = 42.0       # sepal tip, measured from the calyx centre over the fruit surface
CALYX_RGBA = "0.22 0.38 0.12 1"

# The truss stem (rachis) hangs off the vine on a sprung ball joint, so a truss sways when the
# gripper drags a fruit. Rigid, it pushed all of a closing jaw's ~8 mm drag into the tether and
# plucked the weakest fruit (check_scene.py). 0.3 N.m/rad = ~10 deg of sag for a 10-fruit truss on an
# 8 cm rachis: guess, unverified (push a real truss with a scale and measure the sway).
RACHIS_STIFFNESS = 0.3        # N.m/rad
RACHIS_DAMPING = 0.01         # N.m.s/rad, guess: settles in ~1 s
RACHIS_ARMATURE = 1e-4        # ⚠️ the 2 mm rachis capsule is nearly massless; without this the joint jitters
RACHIS_LEVER_M = 0.08         # single_xml's stand-in truss length

# Weld softness = the tether spring. (timeconst, dampratio) in MuJoCo solref. 20 ms is MuJoCo's
# default: stiff enough that a hanging fruit moves < 1 mm. (unverified against a real pedicel)
TETHER_SOLREF = (0.02, 1.0)

# Gap kept between neighbouring fruits when they're laid out (guess; real trusses touch sometimes).
FRUIT_GAP_M = 0.002

# Ripeness colours, rough sRGB of the classes the gate knows (guess from photos, unverified).
RIPENESS_RGBA = {
    "green":   (0.35, 0.55, 0.20, 1),
    "turning": (0.85, 0.55, 0.20, 1),
    "ripe":    (0.75, 0.08, 0.05, 1),
}
STEM_RGBA = "0.30 0.45 0.15 1"


def _rgba(rip):
    return " ".join(f"{c:.2f}" for c in RIPENESS_RGBA[rip])


def _calyx(name, r, up, spin):
    """Sepal geoms (fruit body frame) around the point of the fruit facing `up` (unit, toward the pedicel)."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    e1 = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    c0, sp = up * r * 1.02, np.radians(CALYX_SPREAD_DEG)
    out = []
    for k in range(CALYX_SEPALS):
        th = spin + 2 * np.pi * k / CALYX_SEPALS
        tipp = (np.cos(sp) * up + np.sin(sp) * (np.cos(th) * e1 + np.sin(th) * e2)) * r * 1.04
        out.append(f'<geom name="{name}_calyx{k}" type="capsule" fromto="{c0[0]:.5f} {c0[1]:.5f} {c0[2]:.5f} {tipp[0]:.5f} {tipp[1]:.5f} '
                   f'{tipp[2]:.5f}" size="0.0011" density="0" rgba="{CALYX_RGBA}" contype="0" conaffinity="0"/>')
    return "".join(out)


def _fruit(name, local, world, tip, r, rip, threshold, hang=(0.0, 0.0, -1.0), spin=0.0):
    """Pedicel body (local pos in its parent), tomato body (world), weld, info dict.
    hang: unit direction from the pedicel tip to the fruit centre (straight down by default)."""
    tn, pn = f"{name}", f"{name}_ped"
    ped = (f'<body name="{pn}" pos="{local[0]:.5f} {local[1]:.5f} {local[2]:.5f}">'
           f'<joint name="{pn}_ball" type="ball" stiffness="{PEDICEL_STIFFNESS}" '
           f'damping="{PEDICEL_DAMPING}" armature="{PEDICEL_ARMATURE}"/>'
           f'<geom name="{pn}" type="capsule" fromto="0 0 0 {tip[0]:.5f} {tip[1]:.5f} {tip[2]:.5f}" size="0.0012" '
           f'mass="0.0005" rgba="{STEM_RGBA}" contype="0" conaffinity="0"/></body>')
    hang = np.asarray(hang, float) / np.linalg.norm(hang)
    c = world + tip + hang * r
    # Tomatoes are world children (free bodies), not children of the pedicel, so that switching
    # the weld off really frees them.
    body = (f'<body name="{tn}" pos="{c[0]:.5f} {c[1]:.5f} {c[2]:.5f}"><freejoint/>'
            f'<geom name="{tn}_g" type="sphere" size="{r:.5f}" density="{DENSITY}" rgba="{_rgba(rip)}" '
            f'friction="0.8 0.01 0.001" condim="4" {FRUIT_BITS}/>{_calyx(tn, r, -hang, spin)}</body>')
    eq = (f'<weld name="{tn}_tether" body1="{tn}" body2="{pn}" '
          f'solref="{TETHER_SOLREF[0]} {TETHER_SOLREF[1]}"/>')
    return ped, body, eq, dict(tomato=tn, pedicel=pn, eq=f"{tn}_tether", geom=f"{tn}_g", radius=r,
                               ripeness=rip, threshold=threshold, pos=c)


def truss_xml(name, rachis_from, rachis_to, n, rng, ripeness=None, avoid=()):
    """MJCF for one truss: a static rachis capsule with up to n tomatoes hanging off it.

    rachis_from/to: world points (m) of the truss stem. ripeness: list of RIPENESS_RGBA keys (len
    n), random if None. avoid: (centre, radius) of fruits already placed elsewhere. A fruit that
    would overlap a neighbour gets a smaller radius, then is dropped, so a truss can come out
    with fewer than n tomatoes.
    Returns (rachis_body_xml, tomato_bodies_xml, equality_xml, fruit_info_list).
    """
    a, b = np.asarray(rachis_from, float), np.asarray(rachis_to, float)
    mid, axis = (a + b) / 2, b - a
    L = np.linalg.norm(axis)
    out = np.cross(axis / L, [0, 0, 1])
    out = out / np.linalg.norm(out) if np.linalg.norm(out) > 1e-6 else np.array([1.0, 0, 0])
    placed = list(avoid)
    peds, bodies, eqs, fruit = [], [], [], []
    for i in range(n):
        p_world = a + (i + 0.5) / n * axis
        side = 1 if i % 2 == 0 else -1
        lp = rng.uniform(*PEDICEL_LEN_M)
        lean = np.radians(rng.uniform(*PEDICEL_LEAN_DEG))
        sw = np.radians(rng.uniform(-PEDICEL_SWING_DEG, PEDICEL_SWING_DEG))
        o = side * (np.cos(sw) * out + np.sin(sw) * np.cross([0, 0, 1.0], out))
        tip = (np.sin(lean) * o + np.array([0, 0, -np.cos(lean)])) * lp
        tilt, az = np.radians(rng.uniform(*HANG_TILT_DEG)), rng.uniform(0, 2 * np.pi)
        hang = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), -np.cos(tilt)])
        spin = rng.uniform(0, 2 * np.pi)
        rip = ripeness[i] if ripeness else str(rng.choice(list(RIPENESS_RGBA)))
        r = rng.uniform(*RADIUS_M)
        for _ in range(4):
            c = p_world + tip + hang * r
            if all(np.linalg.norm(c - c2) >= r + r2 + FRUIT_GAP_M for c2, r2 in placed):
                break
            r *= 0.85
        else:
            continue
        if r < RADIUS_M[0] * 0.8:
            continue
        ped, body, eq, info = _fruit(f"{name}_t{i}", p_world - mid, p_world, tip, r, rip,
                                     0.0, hang=hang, spin=spin)
        info["truss"] = name
        placed.append((info["pos"], r))
        peds.append(ped); bodies.append(body); eqs.append(eq); fruit.append(info)
    rachis = (f'<body name="{name}_rachis" pos="{mid[0]:.5f} {mid[1]:.5f} {mid[2]:.5f}">'
              f'<joint name="{name}_rachis_ball" type="ball" pos="{-axis[0]/2:.5f} {-axis[1]/2:.5f} {-axis[2]/2:.5f}" '
              f'stiffness="{RACHIS_STIFFNESS}" damping="{RACHIS_DAMPING}" armature="{RACHIS_ARMATURE}"/>'
              f'<geom name="{name}_rachis" type="capsule" fromto="{-axis[0]/2:.5f} {-axis[1]/2:.5f} {-axis[2]/2:.5f} '
              f'{axis[0]/2:.5f} {axis[1]/2:.5f} {axis[2]/2:.5f}" size="0.002" '
              f'rgba="{STEM_RGBA}" {RACHIS_BITS}/>' + "".join(peds) + "</body>")
    return rachis, "\n".join(bodies), "\n".join(eqs), fruit
