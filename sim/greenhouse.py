"""The mini greenhouse: the laser-cut 6 mm birch ply frame the SO-101 works in.

    xml = greenhouse_xml()

Geometry copied from B.L.U.S.H sim/scene.py (commit e91ef0c, greenhouse_xml), which takes it from the
cut generator (B.L.U.S.H hardware/greenhouse/lasercut/greenhouse_lasercut.py, mm there, metres here):
three vine rows at y = +0.4 / 0 / -0.4, end frames at x = 0 and 0.74, crop rails at z = 0.52 that the
vines hang from, anchor rails along the floor under each row.

Changed from B.L.U.S.H: the camera mast is left out (SHEAR's birdseye sits elsewhere, layout.py), and the
frame uses plant.FLOOR_BITS, so it touches the robot and the plants. With MuJoCo's default bits it would be
SHEAR's robot bit.
"""
import numpy as np

import plant as pl

T = 0.006                        # ply thickness
HOUSE_L = 0.740                  # inner length between end frames
ROWS = (0.400, 0.0, -0.400)
POST_W = 0.060
BAND = (0.500, 0.590)            # crop tie band in the end frames
CROP_Z = 0.520                   # underside of the crop flange = where the vines hang from
WEB_H, ANCHOR_WEB_H, FLANGE_W = 0.050, 0.030, 0.050
BASE_H, EAR = 0.030, 0.040
ROOF_DEG = 15.0
KEYHOLE_OFF = 0.015              # vines hang from keyholes 15 mm off the row line, on the aisle side
PLY_RGBA = "0.82 0.70 0.52 1"    # birch ply, guess (unverified)
ANCHOR_TOP = ANCHOR_WEB_H + T    # top of a row's anchor rail: where a stem starts


# The roof (rafters, king posts, ridge) is in geom group 1, so a top-down picture can hide it.
ROOF_GROUP = 1


def _box(name, lo, hi, rgba=PLY_RGBA, group=0):
    lo, hi = np.asarray(lo), np.asarray(hi)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    return (f'<geom name="{name}" type="box" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" '
            f'size="{h[0]:.4f} {h[1]:.4f} {h[2]:.4f}" rgba="{rgba}" group="{group}" '
            f'contype="{pl.FLOOR_BITS[0]}" conaffinity="{pl.FLOOR_BITS[1]}"/>')


def greenhouse_xml():
    g = []
    yo = max(abs(r) for r in ROWS) + POST_W / 2
    apex = BAND[1] + yo * np.tan(np.radians(ROOF_DEG))
    for k, x0 in enumerate((-T, HOUSE_L)):           # end frames: posts, base tie, crop band, rafters
        x1 = x0 + T
        for j, r in enumerate(ROWS):
            g.append(_box(f"gh_ef{k}_post{j}", (x0, r - POST_W / 2, 0), (x1, r + POST_W / 2, BAND[1])))
        g.append(_box(f"gh_ef{k}_tie", (x0, -yo - EAR, 0), (x1, yo + EAR, BASE_H)))
        g.append(_box(f"gh_ef{k}_band", (x0, -yo, BAND[0]), (x1, yo, BAND[1])))
        g.append(_box(f"gh_ef{k}_king", (x0, -POST_W / 4, BAND[1]), (x1, POST_W / 4, apex), group=ROOF_GROUP))
        for s in (1, -1):
            half = np.hypot(yo, apex - BAND[1]) / 2
            g.append(f'<geom name="gh_ef{k}_rafter{s:+d}" type="box" pos="{x0 + T/2:.4f} {s*yo/2:.4f} '
                     f'{(BAND[1]+apex)/2:.4f}" size="{T/2:.4f} {half:.4f} 0.02" euler="{-s*np.radians(ROOF_DEG):.4f} 0 0" '
                     f'rgba="{PLY_RGBA}" group="{ROOF_GROUP}" contype="{pl.FLOOR_BITS[0]}" conaffinity="{pl.FLOOR_BITS[1]}"/>')
    for j, r in enumerate(ROWS):                     # crop rails (T: flange under, web on top) + anchor rails
        g.append(_box(f"gh_crop{j}_flange", (0, r - FLANGE_W / 2, CROP_Z), (HOUSE_L, r + FLANGE_W / 2, CROP_Z + T)))
        g.append(_box(f"gh_crop{j}_web", (0, r - T / 2, CROP_Z + T), (HOUSE_L, r + T / 2, CROP_Z + T + WEB_H)))
        g.append(_box(f"gh_anchor{j}_web", (0, r - T / 2, 0), (HOUSE_L, r + T / 2, ANCHOR_WEB_H)))
        g.append(_box(f"gh_anchor{j}_flange", (0, r - FLANGE_W / 2, ANCHOR_WEB_H), (HOUSE_L, r + FLANGE_W / 2, ANCHOR_TOP)))
    g.append(_box("gh_ridge", (0, -0.02, apex - 0.055), (HOUSE_L, 0.02, apex - 0.015), group=ROOF_GROUP))
    return "\n".join(g)
