"""The InnoMaker 130 deg lens on top of MuJoCo's pinhole cameras.

Copied from B.L.U.S.H sim/lens.py (commit e91ef0c) unchanged: it models the camera hardware, which SHEAR
shares with B.L.U.S.H. Where the cameras sit is SHEAR's own (layout.py).

MuJoCo cameras are pinhole. The real birdseye and claw cameras (InnoMaker U20CAM-1080P, M12 wide
lens) bend straight lines and squeeze the edges. So each sim camera renders a wider pinhole image
and remaps it through a fisheye model into the 640x480 frame the gate and the policy see:

    lens = LENSES["birdseye"]
    rgb = lens.distort(pinhole_rgb)                  # (lens.pin_h, lens.pin_w, 3) -> (480, 640, 3)
    ids = lens.distort(pinhole_ids, nearest=True)    # segmentation: no blending between object ids
    uv = lens.project(points_cam)                    # OpenCV camera frame (x right, y down, z forward)

The model is OpenCV's fisheye one (Kannala-Brandt): a ray at angle theta from the optical axis lands
at radius f * theta_d, theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8).
So `cv2.fisheye.calibrate` output (K, D) drops straight in: put it in sim/cameras.json (see
MEASURED below) and every sim camera uses it.
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MEASURED = HERE / "cameras.json"   # {"birdseye": {"K": 3x3, "D": [k1..k4], "size": [w, h], "source": "..."}, "claw": ...}
W, H = 640, 480                    # config/hardware.env CAM_W x CAM_H

# --- the guess, until a checkerboard calibration replaces it ---
# Retail spec of InnoMaker's 1080p USB module with the 130 deg M12 lens: FOV 130 deg diagonal,
# 103 deg horizontal at 1920x1080 (Newegg listings; the U20CAM-1080P part number isn't on them, so
# unverified). One distortion term fits both numbers: k1 = -0.165, f = 1232 px at 1080p
# (solved for both numbers exactly, 2026-10-20). A plain f-theta lens with 130 deg diagonal would give
# 113 deg horizontal, so the spec says the edges are squeezed harder than f-theta.
SPEC_FULL = (1920, 1080)
SPEC_K1 = -0.16459
SPEC_F_FULL = 1231.85
# ⚠️ unverified: how the camera makes 640x480 out of a 16:9 sensor. "crop" = cut the sides to
# 1440x1080, then scale (square pixels; most UVC webcams do this). "squash" = scale the full
# 1920x1080 frame down unevenly (non-square pixels). One frame of the same view at 640x480 and
# 1920x1080 settles it.
ASPECT = "crop"


def _spec_guess():
    sw, sh = SPEC_FULL
    if ASPECT == "crop":
        sx = sy = H / sh
    else:
        sx, sy = W / sw, H / sh
    K = np.array([[SPEC_F_FULL * sx, 0, (W - 1) / 2], [0, SPEC_F_FULL * sy, (H - 1) / 2], [0, 0, 1]])
    return K, np.array([SPEC_K1, 0, 0, 0])


class Lens:
    def __init__(self, K, D, size=(W, H), source=""):
        self.K, self.D = np.asarray(K, float), np.asarray(D, float)
        self.w, self.h = size
        self.source = source
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        # Each output pixel's ray, as a point on the z = 1 plane.
        u, v = np.meshgrid(np.arange(self.w, dtype=float), np.arange(self.h, dtype=float))
        a, b = (u - cx) / fx, (v - cy) / fy
        rd = np.hypot(a, b)
        th = self.theta(rd)
        s = np.where(rd > 1e-9, np.tan(th) / np.maximum(rd, 1e-9), 1.0)
        x, y = a * s, b * s
        # The pinhole render: same pixel size as the fisheye frame's centre, wide enough for its corners.
        self.f_pin = max(fx, fy)
        self.pin_w = 2 * int(np.ceil(self.f_pin * np.abs(x).max())) + 4
        self.pin_h = 2 * int(np.ceil(self.f_pin * np.abs(y).max())) + 4
        self.fovy = float(np.degrees(2 * np.arctan(self.pin_h / 2 / self.f_pin)))   # MuJoCo camera fovy
        # MuJoCo puts the principal point at the image centre; pixel k's centre is at k + 0.5.
        self.map_x = (self.f_pin * x + self.pin_w / 2 - 0.5).astype(np.float32)
        self.map_y = (self.f_pin * y + self.pin_h / 2 - 0.5).astype(np.float32)
        self.fov_deg = {k: float(np.degrees(2 * self.theta(r))) for k, r in
                        (("h", self.w / 2 / fx), ("v", self.h / 2 / fy), ("d", np.hypot(self.w / 2 / fx, self.h / 2 / fy)))}

    def _poly(self, th):
        t2 = th * th
        k1, k2, k3, k4 = self.D
        return th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def theta(self, rd):
        """Invert theta_d(theta) by Newton steps. rd: distorted radius on the normalised plane."""
        rd = np.asarray(rd, float)
        th = np.array(rd, copy=True)
        for _ in range(30):
            e = self._poly(th) - rd
            dth = (self._poly(th + 1e-6) - self._poly(th - 1e-6)) / 2e-6
            th = th - e / dth
        return th

    def distort(self, pinhole, nearest=False):
        import cv2
        assert pinhole.shape[:2] == (self.pin_h, self.pin_w), (pinhole.shape, (self.pin_h, self.pin_w))
        return cv2.remap(pinhole, self.map_x, self.map_y, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def project(self, pts):
        """(N, 3) points in the OpenCV camera frame -> (N, 2) pixels, NaN behind the camera."""
        pts = np.atleast_2d(np.asarray(pts, float))
        r = np.hypot(pts[:, 0], pts[:, 1])
        th = np.arctan2(r, pts[:, 2])
        td = self._poly(th)
        s = np.where(r > 1e-12, td / np.maximum(r, 1e-12), 0.0)
        uv = np.stack([self.K[0, 0] * pts[:, 0] * s + self.K[0, 2], self.K[1, 1] * pts[:, 1] * s + self.K[1, 2]], 1)
        uv[pts[:, 2] <= 0] = np.nan
        return uv

    def describe(self):
        f = self.fov_deg
        return (f"K fx {self.K[0, 0]:.1f} fy {self.K[1, 1]:.1f}, D {np.round(self.D, 4).tolist()}, "
                f"FOV {f['h']:.0f}H x {f['v']:.0f}V ({f['d']:.0f} diagonal), "
                f"pinhole render {self.pin_w}x{self.pin_h} fovy {self.fovy:.1f}  [{self.source}]")


def load():
    K, D = _spec_guess()
    guess = f"⚠️ guess: InnoMaker spec 130D/103H, {ASPECT} to 640x480, unverified"
    out = {c: Lens(K, D, source=guess) for c in ("birdseye", "claw")}
    if MEASURED.exists():
        for c, v in json.loads(MEASURED.read_text()).items():
            out[c] = Lens(v["K"], v["D"], tuple(v.get("size", (W, H))), source=v.get("source", "measured"))
    return out


LENSES = load()

if __name__ == "__main__":
    for c, lens in LENSES.items():
        print(f"{c:8s} {lens.describe()}")
