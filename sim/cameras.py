"""The sim's birdseye + claw cameras, rendered through the lens model (lens.py).

Copied from B.L.U.S.H sim/cameras.py (commit e91ef0c): the renderers only. Camera placement is SHEAR's
own, in layout.py. Changed: CAM_IN_MODEL names; sphere_box dropped (fruit boxes are B.L.U.S.H's job).
Added (SHEAR step 4): depth(), for the keypoint labels' visibility test (labels.py).

    cams = SimCameras(m)                      # ⚠️ one per model AND per thread: a MuJoCo renderer is
    rgb = cams.rgb(d, "birdseye")             #    bound to the model and to the GL context's thread
    box, px = cams.true_box(d, "birdseye", geom_id)
    cams.close()
"""
import mujoco
import numpy as np

from lens import LENSES

CAMERAS = ("birdseye", "claw")                         # same names as B.L.U.S.H, so the real rig matches
CAM_IN_MODEL = {"birdseye": "birdseye", "claw": "arm/claw"}


def pinhole_size():
    """The offscreen buffer every camera's pinhole render fits in (scene.py sets it on the model)."""
    return max(l.pin_w for l in LENSES.values()), max(l.pin_h for l in LENSES.values())


class SimCameras:
    def __init__(self, m):
        self.m = m
        self._r = {}
        self._seg = {}
        self._depth = {}

    def _renderer(self, lens):
        key = (lens.pin_h, lens.pin_w)
        if key not in self._r:
            self._r[key] = mujoco.Renderer(self.m, lens.pin_h, lens.pin_w)
        return self._r[key]

    def _seg_renderer(self, lens):
        """A renderer for segmentation only, without multisampling. ⚠️ With MSAA the ids (spread over the
        RGB bytes) blend at edges byte by byte: once a scene has more than 256 geoms (22 fruit x 5 calyx
        sepals) a blended pixel decodes to an id past the last geom, and MuJoCo's render() raised
        IndexError (seed 100060, 2026-10-27). offsamples is read when the GL context is made."""
        key = (lens.pin_h, lens.pin_w)
        if key not in self._seg:
            keep = int(self.m.vis.quality.offsamples)
            self.m.vis.quality.offsamples = 0
            try:
                self._seg[key] = mujoco.Renderer(self.m, lens.pin_h, lens.pin_w)
            finally:
                self.m.vis.quality.offsamples = keep
            self._seg[key].enable_segmentation_rendering()
        return self._seg[key]

    def rgb(self, d, cam):
        lens = LENSES[cam]
        r = self._renderer(lens)
        r.update_scene(d, camera=CAM_IN_MODEL[cam])
        return lens.distort(r.render())

    def ids(self, d, cam):
        """Per pixel: the geom id seen there, -1 for none (the segmentation render through the lens)."""
        lens = LENSES[cam]
        r = self._seg_renderer(lens)
        r.update_scene(d, camera=CAM_IN_MODEL[cam])
        seg = r.render()
        geom = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0] + 1, 0).astype(np.uint16)
        # ⚠️ nearest, not linear: blending two ids gives a third, unrelated id.
        return lens.distort(geom, nearest=True).astype(np.int32) - 1

    def depth(self, d, cam):
        """Per pixel: depth (m) along the camera's view axis, through the lens; 0 where nothing was drawn.
        Own renderer, like ids(): the depth buffer of the MSAA one would blend at edges too."""
        lens = LENSES[cam]
        key = (lens.pin_h, lens.pin_w)
        if key not in self._depth:
            keep = int(self.m.vis.quality.offsamples)
            self.m.vis.quality.offsamples = 0
            try:
                self._depth[key] = mujoco.Renderer(self.m, lens.pin_h, lens.pin_w)
            finally:
                self.m.vis.quality.offsamples = keep
            self._depth[key].enable_depth_rendering()
        r = self._depth[key]
        r.update_scene(d, camera=CAM_IN_MODEL[cam])
        z = r.render().astype(np.float32)
        z[z >= self.m.stat.extent * self.m.vis.map.zfar * 0.99] = 0.0       # the far plane: nothing there
        # ⚠️ nearest, not linear: blending a near and a far surface makes a depth where nothing is.
        return lens.distort(z, nearest=True)

    def cam_depth(self, d, cam, pts_world):
        """World points -> their depth along this camera's view axis (m), as depth() reports it."""
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_IN_MODEL[cam])
        R = d.cam_xmat[cid].reshape(3, 3)
        return -((np.atleast_2d(pts_world) - d.cam_xpos[cid]) @ R)[:, 2]

    def true_box(self, d, cam, geom_id, ids=None):
        """(x0, y0, x1, y1) around a geom's visible pixels and its pixel count, or (None, 0) if hidden."""
        import cv2
        ids = self.ids(d, cam) if ids is None else ids
        # ⚠️ The offscreen buffer is multisampled, so the segmentation render blends ids at object
        # edges: two neighbouring fruits' ids average to a third geom's id, scattering 1-2 px of
        # "target" across the truss (seed 8: a 23 px fruit got a 107 px box). Keep the target's real
        # blobs only, i.e. connected pieces at least a quarter the size of the biggest (a stem in
        # front can split the fruit in two).
        n, lab, stats, _ = cv2.connectedComponentsWithStats((ids == geom_id).astype(np.uint8), connectivity=8)
        if n <= 1:
            return None, 0
        area = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(area >= max(area.max() / 4, 1))[0]
        ys, xs = np.nonzero(np.isin(lab, keep))
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), len(xs)

    def project(self, d, cam, pts_world):
        """World points -> pixels in this camera's 640x480 frame (NaN behind it)."""
        cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_IN_MODEL[cam])
        R = d.cam_xmat[cid].reshape(3, 3)                # columns: camera x, y, z in world
        c = (np.atleast_2d(pts_world) - d.cam_xpos[cid]) @ R
        # MuJoCo cameras look down -z with y up; OpenCV's look down +z with y down.
        return LENSES[cam].project(np.c_[c[:, 0], -c[:, 1], -c[:, 2]])

    def close(self):
        for r in (*self._r.values(), *self._seg.values(), *self._depth.values()):
            r.close()
        self._r, self._seg, self._depth = {}, {}, {}
