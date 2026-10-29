# Third-party files

`assets/so101/` (fetched by `fetch_so101.sh`, not committed) comes from TheRobotStudio/SO-ARM100
(https://github.com/TheRobotStudio/SO-ARM100), commit a758567c3978dfeefe282ede0500085a48fe8f78,
`Simulation/SO101/`. Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
`scene.py` modifies the model at load time: the stock jaw meshes become visual only, boxes stand in for
the fingers as colliders, each fingertip gets a blade, and the gripper's range and torque limit are set
to the real follower's.

`fetch_so101.sh` is copied from B.L.U.S.H (`krishaanth5831/B.L.U.S.H`, `sim/`, commit e91ef0c).

`truss.py` is copied from B.L.U.S.H `sim/tomato.py` (commit e91ef0c): the truss and fruit builder only, not
the Detacher. Changed: fruit and rachis use SHEAR's plant collision bits (B.L.U.S.H's default 1/1 is SHEAR's
robot bit), the rachis collides, geoms are named for segmentation, and ripeness comes from the truss stage.

`greenhouse.py` takes the mini greenhouse's geometry from B.L.U.S.H `sim/scene.py` (commit e91ef0c,
greenhouse_xml), without its camera mast, on SHEAR's collision bits. `lens.py` is copied unchanged and
`cameras.py` (renderers only) from the same commit. Camera placement is SHEAR's own (`layout.py`).

`arm_map.py` is copied unchanged from B.L.U.S.H `sim/arm_map.py` (commit 5c12eea). `cutpose.py` copies the
Levenberg-Marquardt loop and the sampled warm starts of B.L.U.S.H `sim/expert.py` `ArmIK` (commit cec86fb); its
residuals are SHEAR's own. `cameras.py` gained `depth()` and `cam_depth()` (SHEAR step 4).
