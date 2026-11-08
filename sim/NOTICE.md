# Third-party files

## SO-101 model

`assets/so101/` (fetched by `fetch_so101.sh`, not committed) comes from TheRobotStudio/SO-ARM100
(https://github.com/TheRobotStudio/SO-ARM100), commit a758567c3978dfeefe282ede0500085a48fe8f78,
`Simulation/SO101/`. Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
`scene.py` modifies the model at load time: the stock jaw meshes become visual only, boxes stand in for
the fingers as colliders, each fingertip gets blades, and the gripper's range and torque limit are set
to the real follower's.

## Code from B.L.U.S.H

From B.L.U.S.H (`krishaanth5831/B.L.U.S.H`, `sim/`), Vinea's harvest module, which shares the arm and the
cameras:

- `fetch_so101.sh`: copied (commit e91ef0c).
- `truss.py`: copied from `sim/tomato.py` (commit e91ef0c), the truss and fruit builder only, not the Detacher.
  Changed: fruit and rachis use SHEAR's plant collision bits (B.L.U.S.H's default 1/1 is SHEAR's robot bit),
  the rachis collides, geoms are named for segmentation, and ripeness comes from the truss stage.
- `greenhouse.py`: the mini greenhouse's geometry from `sim/scene.py` (commit e91ef0c, greenhouse_xml), without
  its camera mast, on SHEAR's collision bits.
- `lens.py`: copied (commit e91ef0c). Changed: it adds the v1 rig's "base" camera, with the same lens as the
  birdseye (2026-11-06).
- `cameras.py`: the renderers only (commit e91ef0c). Changed: `depth()` and `cam_depth()` added (SHEAR step 4),
  and the camera list is SHEAR's (base and claw since 2026-11-06). Where the cameras sit is SHEAR's own
  (`layout.py`).
- `arm_map.py`: copied from `sim/arm_map.py` (commit 5c12eea), code unchanged.
- `cutpose.py`: copies the Levenberg-Marquardt loop and the sampled warm starts of `sim/expert.py` `ArmIK`
  (commit cec86fb). Its residuals are SHEAR's own.
- `randomize.py`: adapted from `sim/randomize.py` (commit 41da64d): its lights, surfaces and camera jitter,
  on SHEAR's own geoms, without its physics draws.
- `cutter.py`: the robot's start pose is B.L.U.S.H's `START_NORM` (`sim/scene.py`, commit 41da64d).
