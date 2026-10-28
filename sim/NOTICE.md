# Third-party files

`assets/so101/` (fetched by `fetch_so101.sh`, not committed) comes from TheRobotStudio/SO-ARM100
(https://github.com/TheRobotStudio/SO-ARM100), commit a758567c3978dfeefe282ede0500085a48fe8f78,
`Simulation/SO101/`. Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
`scene.py` modifies the model at load time: the stock jaw meshes become visual only, boxes stand in for
the fingers as colliders, each fingertip gets a blade, and the gripper's range and torque limit are set
to the real follower's.

`fetch_so101.sh` is copied from B.L.U.S.H (`krishaanth5831/B.L.U.S.H`, `sim/`, commit e91ef0c).
