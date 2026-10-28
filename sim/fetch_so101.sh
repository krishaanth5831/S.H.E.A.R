#!/usr/bin/env bash
# Fetch the SO-101 MJCF + meshes the sim needs into sim/assets/so101 (gitignored: ~16 MB of STLs).
# Pinned to the SO-ARM100 commit the fin-ray gripper was built against. Apache-2.0, see NOTICE.md.
set -euo pipefail
COMMIT=a758567c3978dfeefe282ede0500085a48fe8f78
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/assets/so101"
TMP="$(mktemp -d)"
trap 'rm -r "$TMP"' EXIT
git clone -q --filter=blob:none --no-checkout https://github.com/TheRobotStudio/SO-ARM100.git "$TMP/repo"
git -C "$TMP/repo" sparse-checkout set Simulation/SO101
git -C "$TMP/repo" checkout -q "$COMMIT"
SRC="$TMP/repo/Simulation/SO101"
mkdir -p "$DEST/assets"
cp "$SRC/so101_new_calib.xml" "$SRC/so101_new_calib_camera.xml" "$DEST/"
for m in $(grep -oh 'file="[^"]*\.stl"' "$SRC/so101_new_calib_camera.xml" | sed 's/file="//;s/"//' | sort -u); do
  cp "$SRC/assets/$m" "$DEST/assets/"
done
echo "SO-101 model at $DEST (SO-ARM100 $COMMIT)"
