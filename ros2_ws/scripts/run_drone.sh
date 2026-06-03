#!/usr/bin/env bash
# Drone role (Raspberry Pi): source network + workspace, launch cameras.
#   ros2_ws/scripts/run_drone.sh                 # auto-detect camera
#   ros2_ws/scripts/run_drone.sh synthetic       # no hardware (testing)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(dirname "$HERE")"

export SENTINEL_ROLE=drone
source "$WS/network/sentinel_network.sh"

if [ ! -f "$WS/install/setup.bash" ]; then
  echo "[drone] workspace not built — run: (cd $WS && colcon build --packages-up-to sentinel_camera)" >&2
  exit 1
fi
source "$WS/install/setup.bash"

BACKEND="${1:-auto}"
exec ros2 launch sentinel_camera drone.launch.py backend:="$BACKEND"
