#!/usr/bin/env bash
# Computer role (Omen 15): source network + workspace, launch inference.
#   ros2_ws/scripts/run_computer.sh                          # cpu, random weights (smoke)
#   ros2_ws/scripts/run_computer.sh cuda /path/to/model.pt   # gpu + checkpoint
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(dirname "$HERE")"

export SENTINEL_ROLE=computer
source "$WS/network/sentinel_network.sh"

if [ ! -f "$WS/install/setup.bash" ]; then
  echo "[computer] workspace not built — run: (cd $WS && colcon build --packages-up-to sentinel_inference)" >&2
  exit 1
fi
source "$WS/install/setup.bash"

DEVICE="${1:-cpu}"
CKPT="${2:-}"
exec ros2 launch sentinel_inference computer.launch.py device:="$DEVICE" checkpoint:="$CKPT"
