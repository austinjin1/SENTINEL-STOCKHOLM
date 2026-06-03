# SENTINEL drone <-> ground-computer DDS network config.
# Source this on BOTH machines before launching. Mirrors CINE-Sensing.
#
#   source ros2_ws/network/sentinel_network.sh
#
# The two machines discover each other automatically over DDS as long as they
# share ROS_DOMAIN_ID, are on the same L2 network (same WiFi/subnet), and
# multicast is not blocked. No broker, no static IPs required for the happy path.

# --- Domain: both machines MUST match -----------------------------------------
export ROS_DOMAIN_ID=42

# --- Allow off-localhost traffic (0 = talk to other hosts) --------------------
export ROS_LOCALHOST_ONLY=0

# --- DDS implementation -------------------------------------------------------
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# --- QoS defaults (match sentinel_camera / sentinel_inference nodes) ----------
export RMW_QOS_RELIABILITY_POLICY=RELIABLE
export RMW_QOS_DURABILITY_POLICY=VOLATILE
export RMW_QOS_HISTORY_POLICY=KEEP_LAST
export RMW_QOS_DEPTH=10

# --- Role hint (optional): "drone" or "computer" ------------------------------
# Launch files don't require it, but scripts/diagnostics read it.
export SENTINEL_ROLE="${SENTINEL_ROLE:-unset}"

# --- Unicast peers (uncomment if multicast is blocked on your network) --------
# When WiFi APs drop multicast, list the OTHER machine's IP here on each box and
# point Fast DDS at this profile. Fill in real IPs.
#   Drone box:    export SENTINEL_PEER=192.168.1.50   # the computer
#   Computer box: export SENTINEL_PEER=192.168.1.42   # the Pi
# export FASTRTPS_DEFAULT_PROFILES_FILE="$(dirname "${BASH_SOURCE[0]}")/fastdds_unicast.xml"

echo "[sentinel-network] ROS_DOMAIN_ID=$ROS_DOMAIN_ID role=$SENTINEL_ROLE rmw=$RMW_IMPLEMENTATION"
