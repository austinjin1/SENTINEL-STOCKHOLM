# SENTINEL-Lite ROS 2 pipeline

Drone captures imagery → streams it over the network → ground computer runs
**WaterDroneNet** and predicts water-quality parameters. One repo, cloned on
both machines; each machine runs only its half.

```
  Raspberry Pi (drone)                         Omen 15 (ground computer)
  ┌──────────────────────┐                     ┌──────────────────────────────┐
  │ sentinel_camera       │   sensor_msgs/Image │ sentinel_inference            │
  │  rgb_publisher  ──────┼──── /sentinel/camera/rgb/image_raw ──▶ frame_sync   │
  │  nir_publisher  ──────┼──── /sentinel/camera/nir/image_raw ──▶  + assemble  │
  │  (CameraBackend HAL)   │        DDS / FastRTPS         │      → WaterDroneNet │
  └──────────────────────┘     ROS_DOMAIN_ID=42           │      → /sentinel/water_quality │
                                                          └──────────────────────────────┘
                                                  sentinel_perception_msgs/WaterQuality
```

## Why DDS role-launch (not a socket server / broker)

The two machines are already a natural fit for ROS 2's transport: **DDS does the
discovery and delivery.** As long as both boxes share `ROS_DOMAIN_ID` and sit on
the same subnet, the publisher on the Pi and the subscriber on the Omen find each
other with zero wiring. So orchestration is just: _source one env file, launch
your role._ No extra broker to run, no IPs to hardcode (until multicast is
blocked — then drop in a unicast peer list, see `network/sentinel_network.sh`).

The two halves are **independent features in one workspace**:

| Package                    | Build type     | Runs on      | Role     |
| -------------------------- | -------------- | ------------ | -------- |
| `sentinel_perception_msgs` | `ament_cmake`  | both (build) | shared   |
| `sentinel_camera`          | `ament_python` | Raspberry Pi | drone    |
| `sentinel_inference`       | `ament_python` | Omen 15      | computer |

Both machines build `sentinel_perception_msgs` (the message type must exist on
both ends). The Pi need not build `sentinel_inference` (no torch on the Pi); the
Omen need not build `sentinel_camera`. `colcon build --packages-up-to` selects.

## Build

```bash
cd ros2_ws

# Drone (Raspberry Pi): cameras + shared msgs only
colcon build --packages-up-to sentinel_camera

# Ground computer (Omen 15): inference + shared msgs only
colcon build --packages-up-to sentinel_inference

source install/setup.bash
```

## Run

On **both** machines first:

```bash
source ros2_ws/network/sentinel_network.sh   # shared ROS_DOMAIN_ID
```

Then, per role:

```bash
# Raspberry Pi
ros2 launch sentinel_camera drone.launch.py

# Omen 15
ros2 launch sentinel_inference computer.launch.py
```

Verify the link from either box: `ros2 topic list` should show both camera
topics and `/sentinel/water_quality`; `ros2 topic hz /sentinel/camera/rgb/image_raw`
checks the stream rate across the network.

## Test without hardware or ROS

Capture, 4-band assembly, and time-sync logic are isolated from `rclpy`,
`torch`, and the camera (HAL `synthetic` backend + pure-NumPy modules), so they
run on any laptop:

```bash
python -m pytest ros2_ws/src/sentinel_camera/test ros2_ws/src/sentinel_inference/test -q
```

Full-graph + torch inference must run on a box with ROS 2 and torch installed.

## Finalized parameters (from CINE-Sensing)

Capture params below are agreed across CINE-Sensing's `camera.json`, package
`camera_params.yaml`, and the calibration run — locked for `sentinel_camera`.

| Param           | Value                                            |
| --------------- | ------------------------------------------------ |
| RGB camera      | RPi Camera Module 3 Wide                         |
| Resolution      | 1152 × 648                                       |
| FPS             | 30.0                                             |
| Pixel format    | `bgr8` (color — we do **not** grayscale)         |
| Backend         | libcamera `rpicam-vid` (mjpeg) → opencv fallback |
| Exposure / gain | 10000 µs / 1.0, AWB auto                         |
| QoS             | RELIABLE · VOLATILE · KEEP_LAST · depth 10       |
| Optical frame   | `camera_optical_frame`                           |

**RGB intrinsics** (`camera_info`): CINE-Sensing ships three _conflicting_ sets;
we use the newest calibration (`calibration/camera_calibration.yaml`,
2025-10-31, mean reproj err 0.46): `fx=497.60 fy=497.43 cx=578.45 cy=320.93`,
distortion `k1=-0.0578 k2=-0.0328 p1=-0.00243 p2=0.000290`, PINHOLE @ 1152×648.
Intrinsics do **not** affect WaterDroneNet (it consumes a 224×224 image); they
matter only for `camera_info`/undistort/geo-registration.

**NIR camera** (NoIR V2, 8 MP, 1080p30): CINE-Sensing has no NIR calibration.
NIR `camera_info` intrinsics are a **TODO** — to be filled after a NoIR
calibration run; not fabricated here.

## Status

- [x] Phase 1 — foundation: workspace, DDS env, shared `WaterQuality` message
- [ ] Phase 2 — drone: `sentinel_camera` (HAL backends, RGB/NIR publishers)
- [ ] Phase 3 — computer: `sentinel_inference` (assembly, sync, model node)
- [ ] Phase 4 — orchestration: role launch wrappers, run scripts, e2e docs
