"""SENTINEL ground-computer inference node.

Runs on the Omen 15. Subscribes to the drone's RGB stream (and NIR, when
present), assembles a 4-band tensor, runs WaterDroneNet, and publishes a
``WaterQuality`` result. Torch and the model are imported lazily so the rest of
the package stays importable without them.
"""

import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from sentinel_perception_msgs.msg import WaterQuality
from sentinel_inference.preprocessing import assemble_four_band
from sentinel_inference.frame_sync import ApproxTimeSync


def _stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class InferenceNode(Node):
    def __init__(self):
        super().__init__("sentinel_inference")

        self.declare_parameter("rgb_topic", "/sentinel/camera/rgb/image_raw")
        self.declare_parameter("nir_topic", "")  # empty => RGB-only (NIR zero-filled)
        self.declare_parameter("result_topic", "/sentinel/water_quality")
        self.declare_parameter("repo_root", os.path.expanduser("~/SENTINEL-STOCKHOLM"))
        self.declare_parameter("checkpoint", "")  # empty => random init (pipeline smoke)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("input_scale", 1.0)
        self.declare_parameter("sync_slop_sec", 0.05)

        gp = self.get_parameter
        self.input_scale = float(gp("input_scale").value)
        self.bridge = CvBridge()
        self.nir_sync = ApproxTimeSync(slop_sec=float(gp("sync_slop_sec").value))

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._load_model(gp("repo_root").value, gp("checkpoint").value, gp("device").value)

        self.pub = self.create_publisher(WaterQuality, gp("result_topic").value, qos)
        self.create_subscription(Image, gp("rgb_topic").value, self._on_rgb, qos)

        nir_topic = gp("nir_topic").value
        if nir_topic:
            self.create_subscription(Image, nir_topic, self._on_nir, qos)
            self.get_logger().info(f"NIR enabled on {nir_topic}")
        else:
            self.get_logger().info("NIR disabled — channel zero-filled (nir_present=false)")

        self.get_logger().info(f"inference up: RGB {gp('rgb_topic').value} -> {gp('result_topic').value}")

    def _load_model(self, repo_root, checkpoint, device):
        import torch  # lazy

        if repo_root and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from sentinel.models.waterdronenet import WaterDroneNet
        from sentinel.models.waterdronenet.waterdronenet import TARGET_PARAMS

        self.torch = torch
        self.target_params = list(TARGET_PARAMS)
        self.device = torch.device(device)
        self.model = WaterDroneNet().to(self.device).eval()
        if checkpoint and os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location=self.device)
            state = state.get("model", state) if isinstance(state, dict) else state
            self.model.load_state_dict(state, strict=False)
            self.get_logger().info(f"loaded checkpoint {checkpoint}")
        else:
            self.get_logger().warn("no checkpoint — random weights (pipeline smoke only)")

    def _on_nir(self, msg: Image):
        nir = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        self.nir_sync.add(_stamp_to_sec(msg.header.stamp), nir)

    def _on_rgb(self, msg: Image):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            nir = self.nir_sync.match(_stamp_to_sec(msg.header.stamp))
            chw, nir_present = assemble_four_band(
                rgb, nir, input_scale=self.input_scale
            )
            out = self._infer(chw)
            self._publish(msg.header, out, nir_present)
        except Exception as e:
            self.get_logger().error(f"inference failed: {e}")

    def _infer(self, chw: np.ndarray) -> dict:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(chw).unsqueeze(0).to(self.device)
            raw = self.model(x)
            return {
                "mu": raw["mu"][0].cpu().numpy(),
                "sigma": raw["sigma"][0].cpu().numpy(),
                "trust": float(torch.sigmoid(raw["trust_logit"][0]).item()),
            }

    def _publish(self, header, out: dict, nir_present: bool):
        msg = WaterQuality()
        msg.header = header
        msg.param_names = self.target_params
        msg.mu = [float(v) for v in out["mu"]]
        msg.sigma = [float(v) for v in out["sigma"]]
        msg.trust_score = out["trust"]
        msg.trust_flag = "green" if out["trust"] >= 0.7 else ("yellow" if out["trust"] >= 0.3 else "red")
        msg.nir_present = nir_present
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
