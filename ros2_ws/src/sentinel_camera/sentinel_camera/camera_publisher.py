"""SENTINEL drone RGB camera publisher.

Runs on the Raspberry Pi. Captures via a CameraBackend and publishes colour
``sensor_msgs/Image`` (bgr8) for the ground computer to consume. QoS and frame
conventions match CINE-Sensing so the two stacks interoperate.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Thread, Event
import time

from sentinel_camera.backends import make_backend


class RgbCameraPublisher(Node):
    def __init__(self):
        super().__init__("sentinel_rgb_camera")

        self.declare_parameter("backend", "auto")          # auto|libcamera|opencv|synthetic
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("image_width", 1152)
        self.declare_parameter("image_height", 648)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("topic", "/sentinel/camera/rgb/image_raw")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("shutter_us", 10000)
        self.declare_parameter("gain", 1.0)

        gp = self.get_parameter
        self.width = gp("image_width").value
        self.height = gp("image_height").value
        self.fps = float(gp("fps").value)
        self.frame_id = gp("frame_id").value

        self.bridge = CvBridge()
        self.shutdown = Event()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub = self.create_publisher(Image, gp("topic").value, qos)

        self.backend = make_backend(
            gp("backend").value,
            width=self.width, height=self.height, fps=self.fps,
            camera_id=gp("camera_id").value,
            shutter_us=gp("shutter_us").value, gain=float(gp("gain").value),
        )
        self.get_logger().info(
            f"RGB camera up: {type(self.backend).__name__} "
            f"{self.width}x{self.height}@{self.fps:.0f} -> {gp('topic').value}"
        )

        self.thread = Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        period = 1.0 / self.fps if self.fps > 0 else 0.0
        next_t = time.time()
        for frame in self.backend.frames():
            if self.shutdown.is_set() or not rclpy.ok():
                break
            stamp = self.get_clock().now().to_msg()
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                self.pub.publish(msg)
            except Exception as e:  # never let one bad frame kill the stream
                self.get_logger().warn(f"drop frame: {e}")
            next_t += period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time() + period

    def destroy_node(self):
        self.shutdown.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.backend.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = RgbCameraPublisher()
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
