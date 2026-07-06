import sys
import time

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from realsense_yolov7.config import DemoConfig
from realsense_yolov7.yolov7_detector import YoloV7Detector


_ENCODINGS = {
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "16UC1": (np.uint16, 1),
    "mono16": (np.uint16, 1),
    "32FC1": (np.float32, 1),
}


def image_to_array(msg: Image) -> np.ndarray:
    if msg.encoding not in _ENCODINGS:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    dtype, channels = _ENCODINGS[msg.encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_items = msg.step // dtype.itemsize
    data = np.frombuffer(memoryview(msg.data), dtype=dtype)
    if channels == 1:
        image = data.reshape(msg.height, row_items)[:, : msg.width]
    else:
        row_pixels = row_items // channels
        image = data.reshape(msg.height, row_pixels, channels)[:, : msg.width, :]

    if msg.is_bigendian == (sys.byteorder == "little"):
        image = image.byteswap().view(image.dtype.newbyteorder())
    return np.ascontiguousarray(image)


def image_to_bgr(msg: Image) -> np.ndarray:
    image = image_to_array(msg)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding != "bgr8":
        raise ValueError(f"Unsupported color encoding: {msg.encoding}")
    return image


def bgr_to_image(image: np.ndarray, source: Image) -> Image:
    image = np.ascontiguousarray(image)
    msg = Image()
    msg.header = source.header
    msg.height = image.shape[0]
    msg.width = image.shape[1]
    msg.encoding = "bgr8"
    msg.is_bigendian = sys.byteorder == "big"
    msg.step = image.shape[1] * image.shape[2]
    msg.data = image.tobytes()
    return msg


class ImageDetectionNode(Node):
    def __init__(self, config: DemoConfig) -> None:
        super().__init__("realsense_yolov7")
        self.config = config
        self.get_logger().info(f"Running code from {__file__}")
        self.get_logger().info("Loading YOLOv7 model")
        self.detector = YoloV7Detector(
            config.yolov7_dir,
            config.weights,
            config.device,
            config.img_size,
            config.conf_thres,
            config.iou_thres,
            trace=not config.no_trace,
        )
        self.depth_frame = None
        self.fps = 0.0
        self.frames_count = 0
        self.color_count = 0
        self.depth_count = 0
        self.processed_count = 0
        self.last_fps_time = time.perf_counter()

        self.publisher = self.create_publisher(Image, config.processed_topic, qos_profile_sensor_data)
        self.depth_subscription = self.create_subscription(
            Image, config.depth_topic, self._on_depth, qos_profile_sensor_data
        )
        self.color_subscription = self.create_subscription(
            Image, config.color_topic, self._on_color, qos_profile_sensor_data
        )
        self.status_timer = self.create_timer(2.0, self._log_status)
        self.get_logger().info(
            f"Subscribed color={config.color_topic} depth={config.depth_topic}; publishing {config.processed_topic}"
        )
        self._log_graph()

    def _log_graph(self) -> None:
        for label, topic in (("color", self.config.color_topic), ("depth", self.config.depth_topic)):
            infos = self.get_publishers_info_by_topic(topic)
            if not infos:
                self.get_logger().warning(f"No publishers discovered for {label} topic {topic}")
                continue
            summary = ", ".join(
                f"{info.topic_type} reliability={info.qos_profile.reliability} durability={info.qos_profile.durability}"
                for info in infos
            )
            self.get_logger().info(f"Discovered {len(infos)} publisher(s) for {label} topic {topic}: {summary}")

    def _log_status(self) -> None:
        if not self.color_count or not self.depth_count:
            self._log_graph()
        if self.processed_count:
            self.get_logger().info(
                f"Published={self.processed_count} color={self.color_count} depth={self.depth_count} fps={self.fps:.1f}"
            )
            return
        missing = []
        if not self.color_count:
            missing.append("color")
        if not self.depth_count:
            missing.append("depth")
        if missing:
            self.get_logger().info(f"Waiting for {', '.join(missing)} image messages")
        else:
            self.get_logger().info("Images received, waiting for first processed frame")

    def _on_depth(self, msg: Image) -> None:
        try:
            self.depth_frame = image_to_array(msg)
            self.depth_count += 1
            if self.depth_count == 1:
                self.get_logger().info(f"First depth image: {msg.width}x{msg.height} {msg.encoding}")
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def _on_color(self, msg: Image) -> None:
        self.color_count += 1
        if self.color_count == 1:
            self.get_logger().info(f"First color image: {msg.width}x{msg.height} {msg.encoding}")
        if msg.width != self.config.expected_width or msg.height != self.config.expected_height:
            self.get_logger().warning(
                f"Skipping image with size {msg.width}x{msg.height}; "
                f"expected {self.config.expected_width}x{self.config.expected_height}"
            )
            return
        if self.depth_frame is None:
            return

        try:
            color_frame = image_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        annotated, count = self.detector.annotate(color_frame, self.depth_frame)
        self.processed_count += 1
        self.frames_count += 1
        now = time.perf_counter()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self.frames_count / elapsed
            self.frames_count = 0
            self.last_fps_time = now

        cv2.putText(
            annotated,
            f"FPS: {self.fps:.1f}  detections: {count}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        self.publisher.publish(bgr_to_image(annotated, msg))



