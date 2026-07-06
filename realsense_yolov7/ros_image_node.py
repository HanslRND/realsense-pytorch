import sys
import time

import cv2
import numpy as np
import rclpy
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


class ImageDetectionNode(Node):
    def __init__(self, config: DemoConfig) -> None:
        super().__init__("realsense_yolov7")
        self.config = config
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
        self.last_fps_time = time.perf_counter()
        self.window = "RealSense YOLOv7"
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)

        self.create_subscription(Image, config.depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(Image, config.color_topic, self._on_color, qos_profile_sensor_data)

    def _on_depth(self, msg: Image) -> None:
        try:
            self.depth_frame = image_to_array(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def _on_color(self, msg: Image) -> None:
        if msg.width != self.config.expected_width or msg.height != self.config.expected_height:
            self.get_logger().warning(
                f"Skipping image with size {msg.width}x{msg.height}; "
                f"expected {self.config.expected_width}x{self.config.expected_height}"
            )
            return
        if self.depth_frame is None:
            self.get_logger().warning("Skipping color image until first depth image arrives")
            return

        try:
            color_frame = image_to_bgr(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        annotated, count = self.detector.annotate(color_frame, self.depth_frame)
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
        cv2.imshow(self.window, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            rclpy.shutdown()

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()
