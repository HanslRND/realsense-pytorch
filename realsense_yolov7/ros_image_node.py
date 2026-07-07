import sys
import time
from threading import Event, Lock, Thread

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from realsense_yolov7.config import DemoConfig
from realsense_yolov7.tensorrt_detector import TensorRTYoloV7Detector
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


def image_key(msg: Image) -> tuple[int, int, str]:
    return msg.header.stamp.sec, msg.header.stamp.nanosec, msg.header.frame_id


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
        self.detector: YoloV7Detector | TensorRTYoloV7Detector | None = None
        self.depth_frame = None
        self.latest_color_msg: Image | None = None
        self.frame_lock = Lock()
        self.stop_event = Event()
        self.new_frame_event = Event()
        self.last_color_key = None
        self.last_depth_key = None
        self.last_processed_key = None
        self.last_bad_size = None
        self.fps = 0.0
        self.frames_count = 0
        self.processed_frames = 0
        self.last_fps_time = time.perf_counter()

        self.publisher = self.create_publisher(Image, config.processed_topic, qos_profile_sensor_data)
        self.depth_subscription = self.create_subscription(Image, config.depth_topic, self._on_depth, 10)
        self.color_subscription = self.create_subscription(Image, config.color_topic, self._on_color, 10)
        self.worker = Thread(target=self._process_latest_frames, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f"Subscribed color={config.color_topic} depth={config.depth_topic}; publishing {config.processed_topic}"
        )

    def destroy_node(self) -> bool:
        self.stop_event.set()
        self.new_frame_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        return super().destroy_node()

    def _get_detector(self):
        if self.detector is None:
            if self.config.engine:
                self.get_logger().info(f"Loading TensorRT engine {self.config.engine}")
                self.detector = TensorRTYoloV7Detector(
                    self.config.engine,
                    self.config.weights,
                    self.config.img_size,
                    self.config.conf_thres,
                    self.config.iou_thres,
                )
            else:
                self.get_logger().info("Loading YOLOv7 model")
                self.detector = YoloV7Detector(
                    self.config.yolov7_dir,
                    self.config.weights,
                    self.config.device,
                    self.config.img_size,
                    self.config.conf_thres,
                    self.config.iou_thres,
                    trace=not self.config.no_trace,
                )
        return self.detector

    def _on_depth(self, msg: Image) -> None:
        key = image_key(msg)
        if key == self.last_depth_key:
            return
        self.last_depth_key = key
        try:
            depth_frame = image_to_array(msg)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return

        with self.frame_lock:
            self.depth_frame = depth_frame

    def _on_color(self, msg: Image) -> None:
        key = image_key(msg)
        if key == self.last_color_key:
            return
        self.last_color_key = key
        with self.frame_lock:
            self.latest_color_msg = msg
        self.new_frame_event.set()

    def _process_latest_frames(self) -> None:
        while not self.stop_event.is_set():
            self.new_frame_event.wait(timeout=0.2)
            self.new_frame_event.clear()
            with self.frame_lock:
                color_msg = self.latest_color_msg
                depth_frame = self.depth_frame

            if color_msg is None or depth_frame is None:
                continue
            key = image_key(color_msg)
            if key == self.last_processed_key:
                continue
            self.last_processed_key = key

            if color_msg.width != self.config.expected_width or color_msg.height != self.config.expected_height:
                size = (color_msg.width, color_msg.height)
                if size != self.last_bad_size:
                    self.last_bad_size = size
                    self.get_logger().warning(
                        f"Skipping image with size {color_msg.width}x{color_msg.height}; "
                        f"expected {self.config.expected_width}x{self.config.expected_height}"
                    )
                continue

            try:
                detector = self._get_detector()
                frame_start = time.perf_counter()
                start = time.perf_counter()
                color_frame = image_to_bgr(color_msg)
                convert_ms = (time.perf_counter() - start) * 1000
                profile = self.config.profile_every > 0 and self.processed_frames % self.config.profile_every == 0
                annotated, count, timings = detector.annotate(color_frame, depth_frame, profile=profile)
            except ValueError as exc:
                self.get_logger().warning(str(exc))
                continue

            self.processed_frames += 1
            self.frames_count += 1
            now = time.perf_counter()
            elapsed = now - self.last_fps_time
            if elapsed >= 1.0:
                self.fps = self.frames_count / elapsed
                self.frames_count = 0
                self.last_fps_time = now

            start = time.perf_counter()
            cv2.putText(
                annotated,
                f"FPS: {self.fps:.1f}  detections: {count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            self.publisher.publish(bgr_to_image(annotated, color_msg))
            publish_ms = (time.perf_counter() - start) * 1000
            total_ms = (time.perf_counter() - frame_start) * 1000

            if profile:
                self.get_logger().info(
                    "profile "
                    f"total={total_ms:.1f}ms convert={convert_ms:.1f}ms "
                    f"pre={timings['preprocess_ms']:.1f}ms tensor={timings['tensor_ms']:.1f}ms "
                    f"infer={timings['inference_ms']:.1f}ms nms={timings['nms_ms']:.1f}ms "
                    f"draw={timings['draw_ms']:.1f}ms publish={publish_ms:.1f}ms"
                )

