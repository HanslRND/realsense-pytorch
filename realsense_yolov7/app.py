import time

import cv2

from realsense_yolov7.config import DemoConfig
from realsense_yolov7.realsense_stream import RealSenseColorStream
from realsense_yolov7.yolov7_detector import YoloV7Detector


def run(config: DemoConfig) -> None:
    detector = YoloV7Detector(
        config.yolov7_dir,
        config.weights,
        config.device,
        config.img_size,
        config.conf_thres,
        config.iou_thres,
        trace=not config.no_trace,
    )

    fps = 0.0
    frames_count = 0
    last_fps_time = time.perf_counter()

    window = "RealSense YOLOv7"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        with RealSenseColorStream(config.width, config.height, config.fps) as stream:
            for color_frame, depth_frame in stream.frames():
                annotated, count = detector.annotate(color_frame, depth_frame)
                frames_count += 1

                now = time.perf_counter()
                elapsed = now - last_fps_time
                if elapsed >= 1.0:
                    fps = frames_count / elapsed
                    frames_count = 0
                    last_fps_time = now

                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}  detections: {count}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cv2.destroyAllWindows()
