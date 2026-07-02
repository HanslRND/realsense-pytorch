from collections.abc import Iterator

import numpy as np
import pyrealsense2 as rs


class RealSenseColorStream:
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config = config

    def __enter__(self) -> "RealSenseColorStream":
        self.pipeline.start(self.config)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.pipeline.stop()

    def frames(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        while True:
            frame = self.pipeline.wait_for_frames()
            color_frame = frame.get_color_frame()
            depth_frame = frame.get_depth_frame()

            if color_frame or depth_frame:
                yield (np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data()))
