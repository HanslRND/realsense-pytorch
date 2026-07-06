import rclpy
from rclpy.executors import ExternalShutdownException

from realsense_yolov7.config import DemoConfig
from realsense_yolov7.ros_image_node import ImageDetectionNode


def run(config: DemoConfig) -> None:
    rclpy.init()
    node = ImageDetectionNode(config)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

